open Minitorch_ocaml.Tensor

type activation =
  | TanhAct
  | ReluAct
  | LinearAct

type neuron = {
  w : t list;
  b : t;
  act : activation;
}

type layer = {
  neurons : neuron list;
}

type mlp = {
  layers : layer list;
}

type optimizer_kind =
  | SGD
  | Adam

type experiment_result = {
  name : string;
  optimizer : string;
  hidden_act : string;
  final_train_loss : float;
  mae_train : float;
  mae_test : float;
  pred_neg3 : float;
  pred_pos3 : float;
}

type adam_state = {
  m : float array;
  v : float array;
  beta1 : float;
  beta2 : float;
  eps : float;
  mutable t : int;
}

let rand_param () = parameter ((Random.float 2.0) -. 1.0)

let sum_tensors ts =
  match ts with
  | [] -> failwith "Cannot sum empty tensor list"
  | first :: rest -> List.fold_left add first rest

let activation_name = function
  | TanhAct -> "Tanh"
  | ReluAct -> "Relu"
  | LinearAct -> "Linear"

let apply_activation act x =
  match act with
  | TanhAct -> tanh x
  | ReluAct -> relu x
  | LinearAct -> x

let make_neuron nin act =
  {
    w = List.init nin (fun _ -> rand_param ());
    b = parameter 0.0;
    act;
  }

let neuron_forward n xs =
  let wx_terms = List.map2 mul n.w xs in
  let preact = add (sum_tensors wx_terms) n.b in
  apply_activation n.act preact

let make_layer nin nout act =
  {
    neurons = List.init nout (fun _ -> make_neuron nin act);
  }

let layer_forward layer xs =
  List.map (fun n -> neuron_forward n xs) layer.neurons

let rec pairwise xs =
  match xs with
  | a :: (b :: _ as rest) -> (a, b) :: pairwise rest
  | _ -> []

let make_mlp dims hidden_act output_act =
  match dims with
  | [] | [_] -> failwith "MLP needs at least input and output dimensions"
  | _ ->
      let layer_specs = pairwise dims in
      let last_idx = List.length layer_specs - 1 in
      let layers =
        List.mapi
          (fun i (nin, nout) ->
            let act = if i = last_idx then output_act else hidden_act in
            make_layer nin nout act)
          layer_specs
      in
      { layers }

let mlp_forward model x =
  let outputs =
    List.fold_left
      (fun xs layer -> layer_forward layer xs)
      [x]
      model.layers
  in
  match outputs with
  | [out] -> out
  | _ -> failwith "Expected exactly one output"

let neuron_parameters n =
  n.w @ [n.b]

let layer_parameters l =
  List.flatten (List.map neuron_parameters l.neurons)

let mlp_parameters m =
  List.flatten (List.map layer_parameters m.layers)

let square x =
  x *. x

let zero_param_grads params =
  List.iter (fun p -> p.grad <- 0.0) params

let loss_on_data model data =
  let losses =
    List.map
      (fun (x_val, y_val) ->
        let x = make x_val in
        let y_true = make y_val in
        let pred = mlp_forward model x in
        let diff = sub pred y_true in
        mul diff diff)
      data
  in
  sum_tensors losses

let predict model x_val =
  let x = make x_val in
  (mlp_forward model x).value

let mae_on_data model data =
  let errs =
    List.map
      (fun (x_val, y_val) ->
        abs_float (predict model x_val -. y_val))
      data
  in
  let total = List.fold_left ( +. ) 0.0 errs in
  total /. float_of_int (List.length errs)

let print_layer_summary idx layer =
  match layer.neurons with
  | [] ->
      Printf.printf "Layer %d: 0 neurons\n" (idx + 1)
  | first_neuron :: _ ->
      let num_neurons = List.length layer.neurons in
      let act = activation_name first_neuron.act in
      let inputs_per_neuron = List.length first_neuron.w in
      let params_in_layer = List.length (layer_parameters layer) in
      Printf.printf
        "Layer %d: %d neurons | %d input(s) each | %s | %d parameter(s)\n"
        (idx + 1)
        num_neurons
        inputs_per_neuron
        act
        params_in_layer

let print_mlp_summary model =
  let total_params = List.length (mlp_parameters model) in
  Printf.printf "MLP Summary\n";
  List.iteri print_layer_summary model.layers;
  Printf.printf "Total parameters: %d\n\n" total_params

let gradient_check ?(eps = 1e-5) ?(num_to_check = 6) model data =
  let params = mlp_parameters model in

  zero_param_grads params;
  let loss = loss_on_data model data in
  backward loss;

  let analytical = List.map (fun p -> p.grad) params in
  let n = min num_to_check (List.length params) in

  Printf.printf "Gradient check (first %d parameter(s), eps = %.1e)\n" n eps;

  let max_abs_err = ref 0.0 in
  let max_rel_err = ref 0.0 in

  for i = 0 to n - 1 do
    let p = List.nth params i in
    let orig = p.value in

    p.value <- orig +. eps;
    let loss_plus = (loss_on_data model data).value in

    p.value <- orig -. eps;
    let loss_minus = (loss_on_data model data).value in

    p.value <- orig;

    let numerical = (loss_plus -. loss_minus) /. (2.0 *. eps) in
    let analytical_i = List.nth analytical i in
    let abs_err = abs_float (numerical -. analytical_i) in
    let denom =
      max 1e-12 (max (abs_float numerical) (abs_float analytical_i))
    in
    let rel_err = abs_err /. denom in

    if abs_err > !max_abs_err then max_abs_err := abs_err;
    if rel_err > !max_rel_err then max_rel_err := rel_err;

    Printf.printf
      "p%d | analytical = %.8f | numerical = %.8f | abs err = %.3e | rel err = %.3e\n"
      i analytical_i numerical abs_err rel_err
  done;

  Printf.printf
    "Max abs err: %.3e | Max rel err: %.3e\n\n"
    !max_abs_err !max_rel_err

let make_adam_state params =
  let n = List.length params in
  {
    m = Array.make n 0.0;
    v = Array.make n 0.0;
    beta1 = 0.9;
    beta2 = 0.999;
    eps = 1e-8;
    t = 0;
  }

let sgd_step params lr =
  List.iter
    (fun p ->
      if p.is_parameter then
        p.value <- p.value -. (lr *. p.grad))
    params

let adam_step params state lr =
  state.t <- state.t + 1;
  let t_float = float_of_int state.t in

  List.iteri
    (fun i p ->
      if p.is_parameter then begin
        let g = p.grad in
        state.m.(i) <- (state.beta1 *. state.m.(i)) +. ((1.0 -. state.beta1) *. g);
        state.v.(i) <- (state.beta2 *. state.v.(i)) +. ((1.0 -. state.beta2) *. g *. g);

        let m_hat = state.m.(i) /. (1.0 -. (state.beta1 ** t_float)) in
        let v_hat = state.v.(i) /. (1.0 -. (state.beta2 ** t_float)) in

        p.value <- p.value -. (lr *. m_hat /. (sqrt v_hat +. state.eps))
      end)
    params

let train_model
    ?(verbose = false)
    ?(print_every = 250)
    ~name
    ~dims
    ~hidden_act
    ~output_act
    ~optimizer
    ~lr
    ~iters
    ~data
    ~test_data
    () =
  let model = make_mlp dims hidden_act output_act in
  let params = mlp_parameters model in
  let adam_state = make_adam_state params in

  if verbose then begin
    Printf.printf "\n=== %s ===\n" name;
    print_mlp_summary model
  end;

  for i = 0 to iters - 1 do
    zero_param_grads params;
    let total_loss = loss_on_data model data in
    backward total_loss;

    if verbose && i mod print_every = 0 then
      Printf.printf "iter %d | loss = %.6f\n" i total_loss.value;

    match optimizer with
    | SGD -> sgd_step params lr
    | Adam -> adam_step params adam_state lr
  done;

  let final_loss = (loss_on_data model data).value in
  let mae_train = mae_on_data model data in
  let mae_test = mae_on_data model test_data in
  let pred_neg3 = predict model (-3.0) in
  let pred_pos3 = predict model 3.0 in

  let optimizer_name =
    match optimizer with
    | SGD -> "SGD"
    | Adam -> "Adam"
  in

  if verbose then begin
    Printf.printf "\nPredictions on training data:\n";
    List.iter
      (fun (x_val, y_val) ->
        let pred = predict model x_val in
        Printf.printf "x = %.1f | pred = %.4f | true = %.1f\n"
          x_val pred y_val)
      data;

    Printf.printf "\nGeneralization test:\n";
    List.iter
      (fun (x_val, y_val) ->
        let pred = predict model x_val in
        Printf.printf "x = %.1f | pred = %.4f | true = %.4f\n"
          x_val pred y_val)
      test_data
  end;

  ({
    name;
    optimizer = optimizer_name;
    hidden_act = activation_name hidden_act;
    final_train_loss = final_loss;
    mae_train;
    mae_test;
    pred_neg3;
    pred_pos3;
  }, model)

let print_experiment_table results =
  Printf.printf "\nExperiment Comparison\n";
  Printf.printf
    "%-24s %-8s %-8s %-14s %-12s %-12s %-12s %-12s\n"
    "name" "opt" "act" "train_loss" "train_mae" "test_mae" "pred(-3)" "pred(3)";
  Printf.printf
    "%-24s %-8s %-8s %-14s %-12s %-12s %-12s %-12s\n"
    "------------------------"
    "--------"
    "--------"
    "--------------"
    "------------"
    "------------"
    "------------"
    "------------";

  List.iter
    (fun r ->
      Printf.printf
        "%-24s %-8s %-8s %-14.6f %-12.6f %-12.6f %-12.4f %-12.4f\n"
        r.name
        r.optimizer
        r.hidden_act
        r.final_train_loss
        r.mae_train
        r.mae_test
        r.pred_neg3
        r.pred_pos3)
    results

let () =
  Random.self_init ();

  let train_data =
    [
      (-2.0, 4.0);
      (-1.0, 1.0);
      (0.0, 0.0);
      (1.0, 1.0);
      (2.0, 4.0);
    ]
  in

  let expanded_train_data =
    [
      (-3.0, 9.0);
      (-2.0, 4.0);
      (-1.0, 1.0);
      (0.0, 0.0);
      (1.0, 1.0);
      (2.0, 4.0);
      (3.0, 9.0);
    ]
  in

  let test_data =
    [
      (-2.5, 6.25);
      (-1.5, 2.25);
      (-0.5, 0.25);
      (0.5, 0.25);
      (1.5, 2.25);
      (2.5, 6.25);
    ]
  in

  let showcase_model = make_mlp [1; 8; 8; 1] TanhAct LinearAct in
  print_mlp_summary showcase_model;

  Printf.printf "Initial parameters:\n";
  List.iteri
    (fun i p -> Printf.printf "p%d = %.4f\n" i p.value)
    (mlp_parameters showcase_model);
  Printf.printf "\n";

  gradient_check showcase_model train_data;

  let results = ref [] in

  let add_result name dims hidden_act optimizer lr iters verbose =
    let result, _model =
      train_model
        ~verbose
        ~name
        ~dims
        ~hidden_act
        ~output_act:LinearAct
        ~optimizer
        ~lr
        ~iters
        ~data:train_data
        ~test_data
        ()
    in
    results := result :: !results
  in

  let add_result_with_data name dims hidden_act optimizer lr iters verbose data =
    let result, _model =
      train_model
        ~verbose
        ~name
        ~dims
        ~hidden_act
        ~output_act:LinearAct
        ~optimizer
        ~lr
        ~iters
        ~data
        ~test_data
        ()
    in
    results := result :: !results
  in

  add_result "8x8_tanh_sgd" [1; 8; 8; 1] TanhAct SGD 0.001 5000 true;
  add_result "8x8_tanh_adam" [1; 8; 8; 1] TanhAct Adam 0.01 2000 false;
  add_result "8x8_relu_adam" [1; 8; 8; 1] ReluAct Adam 0.005 2500 false;
  add_result "4x4_tanh_adam" [1; 4; 4; 1] TanhAct Adam 0.01 2000 false;

  add_result_with_data
    "8x8_tanh_adam_expanded"
    [1; 8; 8; 1]
    TanhAct
    Adam
    0.01
    2000
    false
    expanded_train_data;

  print_experiment_table (List.rev !results)