open Tensor

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

type init_kind =
  | UniformInit
  | XavierInit
  | HeInit

type lr_schedule =
  | ConstantLR
  | StepDecay of {
      step_size : int;
      gamma : float;
    }

type experiment_result = {
  name : string;
  optimizer : string;
  hidden_act : string;
  init_name : string;
  final_lr : float;
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

let activation_name = function
  | TanhAct -> "Tanh"
  | ReluAct -> "Relu"
  | LinearAct -> "Linear"

let optimizer_name = function
  | SGD -> "SGD"
  | Adam -> "Adam"

let init_name = function
  | UniformInit -> "Uniform"
  | XavierInit -> "Xavier"
  | HeInit -> "He"

let apply_activation act x =
  match act with
  | TanhAct -> tanh x
  | ReluAct -> relu x
  | LinearAct -> x

let sum_tensors ts =
  match ts with
  | [] -> failwith "Cannot sum empty tensor list"
  | first :: rest -> List.fold_left add first rest

let rand_uniform low high =
  low +. (Random.float (high -. low))

let make_weight ~nin ~act ~init =
  match init with
  | UniformInit ->
      parameter (rand_uniform (-1.0) 1.0)
  | XavierInit ->
      let bound = sqrt (6.0 /. float_of_int (nin + 1)) in
      parameter (rand_uniform (-.bound) bound)
  | HeInit ->
      let scale =
        match act with
        | ReluAct -> sqrt (2.0 /. float_of_int nin)
        | _ -> sqrt (1.0 /. float_of_int nin)
      in
      parameter (rand_uniform (-.scale) scale)

let make_bias () =
  parameter 0.0

let make_neuron nin act init =
  {
    w = List.init nin (fun _ -> make_weight ~nin ~act ~init);
    b = make_bias ();
    act;
  }

let neuron_forward n xs =
  let wx_terms = List.map2 mul n.w xs in
  let preact = add (sum_tensors wx_terms) n.b in
  apply_activation n.act preact

let make_layer nin nout act init =
  {
    neurons = List.init nout (fun _ -> make_neuron nin act init);
  }

let layer_forward layer xs =
  List.map (fun n -> neuron_forward n xs) layer.neurons

let rec pairwise xs =
  match xs with
  | a :: (b :: _ as rest) -> (a, b) :: pairwise rest
  | _ -> []

let make_mlp dims hidden_act output_act init =
  match dims with
  | [] | [_] -> failwith "MLP needs at least input and output dimensions"
  | _ ->
      let layer_specs = pairwise dims in
      let last_idx = List.length layer_specs - 1 in
      let layers =
        List.mapi
          (fun i (nin, nout) ->
            let act = if i = last_idx then output_act else hidden_act in
            make_layer nin nout act init)
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

let zero_param_grads params =
  List.iter (fun p -> p.grad <- 0.0) params

let l2_penalty params =
  let penalties =
    List.filter (fun p -> p.is_parameter) params
    |> List.map (fun p -> make (p.value *. p.value))
  in
  match penalties with
  | [] -> make 0.0
  | _ -> sum_tensors penalties

let loss_on_data ?(weight_decay = 0.0) model data =
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
  let data_loss = sum_tensors losses in
  if weight_decay <= 0.0 then data_loss
  else
    let params = mlp_parameters model in
    let reg = l2_penalty params in
    add data_loss (mul (make weight_decay) reg)

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

let print_tensor label x =
  Printf.printf "%s -> value = %.6f | grad = %.6f\n" label x.value x.grad

let demo_new_ops () =
  Printf.printf "\n=== Primitive op showcase: exp / log / pow / sigmoid ===\n";

  let x = make 2.0 in
  let y = make 3.0 in

  let exp_x = exp x in
  let log_y = log y in
  let x_sq = pow x 2.0 in
  let sig_x = sigmoid x in

  print_tensor "x" x;
  print_tensor "y" y;
  print_tensor "exp(x)" exp_x;
  print_tensor "log(y)" log_y;
  print_tensor "pow(x, 2.0)" x_sq;
  print_tensor "sigmoid(x)" sig_x;

  Printf.printf "\nComposite scalar expression:\n";
  Printf.printf "f(x, y) = sigmoid((x^2 + exp(y)) / log(y + 1))\n";

  let y_plus_one = add y (make 1.0) in
  let numerator = add (pow x 2.0) (exp y) in
  let denominator = log y_plus_one in
  let quotient = div numerator denominator in
  let out = sigmoid quotient in

  backward out;

  print_tensor "output" out;
  print_tensor "x after backward" x;
  print_tensor "y after backward" y;
  Printf.printf "\n"

let gradient_check ?(eps = 1e-5) ?(num_to_check = 6) ?(weight_decay = 0.0) model data =
  let params = mlp_parameters model in

  zero_param_grads params;
  let loss = loss_on_data ~weight_decay model data in
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
    let loss_plus = (loss_on_data ~weight_decay model data).value in

    p.value <- orig -. eps;
    let loss_minus = (loss_on_data ~weight_decay model data).value in

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

let grad_global_norm params =
  let sq_sum =
    List.fold_left
      (fun acc p -> acc +. (p.grad *. p.grad))
      0.0
      params
  in
  sqrt sq_sum

let clip_grad_norm params max_norm =
  if max_norm <= 0.0 then ()
  else
    let norm = grad_global_norm params in
    if norm > max_norm then
      let scale = max_norm /. (norm +. 1e-12) in
      List.iter (fun p -> p.grad <- p.grad *. scale) params

let current_lr schedule base_lr iter =
  match schedule with
  | ConstantLR -> base_lr
  | StepDecay { step_size; gamma } ->
      let steps = iter / step_size in
      base_lr *. (gamma ** float_of_int steps)

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

let write_csv_header oc =
  output_string oc "iter,lr,train_loss,train_mae,test_mae,grad_norm\n"

let append_csv_row oc iter lr train_loss train_mae test_mae grad_norm =
  Printf.fprintf oc "%d,%.8f,%.8f,%.8f,%.8f,%.8f\n"
    iter lr train_loss train_mae test_mae grad_norm

let train_model
    ?(verbose = false)
    ?(print_every = 250)
    ?(weight_decay = 0.0)
    ?(grad_clip = 0.0)
    ?(schedule = ConstantLR)
    ?(log_csv = false)
    ~name
    ~dims
    ~hidden_act
    ~output_act
    ~optimizer
    ~init
    ~lr
    ~iters
    ~data
    ~test_data
    () =
  let model = make_mlp dims hidden_act output_act init in
  let params = mlp_parameters model in
  let adam_state = make_adam_state params in

  let csv_channel =
    if log_csv then begin
      let filename = name ^ "_metrics.csv" in
      let oc = open_out filename in
      write_csv_header oc;
      Some oc
    end else
      None
  in

  if verbose then begin
    Printf.printf "\n=== %s ===\n" name;
    Printf.printf "optimizer = %s | init = %s | weight_decay = %.6f | grad_clip = %.4f\n"
      (optimizer_name optimizer)
      (init_name init)
      weight_decay
      grad_clip;
    print_mlp_summary model
  end;

  for i = 0 to iters - 1 do
    zero_param_grads params;
    let total_loss = loss_on_data ~weight_decay model data in
    backward total_loss;

    let grad_norm_before = grad_global_norm params in
    clip_grad_norm params grad_clip;

    let lr_now = current_lr schedule lr i in

    if verbose && i mod print_every = 0 then
      Printf.printf
        "iter %d | lr = %.6f | loss = %.6f | grad_norm = %.6f\n"
        i lr_now total_loss.value grad_norm_before;

    begin
      match optimizer with
      | SGD -> sgd_step params lr_now
      | Adam -> adam_step params adam_state lr_now
    end;

    begin
      match csv_channel with
      | None -> ()
      | Some oc ->
          if i mod print_every = 0 || i = iters - 1 then
            let train_mae = mae_on_data model data in
            let test_mae = mae_on_data model test_data in
            append_csv_row oc i lr_now total_loss.value train_mae test_mae grad_norm_before
    end
  done;

  begin
    match csv_channel with
    | None -> ()
    | Some oc -> close_out oc
  end;

  let final_loss = (loss_on_data ~weight_decay model data).value in
  let mae_train = mae_on_data model data in
  let mae_test = mae_on_data model test_data in
  let pred_neg3 = predict model (-3.0) in
  let pred_pos3 = predict model 3.0 in
  let final_lr = current_lr schedule lr (iters - 1) in

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
    optimizer = optimizer_name optimizer;
    hidden_act = activation_name hidden_act;
    init_name = init_name init;
    final_lr;
    final_train_loss = final_loss;
    mae_train;
    mae_test;
    pred_neg3;
    pred_pos3;
  }, model)

let print_experiment_table results =
  Printf.printf "\nExperiment Comparison\n";
  Printf.printf
    "%-26s %-8s %-8s %-8s %-10s %-14s %-12s %-12s %-12s %-12s\n"
    "name" "opt" "act" "init" "final_lr" "train_loss" "train_mae" "test_mae" "pred(-3)" "pred(3)";
  Printf.printf
    "%-26s %-8s %-8s %-8s %-10s %-14s %-12s %-12s %-12s %-12s\n"
    "--------------------------"
    "--------"
    "--------"
    "--------"
    "----------"
    "--------------"
    "------------"
    "------------"
    "------------"
    "------------";

  List.iter
    (fun r ->
      Printf.printf
        "%-26s %-8s %-8s %-8s %-10.6f %-14.6f %-12.6f %-12.6f %-12.4f %-12.4f\n"
        r.name
        r.optimizer
        r.hidden_act
        r.init_name
        r.final_lr
        r.final_train_loss
        r.mae_train
        r.mae_test
        r.pred_neg3
        r.pred_pos3)
    results

let () =
  Random.self_init ();

  demo_new_ops ();

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

  let showcase_model = make_mlp [1; 8; 8; 1] TanhAct LinearAct XavierInit in
  print_mlp_summary showcase_model;

  Printf.printf "Initial parameters:\n";
  List.iteri
    (fun i p -> Printf.printf "p%d = %.4f\n" i p.value)
    (mlp_parameters showcase_model);
  Printf.printf "\n";

  gradient_check showcase_model train_data;

  let results = ref [] in

  let add_result
      ?(weight_decay = 0.0)
      ?(grad_clip = 0.0)
      ?(schedule = ConstantLR)
      ?(log_csv = false)
      name dims hidden_act optimizer init lr iters verbose data =
    let result, _model =
      train_model
        ~verbose
        ~weight_decay
        ~grad_clip
        ~schedule
        ~log_csv
        ~name
        ~dims
        ~hidden_act
        ~output_act:LinearAct
        ~optimizer
        ~init
        ~lr
        ~iters
        ~data
        ~test_data
        ()
    in
    results := result :: !results
  in

  add_result
    "8x8_tanh_sgd"
    [1; 8; 8; 1]
    TanhAct
    SGD
    XavierInit
    0.001
    5000
    true
    train_data;

  add_result
    ~weight_decay:0.0005
    ~grad_clip:5.0
    ~schedule:(StepDecay { step_size = 1000; gamma = 0.5 })
    ~log_csv:true
    "8x8_tanh_adam"
    [1; 8; 8; 1]
    TanhAct
    Adam
    XavierInit
    0.01
    2500
    false
    train_data;

  add_result
    ~weight_decay:0.0005
    ~grad_clip:5.0
    ~schedule:(StepDecay { step_size = 1000; gamma = 0.5 })
    ~log_csv:true
    "8x8_relu_adam"
    [1; 8; 8; 1]
    ReluAct
    Adam
    HeInit
    0.005
    2500
    false
    train_data;

  add_result
    ~weight_decay:0.0005
    ~grad_clip:5.0
    ~log_csv:true
    "4x4_tanh_adam"
    [1; 4; 4; 1]
    TanhAct
    Adam
    XavierInit
    0.01
    2000
    false
    train_data;

  add_result
    ~weight_decay:0.0002
    ~grad_clip:5.0
    ~schedule:(StepDecay { step_size = 1000; gamma = 0.5 })
    ~log_csv:true
    "8x8_tanh_adam_expanded"
    [1; 8; 8; 1]
    TanhAct
    Adam
    XavierInit
    0.01
    2500
    false
    expanded_train_data;

  print_experiment_table (List.rev !results)
