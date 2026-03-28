open Tensor

type activation =
  | TanhAct
  | ReluAct
  | LinearAct

type layer = {
  w : t;      (* shape: [nout; nin] *)
  b : t;      (* shape: [nout; 1] *)
  act : activation;
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
  mutable m : Tensor.tensor list;
  mutable v : Tensor.tensor list;
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

let scalar_of_tensor x =
  match (x.shape, x.data) with
  | [], [| v |] -> v
  | [1; 1], [| v |] -> v
  | _ -> failwith "Expected scalar-like tensor"

let scalar_node_value x = scalar_of_tensor x.value
let scalar_node_grad x = scalar_of_tensor x.grad

let zeros_like_tensor x =
  { data = Array.make (Array.length x.data) 0.0; shape = x.shape }

let map_tensor f x =
  { data = Array.map f x.data; shape = x.shape }

let map2_tensor f a b =
  if a.shape <> b.shape then failwith "tensor helper: shape mismatch";
  let n = Array.length a.data in
  let out = Array.make n 0.0 in
  for i = 0 to n - 1 do
    out.(i) <- f a.data.(i) b.data.(i)
  done;
  { data = out; shape = a.shape }

let add_tensor_local a b = map2_tensor ( +. ) a b
let sub_tensor_local a b = map2_tensor ( -. ) a b
let mul_scalar_tensor_local x c = map_tensor (fun v -> v *. c) x

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
  low +. Random.float (high -. low)

let build_matrix rows cols f =
  let data =
    Array.init (rows * cols) (fun idx ->
        let r = idx / cols in
        let c = idx mod cols in
        f r c)
  in
  tensor_of_array data [rows; cols]

let make_weight_matrix ~nin ~nout ~act ~init =
  match init with
  | UniformInit ->
      parameter
        (build_matrix nout nin (fun _ _ -> rand_uniform (-1.0) 1.0))
  | XavierInit ->
      let bound = sqrt (6.0 /. float_of_int (nin + nout)) in
      parameter
        (build_matrix nout nin (fun _ _ -> rand_uniform (-.bound) bound))
  | HeInit ->
      let scale =
        match act with
        | ReluAct -> sqrt (2.0 /. float_of_int nin)
        | _ -> sqrt (1.0 /. float_of_int nin)
      in
      parameter
        (build_matrix nout nin (fun _ _ -> rand_uniform (-.scale) scale))

let make_bias_vector nout =
  parameter (build_matrix nout 1 (fun _ _ -> 0.0))

let make_layer nin nout act init =
  {
    w = make_weight_matrix ~nin ~nout ~act ~init;
    b = make_bias_vector nout;
    act;
  }

let layer_forward layer x =
  let z = add (matmul layer.w x) layer.b in
  apply_activation layer.act z

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
  List.fold_left
    (fun acc layer -> layer_forward layer acc)
    x
    model.layers

let layer_parameters l = [l.w; l.b]

let mlp_parameters m =
  List.flatten (List.map layer_parameters m.layers)

let zero_param_grads params =
  List.iter (fun p -> p.grad <- zeros (shape p.value)) params

let l2_penalty params =
  let penalties =
    List.filter (fun p -> p.is_parameter) params
    |> List.map square
    |> List.map sum
  in
  match penalties with
  | [] -> of_float 0.0
  | _ -> sum_tensors penalties

let col_vector_of_scalar x =
  of_matrix [[x]]

let loss_on_data ?(weight_decay = 0.0) model data =
  let losses =
    List.map
      (fun (x_val, y_val) ->
        let x = col_vector_of_scalar x_val in
        let y_true = col_vector_of_scalar y_val in
        let pred = mlp_forward model x in
        let diff = sub pred y_true in
        sum (square diff))
      data
  in
  let data_loss = sum_tensors losses in
  if weight_decay <= 0.0 then data_loss
  else
    let params = mlp_parameters model in
    let reg = l2_penalty params in
    add data_loss (mulf reg weight_decay)

let predict model x_val =
  let x = col_vector_of_scalar x_val in
  let y = mlp_forward model x in
  scalar_node_value y

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
  let rows, cols =
    match layer.w.value.shape with
    | [r; c] -> (r, c)
    | _ -> failwith "Expected rank-2 weight matrix"
  in
  let bias_params = Array.length layer.b.value.data in
  let weight_params = Array.length layer.w.value.data in
  Printf.printf
    "Layer %d: W[%d x %d] | b[%d x 1] | %s | %d parameter(s)\n"
    (idx + 1)
    rows
    cols
    rows
    (activation_name layer.act)
    (weight_params + bias_params)

let print_mlp_summary model =
  let total_params =
    mlp_parameters model
    |> List.fold_left (fun acc p -> acc + Array.length p.value.data) 0
  in
  Printf.printf "Vectorized MLP Summary\n";
  List.iteri print_layer_summary model.layers;
  Printf.printf "Total parameters: %d\n\n" total_params

let print_tensor label x =
  Printf.printf "%s -> value = %s | grad = %s\n"
    label
    (tensor_to_string x.value)
    (tensor_to_string x.grad)

let demo_new_ops () =
  Printf.printf "\n=== Primitive op showcase: exp / log / pow / sigmoid / matmul ===\n";

  let x = of_matrix [[2.0]] in
  let y = of_matrix [[3.0]] in

  let exp_x = exp x in
  let log_y = log y in
  let x_sq = pow x 2.0 in
  let sig_x = sigmoid x in
  let mm =
    matmul
      (of_matrix [[1.0; 2.0]; [3.0; 4.0]])
      (of_matrix [[5.0]; [6.0]])
  in

  print_tensor "x" x;
  print_tensor "y" y;
  print_tensor "exp(x)" exp_x;
  print_tensor "log(y)" log_y;
  print_tensor "pow(x, 2.0)" x_sq;
  print_tensor "sigmoid(x)" sig_x;
  print_tensor "matmul demo" mm;

  let y_plus_one = add y (of_matrix [[1.0]]) in
  let numerator = add (pow x 2.0) (exp y) in
  let denominator = log y_plus_one in
  let quotient = div numerator denominator in
  let out = sigmoid quotient in

  backward out;

  print_tensor "output" out;
  print_tensor "x after backward" x;
  print_tensor "y after backward" y;
  Printf.printf "\n"

let gradient_check
    ?(eps = 1e-5)
    ?(num_to_check = 8)
    ?(weight_decay = 0.0)
    model
    data =
  let params = mlp_parameters model in

  zero_param_grads params;
  let loss = loss_on_data ~weight_decay model data in
  backward loss;

  let entries =
    params
    |> List.mapi (fun pi p ->
           Array.to_list
             (Array.mapi (fun idx _ -> (pi, p, idx)) p.value.data))
    |> List.flatten
  in

  let n = min num_to_check (List.length entries) in

  Printf.printf
    "Gradient check (first %d scalar parameter(s), eps = %.1e)\n"
    n eps;

  let max_abs_err = ref 0.0 in
  let max_rel_err = ref 0.0 in

  for i = 0 to n - 1 do
    let (pi, p, idx) = List.nth entries i in
    let orig = p.value.data.(idx) in
    let analytical_i = p.grad.data.(idx) in

    p.value.data.(idx) <- orig +. eps;
    let loss_plus = scalar_node_value (loss_on_data ~weight_decay model data) in

    p.value.data.(idx) <- orig -. eps;
    let loss_minus = scalar_node_value (loss_on_data ~weight_decay model data) in

    p.value.data.(idx) <- orig;

    let numerical = (loss_plus -. loss_minus) /. (2.0 *. eps) in
    let abs_err = abs_float (numerical -. analytical_i) in
    let denom =
      max 1e-12 (max (abs_float numerical) (abs_float analytical_i))
    in
    let rel_err = abs_err /. denom in

    if abs_err > !max_abs_err then max_abs_err := abs_err;
    if rel_err > !max_rel_err then max_rel_err := rel_err;

    Printf.printf
      "param%d[%d] | analytical = %.8f | numerical = %.8f | abs err = %.3e | rel err = %.3e\n"
      pi idx analytical_i numerical abs_err rel_err
  done;

  Printf.printf
    "Max abs err: %.3e | Max rel err: %.3e\n\n"
    !max_abs_err !max_rel_err

let make_adam_state params =
  {
    m = List.map (fun p -> zeros_like_tensor p.value) params;
    v = List.map (fun p -> zeros_like_tensor p.value) params;
    beta1 = 0.9;
    beta2 = 0.999;
    eps = 1e-8;
    t = 0;
  }

let grad_global_norm params =
  let sq_sum =
    List.fold_left
      (fun acc p ->
        acc +.
        Array.fold_left (fun s g -> s +. (g *. g)) 0.0 p.grad.data)
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
      List.iter
        (fun p ->
          p.grad <- mul_scalar_tensor_local p.grad scale)
        params

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
        p.value <- sub_tensor_local p.value (mul_scalar_tensor_local p.grad lr))
    params

let adam_step params state lr =
  state.t <- state.t + 1;
  let t_float = float_of_int state.t in

  List.iteri
    (fun i p ->
      if p.is_parameter then begin
        let g = p.grad in

        let m_i =
          add_tensor_local
            (mul_scalar_tensor_local (List.nth state.m i) state.beta1)
            (mul_scalar_tensor_local g (1.0 -. state.beta1))
        in

        let v_i =
          add_tensor_local
            (mul_scalar_tensor_local (List.nth state.v i) state.beta2)
            (mul_scalar_tensor_local
               (map_tensor (fun x -> x *. x) g)
               (1.0 -. state.beta2))
        in

        state.m <- List.mapi (fun j x -> if i = j then m_i else x) state.m;
        state.v <- List.mapi (fun j x -> if i = j then v_i else x) state.v;

        let m_hat =
          mul_scalar_tensor_local
            m_i
            (1.0 /. (1.0 -. (state.beta1 ** t_float)))
        in

        let v_hat =
          mul_scalar_tensor_local
            v_i
            (1.0 /. (1.0 -. (state.beta2 ** t_float)))
        in

        let denom = map_tensor (fun x -> sqrt x +. state.eps) v_hat in
        let step_tensor =
          map2_tensor (fun m d -> lr *. m /. d) m_hat denom
        in

        p.value <- sub_tensor_local p.value step_tensor
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
    Printf.printf
      "optimizer = %s | init = %s | weight_decay = %.6f | grad_clip = %.4f\n"
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
    let loss_scalar = scalar_node_value total_loss in

    if verbose && i mod print_every = 0 then
      Printf.printf
        "iter %d | lr = %.6f | loss = %.6f | grad_norm = %.6f\n"
        i lr_now loss_scalar grad_norm_before;

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
            append_csv_row
              oc i lr_now loss_scalar train_mae test_mae grad_norm_before
    end
  done;

  begin
    match csv_channel with
    | None -> ()
    | Some oc -> close_out oc
  end;

  let final_loss = scalar_node_value (loss_on_data ~weight_decay model data) in
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

  Printf.printf "Initial parameter shapes:\n";
  List.iteri
    (fun i p ->
      Printf.printf "p%d shape = %s\n" i (tensor_to_string p.value))
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
    "8x8_tanh_sgd_vec"
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
    "8x8_tanh_adam_vec"
    [1; 8; 8; 1]
    TanhAct
    Adam
    XavierInit
    0.01
    2500
    false
    train_data;

  add_result
    ~weight_decay:0.0002
    ~grad_clip:5.0
    ~schedule:(StepDecay { step_size = 1000; gamma = 0.5 })
    ~log_csv:true
    "8x8_tanh_adam_vec_expanded"
    [1; 8; 8; 1]
    TanhAct
    Adam
    XavierInit
    0.01
    2500
    false
    expanded_train_data;

  print_experiment_table (List.rev !results)