open Minitorch_ocaml.Tensor

type neuron = {
  w : t list;
  b : t;
  nonlin : bool;
}

type layer = {
  neurons : neuron list;
}

type mlp = {
  l1 : layer;
  l2 : layer;
}

let rand_param () = parameter ((Random.float 2.0) -. 1.0)

let sum_tensors ts =
  match ts with
  | [] -> failwith "Cannot sum empty tensor list"
  | first :: rest -> List.fold_left add first rest

let make_neuron nin nonlin =
  {
    w = List.init nin (fun _ -> rand_param ());
    b = parameter 0.1;
    nonlin;
  }

let neuron_forward n xs =
  let wx_terms = List.map2 mul n.w xs in
  let act = add (sum_tensors wx_terms) n.b in
  if n.nonlin then tanh act else act

let make_layer nin nout nonlin =
  {
    neurons = List.init nout (fun _ -> make_neuron nin nonlin);
  }

let layer_forward layer xs =
  List.map (fun n -> neuron_forward n xs) layer.neurons

let make_mlp () =
  {
    l1 = make_layer 1 4 true;
    l2 = make_layer 4 1 false;
  }

let mlp_forward model x =
  let h = layer_forward model.l1 [x] in
  match layer_forward model.l2 h with
  | [out] -> out
  | _ -> failwith "Expected one output"

let neuron_parameters n =
  n.w @ [n.b]

let layer_parameters l =
  List.flatten (List.map neuron_parameters l.neurons)

let mlp_parameters m =
  layer_parameters m.l1 @ layer_parameters m.l2

let square x =
  x *. x

let activation_name n =
  if n.nonlin then "Tanh" else "Linear"

let print_layer_summary layer layer_name =
  match layer.neurons with
  | [] ->
      Printf.printf "%s: 0 neurons\n" layer_name
  | first_neuron :: _ ->
      let num_neurons = List.length layer.neurons in
      let activation = activation_name first_neuron in
      let inputs_per_neuron = List.length first_neuron.w in
      let params_in_layer = List.length (layer_parameters layer) in
      Printf.printf "%s: %d neurons | %d input(s) each | %s | %d parameter(s)\n"
        layer_name num_neurons inputs_per_neuron activation params_in_layer

let print_mlp_summary model =
  let total_params = List.length (mlp_parameters model) in
  Printf.printf "MLP Summary\n";
  print_layer_summary model.l1 "Layer 1";
  print_layer_summary model.l2 "Layer 2";
  Printf.printf "Total parameters: %d\n\n" total_params

let () =
  Random.self_init ();

  let model = make_mlp () in
  let params = mlp_parameters model in
  let lr = 0.001 in

  print_mlp_summary model;

  Printf.printf "Initial parameters:\n";
  List.iteri
    (fun i p -> Printf.printf "p%d = %.4f\n" i p.value)
    params;
  Printf.printf "\n";

  let data =
    [
      (-2.0, 4.0);
      (-1.0, 1.0);
      (0.0, 0.0);
      (1.0, 1.0);
      (2.0, 4.0);
    ]
  in

  let test_points =
    [-3.0; -1.5; -0.5; 0.5; 1.5; 3.0]
  in

  for i = 0 to 4999 do
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

    let total_loss = sum_tensors losses in

    backward total_loss;

    if i mod 250 = 0 then
      Printf.printf "iter %d | loss = %.6f\n" i total_loss.value;

    step total_loss lr
  done;

  Printf.printf "\nFinal parameters:\n";
  List.iteri
    (fun i p -> Printf.printf "p%d = %.4f\n" i p.value)
    (mlp_parameters model);

  Printf.printf "\nPredictions on training data:\n";
  List.iter
    (fun (x_val, y_val) ->
      let x = make x_val in
      let pred = mlp_forward model x in
      Printf.printf "x = %.1f | pred = %.4f | true = %.1f\n"
        x_val pred.value y_val)
    data;

  Printf.printf "\nGeneralization test:\n";
  List.iter
    (fun x_val ->
      let x = make x_val in
      let pred = mlp_forward model x in
      let true_val = square x_val in
      Printf.printf "x = %.1f | pred = %.4f | true = %.4f\n"
        x_val pred.value true_val)
    test_points