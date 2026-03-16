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
  if n.nonlin then relu act else act

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

let () =
  Random.self_init ();

  let model = make_mlp () in
  let lr = 0.001 in

  let data =
    [
      (-2.0, 4.0);
      (-1.0, 1.0);
      (0.0, 0.0);
      (1.0, 1.0);
      (2.0, 4.0);
    ]
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

  Printf.printf "\nPredictions after training:\n";
  List.iter
    (fun (x_val, y_val) ->
      let x = make x_val in
      let pred = mlp_forward model x in
      Printf.printf "x = %.1f | pred = %.4f | true = %.1f\n"
        x_val pred.value y_val)
    data