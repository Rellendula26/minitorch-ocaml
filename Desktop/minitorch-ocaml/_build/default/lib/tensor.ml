type op =
  | Leaf
  | Add
  | Mul
  | Sub
  | Div
  | Relu
  | Tanh

type t = {
  id : int;
  mutable value : float;
  mutable grad : float;
  op : op;
  parents : t list;
  is_parameter : bool;
}

let next_id =
  let counter = ref 0 in
  fun () ->
    let id = !counter in
    incr counter;
    id

let make v =
  {
    id = next_id ();
    value = v;
    grad = 0.0;
    op = Leaf;
    parents = [];
    is_parameter = false;
  }

let parameter v =
  {
    id = next_id ();
    value = v;
    grad = 0.0;
    op = Leaf;
    parents = [];
    is_parameter = true;
  }

let add a b =
  {
    id = next_id ();
    value = a.value +. b.value;
    grad = 0.0;
    op = Add;
    parents = [a; b];
    is_parameter = false;
  }

let mul a b =
  {
    id = next_id ();
    value = a.value *. b.value;
    grad = 0.0;
    op = Mul;
    parents = [a; b];
    is_parameter = false;
  }

let sub a b =
  {
    id = next_id ();
    value = a.value -. b.value;
    grad = 0.0;
    op = Sub;
    parents = [a; b];
    is_parameter = false;
  }

let div a b =
  if b.value = 0.0 then failwith "Division by zero"
  else
    {
      id = next_id ();
      value = a.value /. b.value;
      grad = 0.0;
      op = Div;
      parents = [a; b];
      is_parameter = false;
    }

let relu a =
  {
    id = next_id ();
    value = if a.value > 0.0 then a.value else 0.0;
    grad = 0.0;
    op = Relu;
    parents = [a];
    is_parameter = false;
  }

let tanh a =
  {
    id = next_id ();
    value = Stdlib.tanh a.value;
    grad = 0.0;
    op = Tanh;
    parents = [a];
    is_parameter = false;
  }

let topo_sort root =
  let visited = Hashtbl.create 128 in
  let topo = ref [] in

  let rec build v =
    if not (Hashtbl.mem visited v.id) then begin
      Hashtbl.add visited v.id true;
      List.iter build v.parents;
      topo := v :: !topo
    end
  in

  build root;
  List.rev !topo

let backward root =
  let topo = topo_sort root in

  List.iter (fun v -> v.grad <- 0.0) topo;
  root.grad <- 1.0;

  List.iter
    (fun node ->
      match node.op, node.parents with
      | Leaf, _ -> ()
      | Add, [a; b] ->
          a.grad <- a.grad +. node.grad;
          b.grad <- b.grad +. node.grad
      | Mul, [a; b] ->
          a.grad <- a.grad +. (b.value *. node.grad);
          b.grad <- b.grad +. (a.value *. node.grad)
      | Sub, [a; b] ->
          a.grad <- a.grad +. node.grad;
          b.grad <- b.grad -. node.grad
      | Div, [a; b] ->
          a.grad <- a.grad +. ((1.0 /. b.value) *. node.grad);
          b.grad <- b.grad +. ((-. a.value /. (b.value *. b.value)) *. node.grad)
      | Relu, [a] ->
          let local_grad = if a.value > 0.0 then 1.0 else 0.0 in
          a.grad <- a.grad +. (local_grad *. node.grad)
      | Tanh, [a] ->
          let t = node.value in
          a.grad <- a.grad +. ((1.0 -. (t *. t)) *. node.grad)
      | _, _ ->
          failwith "Invalid computation graph")
    (List.rev topo)

let zero_grad root =
  let topo = topo_sort root in
  List.iter (fun v -> v.grad <- 0.0) topo

let step params lr =
  List.iter
    (fun v ->
      if v.is_parameter then
        v.value <- v.value -. (lr *. v.grad))
    params

let op_to_string = function
  | Leaf -> "Leaf"
  | Add -> "Add"
  | Mul -> "Mul"
  | Sub -> "Sub"
  | Div -> "Div"
  | Relu -> "Relu"
  | Tanh -> "Tanh"

let to_string x =
  Printf.sprintf
    "{ id = %d; value = %.4f; grad = %.4f; op = %s; parents = %d; is_parameter = %b }"
    x.id
    x.value
    x.grad
    (op_to_string x.op)
    (List.length x.parents)
    x.is_parameter