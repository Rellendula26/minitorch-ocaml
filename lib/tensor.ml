type op =
  | Leaf
  | Add
  | Mul
  | Sub
  | Div
  | Relu
  | Tanh
  | Exp
  | Log
  | Sigmoid
  | Pow of float

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

let of_float = make

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

let div ?(eps = 1e-12) a b =
  if abs_float b.value < eps then
    failwith "Division error: denominator too close to zero"
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

let exp a =
  {
    id = next_id ();
    value = Stdlib.exp a.value;
    grad = 0.0;
    op = Exp;
    parents = [a];
    is_parameter = false;
  }

let log a =
  if a.value <= 0.0 then
    failwith "Log error: input must be positive"
  else
    {
      id = next_id ();
      value = Stdlib.log a.value;
      grad = 0.0;
      op = Log;
      parents = [a];
      is_parameter = false;
    }

let pow a p =
  {
    id = next_id ();
    value = a.value ** p;
    grad = 0.0;
    op = Pow p;
    parents = [a];
    is_parameter = false;
  }

let addf a x = add a (make x)
let mulf a x = mul a (make x)
let subf a x = sub a (make x)
let divf ?(eps = 1e-12) a x = div ~eps a (make x)

let neg a = mulf a (-1.0)

let square a = mul a a

let sigmoid a =
  let s = 1.0 /. (1.0 +. Stdlib.exp (-.a.value)) in
  {
    id = next_id ();
    value = s;
    grad = 0.0;
    op = Sigmoid;
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
          b.grad <- b.grad +. ((-.a.value /. (b.value *. b.value)) *. node.grad)
      | Relu, [a] ->
          let local_grad = if a.value > 0.0 then 1.0 else 0.0 in
          a.grad <- a.grad +. (local_grad *. node.grad)
      | Tanh, [a] ->
          let t = node.value in
          a.grad <- a.grad +. ((1.0 -. (t *. t)) *. node.grad)
      | Exp, [a] ->
          a.grad <- a.grad +. (node.value *. node.grad)
      | Log, [a] ->
          a.grad <- a.grad +. ((1.0 /. a.value) *. node.grad)
      | Sigmoid, [a] ->
          let s = node.value in
          a.grad <- a.grad +. ((s *. (1.0 -. s)) *. node.grad)
      | Pow p, [a] ->
          a.grad <- a.grad +. ((p *. (a.value ** (p -. 1.0))) *. node.grad)
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
  | Exp -> "Exp"
  | Log -> "Log"
  | Sigmoid -> "Sigmoid"
  | Pow p -> Printf.sprintf "Pow(%.2f)" p

let parent_ids x =
  x.parents
  |> List.map (fun p -> string_of_int p.id)
  |> String.concat "; "

let to_string x =
  Printf.sprintf
    "{ id = %d; value = %.4f; grad = %.4f; op = %s; parents = [%s]; is_parameter = %b }"
    x.id
    x.value
    x.grad
    (op_to_string x.op)
    (parent_ids x)
    x.is_parameter
