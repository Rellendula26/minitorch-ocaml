type tensor = {
  data : float array;
  shape : int list;
}

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
  | Sum
  | Transpose
  | MatMul

type t = {
  id : int;
  mutable value : tensor;
  mutable grad : tensor;
  op : op;
  parents : t list;
  is_parameter : bool;
}

let rec product = function
  | [] -> 1
  | x :: xs -> x * product xs

let numel x = product x.shape
let shape x = x.shape

let copy_tensor x =
  { data = Array.copy x.data; shape = x.shape }

let tensor_of_array data shape =
  let expected = product shape in
  if Array.length data <> expected then
    failwith "tensor_of_array: data length does not match shape"
  else
    { data = Array.copy data; shape }

let scalar x =
  { data = [| x |]; shape = [] }

let vector xs =
  let arr = Array.of_list xs in
  { data = arr; shape = [ Array.length arr ] }

let matrix rows =
  match rows with
  | [] -> failwith "matrix: empty matrix not supported"
  | first_row :: rest ->
      let cols = List.length first_row in
      if cols = 0 then failwith "matrix: empty rows not supported";
      List.iter
        (fun row ->
          if List.length row <> cols then
            failwith "matrix: ragged rows are not allowed")
        rest;
      let flat = Array.of_list (List.concat rows) in
      { data = flat; shape = [ List.length rows; cols ] }

let zeros shape =
  { data = Array.make (product shape) 0.0; shape }

let ones shape =
  { data = Array.make (product shape) 1.0; shape }

let zeros_like x = zeros x.shape
let ones_like x = ones x.shape

let ensure_same_shape a b =
  if a.shape <> b.shape then
    failwith "shape mismatch"

let map f x =
  { data = Array.map f x.data; shape = x.shape }

let map2 f a b =
  ensure_same_shape a b;
  let n = Array.length a.data in
  let out = Array.make n 0.0 in
  for i = 0 to n - 1 do
    out.(i) <- f a.data.(i) b.data.(i)
  done;
  { data = out; shape = a.shape }

let add_tensor a b = map2 ( +. ) a b
let sub_tensor a b = map2 ( -. ) a b
let mul_tensor a b = map2 ( *. ) a b

let div_tensor ?(eps = 1e-12) a b =
  ensure_same_shape a b;
  let n = Array.length a.data in
  let out = Array.make n 0.0 in
  for i = 0 to n - 1 do
    if abs_float b.data.(i) < eps then
      failwith "div_tensor: denominator too close to zero";
    out.(i) <- a.data.(i) /. b.data.(i)
  done;
  { data = out; shape = a.shape }

let mul_scalar x c =
  map (fun v -> v *. c) x

let tensor_sum x =
  scalar (Array.fold_left ( +. ) 0.0 x.data)

let fill_like x c =
  { data = Array.make (numel x) c; shape = x.shape }

let transpose_tensor x =
  match x.shape with
  | [ rows; cols ] ->
      let out = Array.make (rows * cols) 0.0 in
      for r = 0 to rows - 1 do
        for c = 0 to cols - 1 do
          out.(c * rows + r) <- x.data.(r * cols + c)
        done
      done;
      { data = out; shape = [ cols; rows ] }
  | _ -> failwith "transpose: expected rank-2 tensor"

let matmul_tensor a b =
  match (a.shape, b.shape) with
  | [ m; n1 ], [ n2; p ] when n1 = n2 ->
      let out = Array.make (m * p) 0.0 in
      for i = 0 to m - 1 do
        for j = 0 to p - 1 do
          let acc = ref 0.0 in
          for k = 0 to n1 - 1 do
            acc := !acc +. (a.data.(i * n1 + k) *. b.data.(k * p + j))
          done;
          out.(i * p + j) <- !acc
        done
      done;
      { data = out; shape = [ m; p ] }
  | _ -> failwith "matmul: incompatible shapes"

let tensor_to_string x =
  let elems =
    x.data
    |> Array.to_list
    |> List.map (Printf.sprintf "%.4f")
    |> String.concat "; "
  in
  let shp =
    x.shape
    |> List.map string_of_int
    |> String.concat "; "
  in
  Printf.sprintf "{ data = [|%s|]; shape = [%s] }" elems shp

let next_id =
  let counter = ref 0 in
  fun () ->
    let id = !counter in
    incr counter;
    id

let make v =
  {
    id = next_id ();
    value = copy_tensor v;
    grad = zeros (shape v);
    op = Leaf;
    parents = [];
    is_parameter = false;
  }

let parameter v =
  {
    id = next_id ();
    value = copy_tensor v;
    grad = zeros (shape v);
    op = Leaf;
    parents = [];
    is_parameter = true;
  }

let of_float x = make (scalar x)
let of_vector xs = make (vector xs)
let of_matrix rows = make (matrix rows)

let make_node value op parents =
  {
    id = next_id ();
    value;
    grad = zeros (shape value);
    op;
    parents;
    is_parameter = false;
  }

let add a b =
  make_node (add_tensor a.value b.value) Add [ a; b ]

let mul a b =
  make_node (mul_tensor a.value b.value) Mul [ a; b ]

let sub a b =
  make_node (sub_tensor a.value b.value) Sub [ a; b ]

let div ?(eps = 1e-12) a b =
  make_node (div_tensor ~eps a.value b.value) Div [ a; b ]

let relu a =
  make_node
    (map (fun v -> if v > 0.0 then v else 0.0) a.value)
    Relu
    [ a ]

let tanh a =
  make_node (map Stdlib.tanh a.value) Tanh [ a ]

let exp a =
  make_node (map Stdlib.exp a.value) Exp [ a ]

let log a =
  if Array.exists (fun v -> v <= 0.0) a.value.data then
    failwith "log: all inputs must be positive"
  else
    make_node (map Stdlib.log a.value) Log [ a ]

let sigmoid a =
  let s v = 1.0 /. (1.0 +. Stdlib.exp (-.v)) in
  make_node (map s a.value) Sigmoid [ a ]

let pow a p =
  make_node (map (fun v -> v ** p) a.value) (Pow p) [ a ]

let sum a =
  make_node (tensor_sum a.value) Sum [ a ]

let transpose a =
  make_node (transpose_tensor a.value) Transpose [ a ]

let matmul a b =
  make_node (matmul_tensor a.value b.value) MatMul [ a; b ]

let addf a x = add a (of_float x)
let mulf a x = mul a (of_float x)
let subf a x = sub a (of_float x)
let divf ?(eps = 1e-12) a x = div ~eps a (of_float x)

let neg a = mulf a (-1.0)
let square a = mul a a

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
  List.iter (fun v -> v.grad <- zeros (shape v.value)) topo;
  root.grad <- ones_like root.value;

  List.iter
    (fun node ->
      match (node.op, node.parents) with
      | Leaf, _ -> ()
      | Add, [ a; b ] ->
          a.grad <- add_tensor a.grad node.grad;
          b.grad <- add_tensor b.grad node.grad
      | Sub, [ a; b ] ->
          a.grad <- add_tensor a.grad node.grad;
          b.grad <- sub_tensor b.grad node.grad
      | Mul, [ a; b ] ->
          a.grad <- add_tensor a.grad (mul_tensor b.value node.grad);
          b.grad <- add_tensor b.grad (mul_tensor a.value node.grad)
      | Div, [ a; b ] ->
          let da = div_tensor node.grad b.value in
          let b_sq = mul_tensor b.value b.value in
          let num = mul_tensor a.value node.grad in
          let db = mul_scalar (div_tensor num b_sq) (-1.0) in
          a.grad <- add_tensor a.grad da;
          b.grad <- add_tensor b.grad db
      | Relu, [ a ] ->
          let local =
            map (fun v -> if v > 0.0 then 1.0 else 0.0) a.value
          in
          a.grad <- add_tensor a.grad (mul_tensor local node.grad)
      | Tanh, [ a ] ->
          let local =
            map (fun v -> 1.0 -. (v *. v)) node.value
          in
          a.grad <- add_tensor a.grad (mul_tensor local node.grad)
      | Exp, [ a ] ->
          a.grad <- add_tensor a.grad (mul_tensor node.value node.grad)
      | Log, [ a ] ->
          let local = map (fun v -> 1.0 /. v) a.value in
          a.grad <- add_tensor a.grad (mul_tensor local node.grad)
      | Sigmoid, [ a ] ->
          let s = node.value in
          let one_minus_s = sub_tensor (ones_like s) s in
          let local = mul_tensor s one_minus_s in
          a.grad <- add_tensor a.grad (mul_tensor local node.grad)
      | Pow p, [ a ] ->
          let local =
            map (fun v -> p *. (v ** (p -. 1.0))) a.value
          in
          a.grad <- add_tensor a.grad (mul_tensor local node.grad)
      | Sum, [ a ] ->
          let upstream =
            match node.grad.data with
            | [| g |] -> fill_like a.value g
            | _ -> failwith "sum backward: upstream gradient must be scalar"
          in
          a.grad <- add_tensor a.grad upstream
      | Transpose, [ a ] ->
          a.grad <- add_tensor a.grad (transpose_tensor node.grad)
      | MatMul, [ a; b ] ->
          let da = matmul_tensor node.grad (transpose_tensor b.value) in
          let db = matmul_tensor (transpose_tensor a.value) node.grad in
          a.grad <- add_tensor a.grad da;
          b.grad <- add_tensor b.grad db
      | _ ->
          failwith "Invalid computation graph")
    (List.rev topo)

let zero_grad root =
  let topo = topo_sort root in
  List.iter (fun v -> v.grad <- zeros (shape v.value)) topo

let step params lr =
  List.iter
    (fun v ->
      if v.is_parameter then
        v.value <- sub_tensor v.value (mul_scalar v.grad lr))
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
  | Sum -> "Sum"
  | Transpose -> "Transpose"
  | MatMul -> "MatMul"

let parent_ids x =
  x.parents
  |> List.map (fun p -> string_of_int p.id)
  |> String.concat "; "

let to_string x =
  Printf.sprintf
    "{ id = %d; value = %s; grad = %s; op = %s; parents = [%s]; is_parameter = %b }"
    x.id
    (tensor_to_string x.value)
    (tensor_to_string x.grad)
    (op_to_string x.op)
    (parent_ids x)
    x.is_parameter