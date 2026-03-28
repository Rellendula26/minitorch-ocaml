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

val scalar : float -> tensor
val vector : float list -> tensor
val matrix : float list list -> tensor
val tensor_of_array : float array -> int list -> tensor

val zeros : int list -> tensor
val ones : int list -> tensor
val shape : tensor -> int list
val numel : tensor -> int

val make : tensor -> t
val parameter : tensor -> t
val of_float : float -> t
val of_vector : float list -> t
val of_matrix : float list list -> t

val add : t -> t -> t
val mul : t -> t -> t
val sub : t -> t -> t
val div : ?eps:float -> t -> t -> t

val relu : t -> t
val tanh : t -> t
val exp : t -> t
val log : t -> t
val sigmoid : t -> t
val pow : t -> float -> t
val sum : t -> t
val transpose : t -> t
val matmul : t -> t -> t

val addf : t -> float -> t
val mulf : t -> float -> t
val subf : t -> float -> t
val divf : ?eps:float -> t -> float -> t

val neg : t -> t
val square : t -> t

val topo_sort : t -> t list
val backward : t -> unit
val zero_grad : t -> unit
val step : t list -> float -> unit

val op_to_string : op -> string
val tensor_to_string : tensor -> string
val parent_ids : t -> string
val to_string : t -> string