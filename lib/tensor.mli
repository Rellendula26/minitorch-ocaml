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

val make : float -> t
val parameter : float -> t
val of_float : float -> t

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
val parent_ids : t -> string
val to_string : t -> string
