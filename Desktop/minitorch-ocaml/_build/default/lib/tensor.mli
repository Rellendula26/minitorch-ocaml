type op =
  | Leaf
  | Add
  | Mul
  | Sub
  | Div
  | Relu
  | Tanh

type t = {
  mutable value : float;
  mutable grad : float;
  op : op;
  parents : t list;
  is_parameter : bool;
}

val make : float -> t
val parameter : float -> t
val add : t -> t -> t
val mul : t -> t -> t
val sub : t -> t -> t
val div : t -> t -> t
val relu : t -> t
val tanh : t -> t
val backward : t -> unit
val zero_grad : t -> unit
val step : t -> float -> unit
val op_to_string : op -> string
val to_string : t -> string