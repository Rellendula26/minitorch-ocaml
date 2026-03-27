# MiniTorch (OCaml)

A minimal deep learning framework built from scratch in OCaml to understand how modern ML systems (e.g., PyTorch) work under the hood.

---

## Overview

This project implements a reverse-mode automatic differentiation engine, dynamic computation graphs, and neural network training from first principles. The focus is on exposing how gradients propagate and how backpropagation is implemented, rather than optimizing for performance.

---

## Core Features

- Reverse-mode autodiff (backpropagation)
- Dynamic computation graph construction
- Topological traversal for gradient propagation
- Custom differentiable operators:
  - Add, Sub, Mul, Div  
  - ReLU, Tanh, Sigmoid  
  - Exp, Log, Power  
- Multi-layer perceptron (MLP) implementation
- Training pipeline with:
  - SGD and Adam optimizers  
  - Learning rate scheduling  
  - Gradient clipping  
- Numerical gradient checking (finite differences)
- Experiment logging and comparison

---

## Example: Autodiff

```ocaml
let x = make 2.0
let y = make 3.0

let out =
  sigmoid (div (add (pow x 2.0) (exp y)) (log (add y (make 1.0))))

backward out
