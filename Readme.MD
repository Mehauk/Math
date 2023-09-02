# Mathematics
A repository for experimenting with Maths.

### Table of Contents
- [Linear Algebra](#linear-algebra)
  - [Vectors](#vectors)
    - [Vector Arithmetic](#vector-arithmetic)
    - [Imaginary Numbers as Vectors](#imaginary-numbers-as-vectors)
  - [Matrices](#matrices)
    - [Matrix Arithmetic](#matrix-arithmetic)
- [Calculus](#calculus)
  - [Pre-Calc](#pre-calc)
    - [Functions](#functions)
  - [Derivatives](#derivatives)
  - [Integrals](#integrals)
  - [Differential Equations](#differential-equations)
  - [Multi-Dimensional Calculus](#multi-dimensional-calculus)
  - [Convolutions](#convolutions)
- [Machine Learning](#machine-learning)
  - [Neural Network](#neural-network)
    - [Introduction](#introduction)
  - [Forward Propagation](#forward-propagation)
  - [Back Propagation](#back-propagation)


&nbsp;
# Linear Algebra
> ### pre-requisites
> - Algebra
 ## Vectors
  ### Vector Arithmetic
  ### Imaginary Numbers as Vectors

 ## Matrices
  ### Matrix Arithmetic


&nbsp;
# Calculus
> ### pre-requisites :c
> - Algebra
> - Trigonometry

 ## Pre-Calc
  ### Functions
  > ## Functions can also be represented in code:
  > ```py
  > def some_polynomial_function(x: float) -> float:
  >    return 3*x*x + 7*x + 14
  > ```
  > 
  > A limitation due to the discrete nature of digital computers is that any 
  > function also becomes discrete. Although ${f(x) = 2x^2 + 7x + 14}$ is a 
  > continuous function both the input to the function $x$ and the output 
  > are floats, and as such they have a limited range and precision.
  > 
  > Here is another example that more clearly demonstrates this:
  > ```rs
  > fn another_polynomial_function(x: i32) -> i32 {
  >    (0.5 as i32)*x
  > }
  > ```
  > This translates to ${f(x) = {1\over2}x}$ and the domain is as follows:
  > $${D: \{x| x \in \mathbb{R}\}}$$
  > 
  > However The domain of an `i32` is: 
  > $${D: \{x| -2^{31} \le x <2^{31} | x \in \mathbb{I}\}}$$
  > Additionaly an input of ${x = 1}$ would return $0.5$, but since an `i32` 
  > cannot represent fractional values, it would return $0$;

 ## Derivatives

 ## Integrals

 ## Differential Equations

 ## Multi-Dimensional Calculus

 ## Convolutions


&nbsp;
# Machine Learning
> ### pre-requisites
> - [Calculus](#calculus) 
> - [Linear Algebra](#linear-algebra)

 ## Neural Network
  ### Introduction
  A `Neural Network` is a `function` that approcimates other `funtions`.

  The table below depicts a series of inputs passed as arguments to a `function` and the resulting return values.

  | Input - $x$ | Output - $f(x)$ |
  | :---------: | :-------------: |
  |      0      |        0        |
  |      1      |        1        |
  |      2      |        2        |
  |      3      |        3        |

  A prediction can be made about the underlying `funtion` with respect to the table above. This particular `function` maps the input to the output in a $1:1$ ratio, in other words it is the `linear function` $f(x) = x$. 
  
  ```dart
  /// This `function` is only able to create `linear functions`, this would be innacurate the further the actual `function` was from being a `first order plynomial`.
  double Function(double) predict_function(
  data: List<({input: double, output: double})>, 
  ) {
  final int length = data.length;
  double avgRatio = 0;

  data.forEach((e) => (e.output / e.input))

  avgRatio = avgRatio / length

  func = (x: double) => avgRatio*x;

  return func;
  }
  ```

  


 ## Forward Propagation

 ## Back Propagation