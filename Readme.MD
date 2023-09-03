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
> ### pre-requisites
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
  > `function` also becomes discrete. Although ${f(x) = 2x^2 + 7x + 14}$ is 
  > a continuous `function` both the input to the `function` $x$ and  
  > the output are floats, and as such they have a limited range 
  > and precision.
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
  A `Neural Network` is a `function` that approximates other `funtions`.

  The table below depicts a series of inputs passed as arguments to a `function` and the resulting return values.

  | Input - $x$ | Output - $f(x)$ |
  | :---------: | :-------------: |
  |      0      |        0        |
  |      1      |        1        |
  |      2      |        2        |
  |      3      |        3        |

  A prediction can be made about the underlying `funtion` with respect to the table above. This particular `function` seems to map the input to the output in a $1:1$ ratio, in other words it can be assumed to be the `linear function` $f(x) = x$. This assumption could be wrong as the next point in the dataset might not be $(4, 4)$. Below is an example of a `function` that could automate this type of approximation.
  
  ```dart
  /// This `function` is only able to create `linear functions`, 
  /// this would be more innacurate the further the actual 
  /// `function` was from being a `first order plynomial`.
  double Function(double) predict_function(
  data: List<({input: double, output: double})>, 
  ) {
  final int length = data.length;
  double avgSlope = 0;

  for (var i = 1; i < length, i++) {
    final double top = (data[i].output - data[i-1].output)
    final double bottom = (data[i].input - data[i-1].input)
    avgSlope += top / bottom
  }

  avgSlope = avgSlope / (length - 1)

  func = (x: double) => avgSlope*x;

  return func;
  }
  ```  

  The above `function` would still attempt to find a linear_equation to fit the input data even if the dataset was from `non-linear functions` such as the `trigonometric functions` or even `higher order polynomial functions` to name a few. In fact `pedict_function` can only return `linear_functions` that pass through the origin. Creating a more generalized `function` estimator requires a change in process.

  Lets first tackle the problem of `linear_functions` that are offset from the origin. The particular function we wish to approximate is: $f(x) = 3x + 3$.
  Using the points $(0, 3)$, $(1, 6)$, $(2, 9)$; lets see what `predict_function` would do.

  ```dart
  final int length = 3;

  double avgSlope = ((6-3) / (1-0) + (9-6) / (2-1)) / (length - 1);

  return (x: double) => 3*x;
  ```

  The slope was calculated correctly, but the offset was lost. 

 ## Forward Propagation

 ## Back Propagation