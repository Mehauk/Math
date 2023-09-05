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
  > $$ {D: \lbrace x| x \in \mathbb{R} \rbrace} $$
  > 
  > However The domain of an `i32` is: 
  > $$ {D: \lbrace x| -2^{31} \le x <2^{31} | x \in \mathbb{I} \rbrace} $$
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
  ![](.misc/visuals/images/simplestnn.png)

  The figure above dipects a simple `neural network` with one input $x$, one weight $w$, and one bias at the output $b$. This can be read as;

  $$ f(x) = wx + b $$

  The following table represents inputs passed to the network above and what the expected outputs should be. From this data we hope that our `neural network` can decipher the correct relationship (i.e. `function`) between $x$ and $f_e(x)$.

  | Input - $x$ | Expected Output - $f_e(x)$ |
  | :---------: | :------------------------: |
  |      0      |             2              |
  |      1      |             5              |
  |      2      |             8              |
  |      3      |             11             |

  To start, we inititalze the constants $w, b$ with arbitrary values; in our case $1$. We can then compare the expected result with the calculated result; 

  $$ (f_e(x) - f(x))^2 $$
  $$ (f_e(0) - f(0))^2  = (2 - (1(0) + 1))^2 = 1$$

  Using the descrepancy to gradually alter the weight and the bias decreasing the the error $E$. More formally $E$ is the sum of all of the differences between excepted and calculated outputs.

  $$ E(w, b) = \sum^x  (f_e(x) - f(x))^2 $$

  $E$ is a function of both $w$ and $b$ as they are changing variables that both contribute to the error. We can use the partial derivatives of this `function` to find better values for $w$ and $b$ such that the error becomes small. 
  
  The partial derivatives are;


 ## Forward Propagation

 ## Back Propagation