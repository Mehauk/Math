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
    - [Extrema](#extrema)
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
  ### Extrema
  

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

  To start, we inititalze the constants $w, b$ with arbitrary values; in our case $1$. 
  
  $$ f(x) = 1x + 1 $$

  We can then compare the Error $E$ between the expected result and the calculated result by taking a square of the differece, This is done for every $x$;

  $$ E_x(w, b) = (f_e(x) - f(x))^2 $$
  $$ --- $$
  $$ (f_e(0) - f(0))^2  = (2 - (1(0) + 1))^2 = 1$$


  The Gradient of the error indicates the direction of steepest change and can be found using partial derivatives. Using these partial derivatives we can minimize the error for each individual $x$.

  The gradient is;
  $$ \Delta{E} = ({\delta{E}\over\delta{w}}, {\delta{E}\over\delta{b}}) $$
  
  The partial derivatives are;

  $$ {\delta{E}\over\delta{w}} = {\delta\over\delta{w}}(f_e(x) - f(x))^2 $$

  $$ {\delta\over\delta{w}}(f_e(x) - f(x))^2 = 2(f_e(x) - f(x)) {-\delta{f(x)}\over\delta{w}} $$

  $$ {\delta{f(x)}\over\delta{w}} = {\delta{w}\over\delta{w}}x + {\delta{b}\over\delta{w}} = x $$

  $$ {\delta{E}\over\delta{w}} = -2x(f_e(x) - f(x)) $$

  $$ --- $$

  $$ {\delta{E}\over\delta{b}} = {\delta\over\delta{b}}(f_e(x) - f(x))^2$$

  $$ {\delta\over\delta{b}}(f_e(x) - f(x))^2 = 2(f_e(x) - f(x)) {-\delta{f(x)}\over\delta{b}} $$

  $$ {\delta{f(x)}\over\delta{b}} = {\delta{w}\over\delta{b}}x + {\delta{b}\over\delta{b}} = 1 $$

  $$ {\delta{E}\over\delta{b}} = -2(f_e(x) - f(x)) $$

  By calculating the gradient at each individual $x$, we can nudge the variables in some negative proportion relative to its partial derivative. Since we are looking for a minimum of the error function, we can nudge the variables to the right if their respective derivative is negative and left is positive. (A minimum can be found if a function is decreasing to the left and increasing to the right.) This means we can nudge the variables negatively proportional to their partial derivatives to gradually locate a minimum. This process if formally known as [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

  $$ {\delta{E_0}\over\delta{w}} = 2(0)(f_e(0) - f(0)) $$
  $$ {\delta{E_0}\over\delta{w}} = 0 $$
  
  $$ --- $$
  
  $$ {\delta{E_0}\over\delta{b}} = -2(f_e(0) - f(0)) $$
  $$ {\delta{E_0}\over\delta{b}} = -2(2 - (1(0) + 1)) = -2(1) $$
  $$ {\delta{E_0}\over\delta{b}} = -2 $$

  Since $\delta{E_0}\over\delta{w}$ is $0$ we cannot determine which direction to nudge the variable. $\delta{E_0}\over\delta{b}$ on the other hand is negative, at $x=0$ the error $E$ is decreasing with respect to $b$. This means we can nudge b is the positive direction.

  $$ w_{new} = w_{old} - {\delta{E_0}\over\delta{w}} $$
  $$ w_{new} = w_{old} - 0 = 1 - 0 $$
  $$ w_{new} = 1 $$

  $$ --- $$

  $$ b_{new} = b_{old} - {\delta{E_0}\over\delta{b}} $$
  $$ b_{new} = 1 - (-1) $$
  $$ b_{new} = 2 $$

  Our equation is now;

  $$ f(x) = 1x + 2 $$

  We can continue this process for each entry in the dataset to slowly approximate the the `function` we are looking for. In this case we would get something close to $f(x) = 3x + 2$.



 ## Forward Propagation

 ## Back Propagation