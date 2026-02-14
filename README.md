# Introduction & Log

## Introduction

Here is an implementation of Kalman Filter. 

Including:

1. The implementation of a Kalman filter based on discrete linear systems

2. An example system is:

$$\begin{bmatrix}
x_k^1 \\
x_k^2
\end{bmatrix}=
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x_{k-1}^1 \\
x_{k-1}^2
\end{bmatrix}+ w$$

$$\begin{bmatrix}
z_k^1 \\
z_k^2
\end{bmatrix}=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x_k^1 \\
x_k^2
\end{bmatrix}+ v$$

where $w \sim N(0,R)$, $v \sim N(0, Q)$, 

$$
Q=R=\begin{bmatrix}
0.1^2 & 0 \\
0 & 0.1^2
\end{bmatrix}$$

## Log

### 2026.02.14

第一次提交，实现了一种基于离散线性系统的卡尔曼滤波器。

   

