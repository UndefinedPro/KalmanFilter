# Introduction & Log

## Introduction

Here is an implementation of Kalman Filter. 

Including:

1. The implementation of a Kalman filter based on discrete linear systems

2. An example system is:
   $$
   \left[
   \begin{array}{c}
      x_{k+1}^1 \\\\
      x_{k+1}^2
   \end{array}
   \right]
   =
   \left[
   \begin{array}{cc}
      1 & 1 \\\\
      0 & 1
   \end{array}
   \right]
   \left[
   \begin{array}{c}
   x_k^1 \\\\
   x_k^2
   \end{array}
   \right] + w
   $$

   $$
   \begin{matrix}
   z_k^1 \\\\
   z_k^2
   \end{matrix}
   =
   \begin{matrix}
   1 & 0 \\\\
   0 & 1
   \end{matrix}
   \begin{matrix}
   x_k^1 \\\\
   x_k^2
   \end{matrix} + v
   $$

   where $w \sim N(0,R)$, $v \sim N(0, Q)$, $R = Q = \begin{matrix}0.1^2 & 0 \\\\ 0 & 0.1^2\end{matrix}$.

## Log

### 2026.02.14

第一次提交，实现了一种基于离散线性系统的卡尔曼滤波器。

   
