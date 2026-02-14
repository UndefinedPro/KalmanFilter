import numpy as np
import matplotlib.pyplot as plt


# 首先针对离散的线性系统进行编程
class DISCRETE_SYSTEM:
    def __init__(self, x_dim, u_dim, y_dim):
        self.state_dim = x_dim
        self.input_dim = u_dim
        self.output_dim = y_dim

        self.x = np.zeros((self.state_dim , 1))
        self.u = np.zeros((self.input_dim , 1))
        self.y = np.zeros((self.output_dim, 1))


    def Iterate(self):
        pass
    

class DISCRETE_LINEAR_SYSTEM(DISCRETE_SYSTEM):

    def __init__(self, x_dim, u_dim, y_dim, 
                 A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, 
                 H: np.ndarray, 
                 Q: np.ndarray, R: np.ndarray):
        super().__init__(x_dim, u_dim, y_dim) 
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.H = H
        self.Q = Q
        self.R = R
        
    def Iterate(self, x, u, w, v):
        x_ = np.matmul(self.A, x) + np.matmul(self.B, u) + np.matmul(self.D, w)
        y  = np.matmul(self.C, x_) + v
        z  = np.matmul(self.H, x) + v

        return x_, y, z
    
    def GenetrateData(self, steps):
        x_real = self.x.copy()
        z = self.y.copy()
        x_ = self.x
        z_ = self.y
        

        for i in range(steps):

            w = np.random.multivariate_normal([0,0], self.Q).reshape(self.Q.shape[1], 1)
            v = np.random.multivariate_normal([0,0], self.R).reshape(self.R.shape[1], 1)
            x_, _, z_ = self.Iterate(x_, self.u, w, v)
            x_real = np.hstack((x_real, x_))
            z = np.hstack((z,z_))           
        
        return x_real, z
        
    


class Kalman:
    def __init__(self, sys : DISCRETE_LINEAR_SYSTEM):
        self.sys = sys
        
    
    def KalmanFilter_OneStep(self, x_post_0, u, z, P_post_0):       
        # 计算先验状态
        pri_state = np.matmul(self.sys.A, x_post_0) + np.matmul(self.sys.B, u)
        
        # 计算先验误差矩阵
        P_pri = np.matmul(np.matmul(self.sys.A, P_post_0), self.sys.A.T) + self.sys.Q 
        
        # 计算Kalman Gain
        num = np.matmul(P_pri, self.sys.H.T)
        den = np.matmul(self.sys.H, np.matmul(P_pri, self.sys.H.T)) + self.sys.R
        K = np.matmul(num, np.linalg.pinv(den))

        # 后验估计
        x_post_1 = pri_state + np.matmul(K, (z - np.matmul(self.sys.H, pri_state)))
        
        # 更新后延估计的协方差误差
        temp = np.ones(self.sys.state_dim) - np.matmul(K, self.sys.H)
        P_post_1 = np.matmul(temp, P_pri)

        return x_post_1, P_post_1
        




if __name__ == "__main__":
    # 利用DR_CAN的匀速运动模型
    A = np.array([[1,1],
                  [0,1]], dtype = np.float32)
    B = np.array([[0,0],
                  [0,0]], dtype = np.float32)
    C = np.array([[0,0],
                  [0,0]], dtype = np.float32)
    D = np.eye(2)
    H = D.copy()
    Q = np.eye(2) * 0.01
    R = np.eye(2) * 0.01
    system = DISCRETE_LINEAR_SYSTEM(2,2,2, A, B, C, D, H,Q,R)
    kal = Kalman(sys=system)
    
    # 生成测试数据
    x_list, z_list = system.GenetrateData(100)
    
    
    # 进行Kalman Filter
    x_post = system.x.copy()
    P_post = np.eye(system.state_dim, dtype = np.float32)
    x_post_list = x_post.copy()
    P_post_list = P_post.copy()

    for i in range(100):
        x_post_, P_post_ = kal.KalmanFilter_OneStep(x_post, system.u, z_list[:,i].reshape(z_list.shape[0],1), P_post)
        
        x_post_list = np.hstack((x_post_list, x_post_))
        P_post_list = np.hstack((P_post_list, P_post_))
        x_post = x_post_.copy()
        P_post = P_post_.copy()
    
    
    plt.plot(x_list[0,:])
    plt.plot(z_list[0,:])
    plt.plot(x_post_list[0,:])
    
    plt.legend(['Real State', 'Observation', 'Estimate'])
    plt.show()
            
    
    