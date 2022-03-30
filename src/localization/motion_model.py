import numpy as np

class MotionModel:

    def __init__(self):
        self.a1 = 0.001
        self.a2 = 0.001

    def evaluate(self, particles, odometry, noise=1):
        S = particles.shape[0]

        if noise:
            dx = odometry[0] - np.random.normal(self.a1, size =(S,))
            dy = odometry[1] - np.random.normal(self.a2, size =(S,))
            dtheta = odometry[2] - np.random.normal(self.a1, size =(S,))
        else:
            dx = odometry[0]
            dy = odometry[1]
            dtheta = odometry[2]

        w_x = particles[:,0]
        w_y = particles[:,1]
        w_theta = particles[:,2]

        orot1 = np.arctan2(dy,dx)
        otran = np.linalg.norm([dx,dy])
        orot2 = dtheta
        
        d_rot1 = orot1
        d_tran = otran
        d_rot2 = orot2

        
        x_new = w_x + d_tran*np.cos(d_rot1 + w_theta)
        y_new = w_y + d_tran*np.sin(d_rot1 + w_theta)
        o_new = w_theta + d_rot2
        
        output = np.column_stack((x_new,y_new,o_new))
        
        return output
