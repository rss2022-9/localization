import numpy as np

class MotionModel:

    def __init__(self):
        self.a1 = 0.01
        self.a2 = 0.01

    def evaluate(self, particles, odometry):
        dx = odometry[0]
        dy = odometry[1]
        dtheta = odometry[2]

        w_x = particles[:,0]
        w_y = particles[:,1]
        w_theta = particles[:,2]

        """
        orot1 = np.arctan2(dy,dx)
        otran = np.linalg.norm([dx,dy])
        orot2 = dtheta
        """
        
        d_rot1 = np.arctan2(dy,dx) #- np.random.normal(np.sqrt(self.a1*orot1))
        d_tran = np.linalg.norm([dx,dy]) #- np.random.normal(np.sqrt(self.a2*otran))
        d_rot2 = dtheta #- np.random.normal(np.sqrt(self.a1*orot2))

        x_new = w_x + d_tran*np.cos(d_rot1 + w_theta)
        y_new = w_y + d_tran*np.sin(d_rot1 + w_theta)
        o_new = w_theta + d_rot2
   
        output = np.column_stack((x_new,y_new,o_new))
        
        return output
