import numpy as np

class MotionModel:

    def __init__(self):
        pass

    def evaluate(self, particles, odometry, noise=0):

        S = particles.shape[0] # Get number of particles

        if noise: # Adding noise to the odometry
            dx = odometry[0] - np.random.normal(0.01, size=(S,))
            dy = odometry[1] - np.random.normal(0.01, size=(S,))
            do = odometry[2] - np.random.normal(0.01, size=(S,))
        else:
            dx = odometry[0]
            dy = odometry[1]
            do = odometry[2]

        w_x = particles[:,0]
        w_y = particles[:,1]
        w_o = particles[:,2]

        orot1 = np.arctan2(dy,dx)
        otran = np.linalg.norm([dx,dy],axis=0)
        orot2 = do
        
        x_new = w_x + otran*np.cos(orot1 + w_o)
        y_new = w_y + otran*np.sin(orot1 + w_o)
        o_new = w_o + orot2
        
        output = np.column_stack((x_new,y_new,o_new))
        
        return output
