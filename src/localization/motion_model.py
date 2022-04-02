import numpy as np
import rospy

class MotionModel:

    def __init__(self):
        self.dist_sd = rospy.get_param("~distance_sd", 0.01)
        self.angl_sd = rospy.get_param("~angle_sd", 0.01)

    def evaluate(self, particles, odometry, noise=1):

        S = particles.shape[0] # Get number of particles

        dx = np.random.normal(loc=odometry[0], scale = self.dist_sd*noise, size=(S,))
        dy = np.random.normal(loc=odometry[1], scale = self.dist_sd*noise, size=(S,))
        do = np.random.normal(loc=odometry[2], scale = self.angl_sd*noise, size=(S,))

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
