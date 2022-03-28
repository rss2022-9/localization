import numpy as np
import math
import rospy
class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size

        sample(b) normal distribution mean 0 variance b
        """
        
        ####################################
        # TODO

        dx = odometry[0]
        dy = odometry[1]
        dtheta = odometry[2]

        w_x = particles[:,0]
        w_y = particles[:,1]
        w_theta = particles[:,2]

        orot1 = np.arctan2(dy,dx)
        otran = np.linalg.norm([dx,dy])
        orot2 = dtheta

        a1 = 0.01
        a2 = 0.01
        a3 = 0.01
        a4 = 0.01
        
        d_rot1 = orot1 - np.random.normal(np.sqrt(a1*orot1))
        d_tran = otran - np.random.normal(np.sqrt(a3*otran))
        d_rot2 = orot2 - np.random.normal(np.sqrt(a1*orot2))

        x_new = w_x + d_tran*np.cos(d_rot1 + w_theta)
        y_new = w_y + d_tran*np.sin(d_rot1 + w_theta)
        o_new = w_theta + d_rot2
   
        output = np.column_stack((x_new,y_new,o_new))
        


        return output


        ####################################
