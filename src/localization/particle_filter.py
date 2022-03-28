#!/usr/bin/env python2

import numpy as np
import rospy
import tf

from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Pose, Quaternion, Twist, Vector3
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from sensor_model import SensorModel
from motion_model import MotionModel
from nav_msgs.msg import Odometry
from scipy.stats import circmean
from threading import Lock

class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")

        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.calcprobs, queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry, self.updposes, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initposes, queue_size=1)
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Initialize important variables
        self.lock = Lock()
        self.num_particles = rospy.get_param("~num_particles")
        self.size = (self.num_particles,3)
        self.particles = np.zeros(self.size)
        self.odom = Odometry()
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def update_particles(self, new_particles):
        self.lock.acquire()
        try:
            x = np.mean(new_particles[:,0])
            y = np.mean(new_particles[:,1])
            o = circmean(new_particles[:,2])
            odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
            self.odom.pose.pose = Pose(Point(x,y,0.), Quaternion(*odom_quat))
            self.odom_pub.publish(self.odom)
            self.particles = new_particles
        finally:
            self.lock.release()

    def initposes(self, clipose):
        X = clipose.pose.pose.position.x
        Y = clipose.pose.pose.position.y
        R = 2
        S = self.num_particles
        phi = np.random.random(size=S)*2*np.pi
        rad = np.random.random(size=S)*R
        xlist = X + R*np.cos(phi)
        ylist = Y + R*np.sin(phi)
        olist = np.random.random(size=S)*2*np.pi
        self.update_particles(np.column_stack((xlist,ylist,olist)))

    def updposes(self,odom):
        inpodom = [odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.angular.z]
        new_particles = self.motion_model.evaluate(self.particles,inpodom)
        self.update_particles(new_particles)
        
    def calcprobs(self,sensdata): 
        probs = self.sensor_model.evaluate(self.particles,np.copy(sensdata.ranges))
        new_particles = self.particles[np.random.choice(self.particles.shape[0],self.num_particles)]
        self.update_particles(new_particles)
    

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
