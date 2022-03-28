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
from visualization_tools import *


class ParticleFilter:

    def __init__(self):

        # Parameters
        cloud_topic = "/wall"
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")
        self.laser_sub = rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.calcprobs, queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry, self.updposes, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initposes, queue_size=1)
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)

        # Initialize the models
        self.m_model = MotionModel()
        self.s_model = SensorModel()

        # Initialize important variables
        self.lock = Lock()
        self.num_parts = rospy.get_param("~num_parts")
        self.size = (self.num_parts,3)
        self.parts = np.zeros(self.size)
        self.odom = Odometry()
        self.marker = Marker()
        self.line_pub = rospy.Publisher(cloud_topic, Marker, queue_size=1)

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your parts, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def update_parts(self, new_parts):
        self.lock.acquire()
        try:
            x = np.mean(new_parts[:,0])
            y = np.mean(new_parts[:,1])
            o = circmean(new_parts[:,2])
            odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
            self.odom.pose.pose = Pose(Point(x,y,0.0), Quaternion(*odom_quat))
            self.odom_pub.publish(self.odom)
            VisualizationTools.plot_line(new_parts[:,0], new_parts[:,1], self.line_pub, frame="/map")
            self.parts = new_parts
        finally:
            self.lock.release()

    def initposes(self, clickpose):
        X = clickpose.pose.pose.position.x
        Y = clickpose.pose.pose.position.y
        R = 2
        S = self.num_parts
        phi = np.random.random(size=S)*2*np.pi
        rad = np.random.random(size=S)*R
        xlist = X + R*np.cos(phi)
        ylist = Y + R*np.sin(phi)
        olist = np.random.random(size=S)*2*np.pi
        new_parts = np.column_stack((xlist,ylist,olist))
        self.update_parts(new_parts)

    def updposes(self,odom):
        inpodom = [odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.angular.z]
        new_parts = self.m_model.evaluate(self.parts,inpodom)
        self.update_parts(new_parts)

    def calcprobs(self,sensdata):
        pass
        observation = np.copy(sensdata.ranges).flatten()
        probs = self.s_model.evaluate(self.parts,observation)
        new_parts = self.parts[np.random.choice(self.parts.shape[0],size=self.num_parts,p=probs)]
        self.update_parts(new_parts)


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
