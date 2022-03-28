#!/usr/bin/env python2

import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from sensor_model import SensorModel
from motion_model import MotionModel
from scipy.stats import circmean
import threading
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Pose, Quaternion, Twist, Vector3


class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, numpy_msg(LaserScan),
                                          self.calcprobs, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.updposes, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initposes, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.num_particles = rospy.get_param("~num_particles")
        self.size = (self.num_particles,3)
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()
        self.particles = np.zeros([self.num_particles, 3])
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.odom = Odometry()
    def update_particles(self, new_particles):
        self.particles = new_particles

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
        #rospy.loginfo(self.particles)

    def updposes(self,odom):
        particles = self.particles
        inpodom = [odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.angular.z]
        new_particles = self.motion_model.evaluate(particles,inpodom)
        self.update_particles(new_particles)
        
        x = np.mean(new_particles[:,0])
        y = np.mean(new_particles[:,1])
        o = circmean(new_particles[:,2])
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
        self.odom.pose.pose = Pose(Point(x,y,0.), Quaternion(*odom_quat))
        self.odom_pub.publish(self.odom)
        

    def calcprobs(self,sensdata):
        #rospy.loginfo(sensdata)
        particles = self.particles
        probs = self.sensor_model.evaluate(particles,np.copy(sensdata.ranges))
        new_particles = particles[np.random.choice(particles.shape[0],self.num_particles)]
        self.update_particles(new_particles)
        x = np.mean(new_particles[:,0])
        y = np.mean(new_particles[:,1])
        o = circmean(new_particles[:,2])
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
        self.odom.pose.pose = Pose(Point(x,y,0.), Quaternion(*odom_quat))
        self.odom_pub.publish(self.odom)
    

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
