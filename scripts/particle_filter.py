#!/usr/bin/env python2

import numpy as np
import rospy
import tf
import tf2_ros

from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Pose, Quaternion, Twist, Vector3
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from sensor_model import SensorModel
from motion_model import MotionModel
from visualization_tools import *
from scipy.stats import circmean
from threading import Lock


class ParticleFilter:

    def __init__(self):

        # Parameters
        cloud_topic = "/wall"
        scan_topic = rospy.get_param("~scan_topic")
        odom_topic = rospy.get_param("~odom_topic")
        map_topic = rospy.get_param("~map_topic")
        DRIVE_TOPIC = "/drive"
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")
        self.turn_msg = AckermannDriveStamped()
        self.turn_msg.header.stamp = rospy.Time.now()
        self.turn_msg.drive.speed = 1
        self.turn_msg.drive.steering_angle = 0.2

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Initialize important variables
        self.lock = Lock()
        self.map_acquired = False
        self.last_time = 0.0;
        self.num_particles = rospy.get_param("~num_particles")
        self.size = (self.num_particles,3)
        self.particles = np.zeros(self.size)
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")

        self.odom = Odometry()
        self.odom.header.stamp = rospy.Time.now()
        self.odom.header.frame_id = map_topic
        self.marker = Marker()
        self.line_pub = rospy.Publisher(cloud_topic, Marker, queue_size=1)

        # Subscribers and Publishers
        self.laser_sub = rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.calcprobs, queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry, self.updposes, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initposes, queue_size=1)
        self.map_init = rospy.Subscriber(map_topic, OccupancyGrid, self.check_map, queue_size=1)
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.transform = 
        self.pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)
        
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def check_map(self, data):
        self.map_acquired = True

    def update_particles(self, new_particles):
        self.lock.acquire()
        try:
            x = np.mean(new_particles[:,0])
            y = np.mean(new_particles[:,1])
            o = circmean(new_particles[:,2])
            odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
            self.odom.pose.pose = Pose(Point(x,y,0.0), Quaternion(*odom_quat))
            self.odom_pub.publish(self.odom)

            VisualizationTools.plot_line(new_particles[:,0], new_particles[:,1], self.line_pub, frame="/map")
            self.particles = new_particles
        finally:
            self.lock.release()

    def initposes(self, clickpose):
        print("Pose Initialized")
        X = clickpose.pose.pose.position.x
        Y = clickpose.pose.pose.position.y
        R = 2
        S = self.num_particles
        phi = np.random.random(size=S)*2*np.pi
        rad = np.random.random(size=S)*R
        xlist = X + R*np.cos(phi)
        ylist = Y + R*np.sin(phi)
        olist = np.random.random(size=S)*2*np.pi
        new_particles = np.column_stack((xlist,ylist,olist))
        self.update_particles(new_particles)

    def updposes(self,odom):
        if self.map_acquired:
            dt = rospy.get_time() - self.last_time
            self.last_time = rospy.get_time()
            inpodom = [odom.twist.twist.linear.x*dt,odom.twist.twist.linear.y*dt,odom.twist.twist.angular.z*dt]
            new_particles = self.motion_model.evaluate(self.particles,inpodom)
            self.update_particles(new_particles)

    def calcprobs(self,sensdata):
        if self.map_acquired:
            # Down sample data
            down_sample_index        = np.array(np.linspace(0, sensdata.ranges.shape[0]-1, num=self.num_beams_per_particle), dtype=int) # generate downsample indicies
            observation_down_sample  = sensdata.ranges[down_sample_index] 

            
            self.pub.publish(self.turn_msg)   

            observation = np.copy(sensdata.ranges)
            probs = self.normalize(self.sensor_model.evaluate(self.particles,observation))
            arr = self.particles
            size = arr.shape[0]
            new_particles = arr[np.random.choice(arr.shape[0],size=size,p=probs),:]
            self.update_particles(new_particles)
        
    def normalize(self,v):
        norm = np.sum(v)
        if norm == 0: 
           return v
        return v / norm


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
