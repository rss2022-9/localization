#!/usr/bin/env python2

import numpy as np
import scipy
import scipy.stats
import rospy
import tf

from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Pose, Quaternion, Twist, Vector3
from ackermann_msgs.msg import AckermannDriveStamped
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from sensor_model import SensorModel
from motion_model import MotionModel
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_tools import *
from scipy.stats import circmean
from threading import Lock



class ParticleFilter:

    def __init__(self):

        # Parameters
        cloud_topic = "/cloud"
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        DRIVE_TOPIC = "/drive"
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame", "/base_link_pf")
        self.turn_msg = AckermannDriveStamped()

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Initialize important variables
        self.lock = Lock()
        self.last_time = None;
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.size = (self.num_particles,3)
        self.particles = np.zeros(self.size)
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle", 100)

        self.odom_msg = Odometry()
        
        self.marker = Marker()
        self.line_pub = rospy.Publisher(cloud_topic, Marker, queue_size=1)
        self.map_initialized = False
        self.pos_initialized = False

        # Subscribers and Publishers
        self.laser_sub = rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.calcprobs, queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry, self.updposes, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initposes, queue_size=1)
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.map_check = rospy.Subscriber("/map", OccupancyGrid, self.map_check, queue_size=1)
        self.pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def map_check(self, map):
        self.map_initialized = True

    def update_particles(self, new_particles):
        with self.lock:
            x = np.mean(new_particles[:,0])
            y = np.mean(new_particles[:,1])
            o = circmean(new_particles[:,2])
            odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
            self.odom_msg.header.stamp = rospy.Time.now()
            self.odom_msg.header.frame_id = "/map"
            self.odom_msg.pose.pose = Pose(Point(x,y,0.0), Quaternion(*odom_quat))
            #rospy.loginfo((x,y))
            self.odom_pub.publish(self.odom_msg)
            VisualizationTools.plot_line(new_particles[:,0], new_particles[:,1], self.line_pub, frame="/map")
            self.particles = new_particles
        

    def initposes(self, clickpose):
        if self.map_initialized:
            X = clickpose.pose.pose.position.x
            Y = clickpose.pose.pose.position.y
            w = clickpose.pose.pose.orientation.w
            R = 0.1
            S = self.num_particles
            phi = np.random.random_sample((S,))*2*np.pi
            rad = np.random.random_sample((S,))*R
            xlist = X + rad*np.cos(phi)
            ylist = Y + rad*np.sin(phi)
            olist = np.random.random_sample((S,))*2*np.pi
            new_particles = np.column_stack((xlist,ylist,olist))
            self.update_particles(new_particles)
            self.pos_initialized = True
        else:
            return

    def updposes(self,odom):
        if self.pos_initialized:
            dt = rospy.get_time() - self.last_time
            self.last_time = rospy.get_time()
            inpodom = [odom.twist.twist.linear.x*dt,odom.twist.twist.linear.y*dt,odom.twist.twist.angular.z*dt]
            new_particles = self.motion_model.evaluate(self.particles,inpodom,noise=0)
            self.update_particles(new_particles)
        else:
            return


    def calcprobs(self,sensdata):
        pass
        """
        self.turn_msg.header.stamp = rospy.Time.now()
        self.turn_msg.drive.speed = 1
        self.turn_msg.drive.steering_angle = 0.2
        self.pub.publish(self.turn_msg)   
        """
        """
        if self.pos_initialized:
            # Down sample data
            down_sample_index        = np.array(np.linspace(0, sensdata.ranges.shape[0]-1, num=self.num_beams_per_particle), dtype=int) # generate downsample indicies
            observation  = sensdata.ranges[down_sample_index] 
            arr = self.particles
            s = arr.shape[0]
            observation = np.copy(sensdata.ranges)
            probs = self.normalize(self.sensor_model.evaluate(arr,observation))
            new_particles = arr[np.random.choice(s,replace=True,size=s,p=probs),:]
            self.update_particles(new_particles)
        else:
            return
            """
        

    def normalize(self,v):
        norm = np.sum(v)
        if norm == 0: 
           return v
        return v / norm


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
