#!/usr/bin/env python2

import numpy as np
import threading
import rospy
import scipy
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
from scipy import signal
from threading import Lock



class ParticleFilter:

    def __init__(self):

        # Parameters
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame", "/base_link_pf")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle", 100)
        self.num_particles = rospy.get_param("~num_particles", 200)
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        cloud_topic = "/cloud"

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Initialize important variables
        self.lock = Lock()
        self.last_time = None
        self.particles = None
        self.odom_msg = Odometry()
        self.marker = Marker()
        self.line_pub = rospy.Publisher(cloud_topic, Marker, queue_size=1)
        self.map_initialized = False
        self.pos_initialized = False
        # Subscribers and Publishers
        rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.low_variance_resample, queue_size=1) # laser_sub
        rospy.Subscriber(odom_topic, Odometry, self.apply_odom, queue_size=1) # odom_sub
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initposes, queue_size=1) # pose_sub
        rospy.Subscriber("/map", OccupancyGrid, self.map_init, queue_size=1) # map_sub
        self.odom_pub = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        

    def map_init(self, map):
        self.map_initialized = True

    def pub_stuff(self, new_particles):
        x = np.mean(new_particles[:,0])
        y = np.mean(new_particles[:,1])
        o = circmean(new_particles[:,2])
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, o)
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.header.frame_id = "/map"
        self.odom_msg.pose.pose = Pose(Point(x,y,0.0), Quaternion(*odom_quat))
        self.odom_pub.publish(self.odom_msg)
        VisualizationTools.plot_cloud(new_particles[:,0], new_particles[:,1], self.line_pub, frame="/map")
        

    def initposes(self, clickpose):
        if self.map_initialized:
            X = clickpose.pose.pose.position.x
            Y = clickpose.pose.pose.position.y
            w = clickpose.pose.pose.orientation.w
            o = 2*np.arccos(w)
            R = 0.2
            S = self.num_particles
            phi = np.random.random_sample((S,))*2*np.pi
            rad = np.random.random_sample((S,))*R
            xlist = X + rad*np.cos(phi)
            ylist = Y + rad*np.sin(phi)
            olist = o*np.ones((S,))
            new_particles = np.column_stack((xlist,ylist,olist))
            self.particles = new_particles
            self.pub_stuff(new_particles)
            self.pos_initialized = True

    def apply_odom(self,odom):
        if self.map_initialized and self.pos_initialized:
            if self.last_time is None:
                self.last_time = rospy.get_time()
                return
            dt = rospy.get_time() - self.last_time
            self.last_time = rospy.get_time()
            inpodom = [odom.twist.twist.linear.x*dt,odom.twist.twist.linear.y*dt,odom.twist.twist.angular.z*dt]
            new_particles = self.motion_model.evaluate(self.particles,inpodom,noise=1)
            self.particles = new_particles
            self.pub_stuff(new_particles)
    """
    def calcprobs(self,sensdata):
        if self.map_initialized and self.pos_initialized:
            particles = self.particles
            s = particles.shape[0]
            weights = self.sensor_model.evaluate(particles,sensdata.ranges)/np.sum(weights)
            new_particles = particles[np.random.choice(s,replace=True,size=s,p=probs),:]
            self.particles = new_particles
        else:
            return
    """
            
    def low_variance_resample(self, sensdata):
        if self.map_initialized and self.pos_initialized:
            """
            Reference: Probablistic Robotics
            By Zhenyang
            The basic idea is that instead of selecting samples independently of 
            each other in the resampling process the selection involves a sequential stochastic process.
            """
            
            #down_sample_index        = np.array(np.linspace(0, sensdata.ranges.shape[0]-1, num=self.num_beams_per_particle), dtype=int) # generate downsample indicies
            #observation  = sensdata.ranges[down_sample_index]

            # Down sample lidar data
            observation  = signal.resample(sensdata.ranges,self.num_beams_per_particle) # Down Sample
            weights = self.sensor_model.evaluate(self.particles, observation)
            weights = weights/np.sum(weights)

            weights_cum = np.cumsum(weights)
            r = np.random.rand()/self.num_particles # generate a random number
            resample_particle = []
            
            
            """
            tresample_particle = []
            tm = np.arange(self.num_particles, dtype = float)
            tU = tm/float(self.num_particles) + r
            ti = np.argmax(weights_cum >= tU)

            tresample_particle = self.particles[ti,:]
            print(ti)
            """

            for m in range(self.num_particles):
                U = r + float(m)/float(self.num_particles)
                i = np.argmax(weights_cum >= U)            
                resample_particle.append(self.particles[i,:])

            new_particles = np.array(resample_particle)
            self.particles = new_particles
            self.pub_stuff(new_particles)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
