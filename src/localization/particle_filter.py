#!/usr/bin/env python2

import numpy as np
import threading
import rospy
import scipy
import tf_conversions
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
from std_msgs.msg import Float32
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
        cloud_topic = "/pf/cloud"
        self.kidnapped = rospy.get_param("~is_kidnapped", False)
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
        self.word_map = None
        self.pos_initialized = False
        # Subscribers and Publishers
        rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.low_variance_resample, queue_size=1) # laser_sub
        rospy.Subscriber(scan_topic, numpy_msg(LaserScan), self.normal_resample, queue_size=1) # laser_sub
        
        rospy.Subscriber(odom_topic, Odometry, self.apply_odom, queue_size=1) # odom_sub
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initposes, queue_size=1) # pose_sub
        rospy.Subscriber("/map", OccupancyGrid, self.map_init, queue_size=1) # map_sub
        self.odom_pub = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.dist_error = rospy.Publisher("/pf/error/distance", Float32, queue_size=1)
        self.angl_error = rospy.Publisher("/pf/error/angle", Float32, queue_size=1)
        self.old_dist_error = rospy.Publisher("/pf/error/old_distance", Float32, queue_size=1)
        self.old_angl_error = rospy.Publisher("/pf/error/old_angle", Float32, queue_size=1)
        self.world_eye = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.world_eye)
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.header.frame_id = "/map"

    def map_init(self, map_msg):
        self.world_map = map_msg
        self.map_initialized = True
        """
        if not self.kidnapped:
            self.map_initialized = True
        else:
            S = self.num_particles
            origin_x = map_msg.info.origin.position.x
            origin_y = map_msg.info.origin.position.y
            origin_w = map_msg.info.origin.orientation.w

            height = (map_msg.info.height *  map_msg.info.resolution)
            width = (map_msg.info.width * map_msg.info.resolution)

            ylist = origin_y - np.random.random_sample((S,))*height
            real_trans = self.world_eye.lookup_transform("map", "base_link", rospy.Time())
            real_w = real_trans.transform.rotation.w
            real_o = 2*np.arccos(real_w)

            xlist = origin_x - np.random.random_sample((S,))*width
            olist = real_o*np.ones((S,))
            new_particles = np.column_stack((xlist,ylist,olist))
            self.particles = new_particles
            #print(S)
            self.pub_stuff(new_particles)
            self.map_initialized = True
            self.pos_initialized = True
        """

    
        
    def initposes(self, clickpose):
        if self.map_initialized:
            S = self.num_particles
            if not self.kidnapped:
                X = clickpose.pose.pose.position.x
                Y = clickpose.pose.pose.position.y
                R = 0.2
                phi = np.random.random_sample((S,))*2*np.pi
                rad = np.random.random_sample((S,))*R
                xlist = X + rad*np.cos(phi)
                ylist = Y + rad*np.sin(phi)
            else:
                X = self.world_map.info.origin.position.x
                Y = self.world_map.info.origin.position.y
                sx = np.sign(X)
                sy = np.sign(Y)
                print(X,Y)
                height = (self.world_map.info.height *  self.world_map.info.resolution)
                width = (self.world_map.info.width * self.world_map.info.resolution)
                xlist = X - sx*np.random.random_sample((S,))*width
                ylist = Y - sy*np.random.random_sample((S,))*height
                
            w = clickpose.pose.pose.orientation.w
            o  = 2*np.arccos(w)
            olist = o*np.ones((S,))
            new_particles = np.column_stack((xlist,ylist,olist))
            self.particles = new_particles
            self.pos_initialized = True
            self.pub_stuff(new_particles)
                
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
    
    def normal_resample(self, sensdata):
        if self.map_initialized and self.pos_initialized:
            arr = self.particles + np.random.random_sample()*0.1
            S = self.num_particles
            observation  = signal.resample(sensdata.ranges,self.num_beams_per_particle) # Down Sample
            weights = self.sensor_model.evaluate(self.particles, observation)
            weights = (weights)/np.sum(weights)
            new_particles = arr[np.random.choice(arr.shape[0],size=S,p=weights),:]
            self.old_pub_stuff(new_particles)


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
            S = float(self.num_particles)

            observation  = signal.resample(sensdata.ranges,self.num_beams_per_particle) # Down Sample
            weights = self.sensor_model.evaluate(self.particles, observation)
            weights = weights/np.sum(weights)

            # Same thing as below just faster
            r = np.random.rand()/self.num_particles # generate a random number
            """
            weights_cum = np.cumsum(weights)
            m = np.arange(S)
            U = m/S + r
            i = np.argmax(np.greater_equal(weights_cum, U[:,None]),axis=1)
            new_particles = self.particles[i,:]

            """
            tc = weights[0]
            ti = 0
            resample_particle = [] # np.zeros(self.size)
            for tm in range(self.num_particles):
                tU = r + tm/float(self.num_particles)
                while tU > tc:
                    ti += 1
                    tc += weights[ti]
                resample_particle.append(self.particles[ti,:])
            new_particles = np.array(resample_particle)
            

            self.particles = new_particles
            self.pub_stuff(new_particles)

    def pub_stuff(self, new_particles):
        x = np.mean(new_particles[:,0])
        y = np.mean(new_particles[:,1])
        o = circmean(new_particles[:,2])
        odom_quat = tf_conversions.transformations.quaternion_from_euler(0, 0, o)
        real_trans = self.world_eye.lookup_transform("map", "base_link", rospy.Time())
        real_x = real_trans.transform.translation.x
        real_y = real_trans.transform.translation.y
        real_w = real_trans.transform.rotation.w
        real_o = 2*np.arccos(real_w)
        pred_pose = np.array([x,y])
        real_pose = np.array([real_x,real_y])
        dist_error = np.linalg.norm(real_pose - pred_pose)
        angl_error = real_o - o
        self.odom_msg.pose.pose = Pose(Point(x,y,0.0), Quaternion(*odom_quat))
        self.odom_pub.publish(self.odom_msg)
        self.dist_error.publish(dist_error)
        self.angl_error.publish(angl_error)
        VisualizationTools.plot_cloud(new_particles[:,0], new_particles[:,1], self.line_pub, frame="/map")
        
    def old_pub_stuff(self, old_particles):
        x = np.mean(old_particles[:,0])
        y = np.mean(old_particles[:,1])
        o = circmean(old_particles[:,2])
        odom_quat = tf_conversions.transformations.quaternion_from_euler(0, 0, o)
        real_trans = self.world_eye.lookup_transform("map", "base_link", rospy.Time())
        real_x = real_trans.transform.translation.x
        real_y = real_trans.transform.translation.y
        real_w = real_trans.transform.rotation.w
        real_o = 2*np.arccos(real_w)
        pred_pose = np.array([x,y])
        real_pose = np.array([real_x,real_y])
        old_dist_error = np.linalg.norm(real_pose - pred_pose)
        old_angl_error = real_o - o
        self.old_dist_error.publish(old_dist_error)
        self.old_angl_error.publish(old_angl_error)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
