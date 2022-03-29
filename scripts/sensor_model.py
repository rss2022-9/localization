from locale import normalize
import numpy as np
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:

    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param('~map_topic')
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.num_particles = rospy.get_param("~num_particles")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201

        self.zmin = 0
        self.zmax = self.table_width-1
        self.eps = 1.0 # for Pmax
        self.sigma_hit = 8.0
        self.squash = 1.0/2.2   # avoid peak in the probablity
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = np.zeros([self.table_width, self.table_width])
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        zmin = self.zmin
        zmax = self.zmax
        sigma_hit = self.sigma_hit
        eps = self.eps # for Pmax

        def Phit(zki, eta, d, sigma=sigma_hit, zmax=zmax):
            if zki>=0 and zki<=zmax:
                return eta/np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(zki-d)**2/(2.0*sigma**2))
            return 0

        def Pshort(zki, d):
            if zki>=0 and zki<=d and d!=0:
                return 2.0/d*(1-zki/d)
            return 0

        def Pmax(zki, zmax=zmax, eps=eps):
            if zki==zmax:
                return 1.0/eps
            return 0

        def Prand(zki, zmax=zmax):
            if zki>=0 and zki<=zmax:
                return 1.0/zmax
            return 0

        def Pall(zki, din, eta, ahit = self.alpha_hit, ashort = self.alpha_short, amax = self.alpha_max, arand = self.alpha_rand):
            return ahit*Phit(zki, eta, din) + ashort*Pshort(zki, din) + amax*Pmax(zki)+arand*Prand(zki)
        
        # column for each actual distance, row for each measured distance
        # val_list map the scale of dmin dmax to the actual value in the LUT, for d and zk, the list is the same (same range and delta)
        # we don't need val_list, after normalization, they will be the same
        # val_list = np.linspace(zmin, zmax, self.table_width)
        # val_list = range(self.table_width)
        for d_col in range(self.table_width):

            eta_sum = 0.0              # normalization factor for Phit 
            for zk_row in range(self.table_width):
                eta_sum += Phit(float(zk_row), 1.0, float(d_col))

            for zk_row in range(self.table_width):
                self.sensor_model_table[zk_row, d_col] = Pall(float(zk_row), float(d_col), 1.0/float(eta_sum))

            # normalize the whole column
            self.sensor_model_table[:,d_col] /= sum(self.sensor_model_table[:,d_col])
        # print(self.sensor_model_table[0,:])

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return
        num_particles = particles.shape[0]

        ####################################
        # how to calculate the distribution, avg or compare each one?

        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle
        # val_list = range(self.zmin, self.zmax, self.table_width)

        # down sample all the measurements of particle to length of observation data
        # TODO: make sure observation is row vec
        # down_sample_matrix= np.tile(down_sample_index, (particles.shape[0], 1))                                   # broadcast to matrix, which is the col indices
        # col, row = np.meshgrid(np.arange(observation.shape[0]), np.arange(particles.shape[0])) # x-> col, y->row
        # down_sample_obs   = observation[down_sample_index]    # down sampled observation
        
        # down sample observation equally to # beams per particle
        # down_sample_index        = np.array(np.linspace(0, observation.shape[0]-1, num=self.num_beams_per_particle), dtype=int) # generate downsample indicies
        # observation_down_sample  = observation[down_sample_index]                                    # down sample observation
        observation_down_sample  = observation / (self.map_resolution*self.lidar_scale_to_map_scale)               # convert meters to pixels
        observation_matrix       = np.tile(np.array(observation_down_sample), (num_particles,1))     # repeat down_sampled observation in row
        observation_matrix       = np.clip(observation_matrix, self.zmin, self.zmax)                 # limit the x, y coordinate value to zmin, zmax (in pixel representation)
        observation_matrix = np.rint(observation_matrix).astype(np.uint16) # np.array(observation_matrix, dtype=int)                                 # convert to int as array indicies

        scans = self.scan_sim.scan(particles)                          # get ray-casting
        scans /= (self.map_resolution*self.lidar_scale_to_map_scale)   # convert meters to pixels
        scans = np.clip(scans, self.zmin, self.zmax)                   # limit the x, y coordinate value to zmin, zmax (in pixel representation)
        scans = np.rint(scans).astype(np.uint16)
        
        probability_m = self.sensor_model_table[observation_matrix, scans]     # scans-particle measurement k-row; observation-real lidar d-col 
        # probability_vec = np.sum(np.log(probability_m), axis=1)              # guarantee sum of the probability for each particle <= 1
        # probability_vec = np.sum(probability_m,axis=1)                         # sum all the probablity for a particle
        probability_vec   = np.prod(probability_m, axis=1)
        probability_vec   = probability_vec**self.squash
        return probability_vec # /np.sum(probability_vec)
        
        # probability = np.zeros(particles.shape[0])

        # for i in range(particles.shape[0]):

        #     zki = particles[i, :2].dot(particles[i, :2])
        #     d = observation[:2].dot(observation[:2])

        #     differencez = np.abs(zki-val_list)
        #     indexz = differencez.argmin()
        #     differenced = np.abs(zki-val_list)
        #     indexd = differenced.argmin()

        #     probability[i] = self.sensor_model_table[indexz, indexd]

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)
        self.map_resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")