#!/usr/bin/env python2

from pyexpat import model
import unittest
import numpy as np

import rospy
import rostest
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
# from localization.sensor_model import SensorModel
from __init__ import TEST_MAP_ARRAY, TEST_PRECOMPUTED_TABLE, TEST_PARTICLES_2, \
    TEST_SENSOR_MODEL_INPUT_SCANS, TEST_SENSOR_MODEL_OUTPUT_PROBABILITIES

class SensorModel():
    def __init__(self):
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201

        self.zmin = 0
        self.zmax = self.table_width-1
        self.eps = 1 # for Pmax

        self.sensor_model_table = np.zeros([self.table_width, self.table_width])

        self.precompute_sensor_model()

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
                return eta*1.0/np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(zki-d)**2/(2.0*sigma**2))
            return 0

        def Pshort(zki, d):
            if zki>=0 and zki<=d and d!=0:
                return 2.0/d*(1.0-zki/d)
            return 0

        def Pmax(zki, zmax=zmax, eps=eps):
            if zki==zmax:
                return 1.0
            return 0

        def Prand(zki, zmax=zmax):
            if zki>=0 and zki<=zmax:
                return 1.0/zmax
            return 0

        def Pall(zki, din, eta, ahit=self.alpha_hit, ashort=self.alpha_short, amax=self.alpha_max, arand=self.alpha_rand):
            Pall = ahit*Phit(zki, eta, din) + ashort*Pshort(zki, din) + amax*Pmax(zki)+arand*Prand(zki)
            return Pall   # no squash to less peak
        
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
                self.sensor_model_table[zk_row, d_col] = Pall(float(zk_row), float(d_col), 1.0/eta_sum)

            # normalize the whole column
            self.sensor_model_table[:,d_col] /= sum(self.sensor_model_table[:,d_col])

class TestSensorModel(unittest.TestCase):
    def __init__(self):
        self.sensor_model = SensorModel()

        self.tol = 1e-6

    def tearDown(self):
        pass

    def test_precompute_sensor_model(self):
        expected_table = np.array(TEST_PRECOMPUTED_TABLE)
        actual_table = self.sensor_model.sensor_model_table

        self.assertTrue(actual_table.shape, expected_table.shape)
        aa = np.zeros([2, 201])
        diff = np.abs(expected_table-actual_table)
        re   = np.where(diff>0.01)
        print(re)

        # for col in range(self.sensor_model.table_width):
        #     # print("number %d,\n", col)
        #     aa[0,col] = np.sum(np.abs(actual_table[:,col]-expected_table[:,col]))
        #     aa[1,col] = np.sum(np.abs(actual_table[col,:]-expected_table[col,:]))
        #     print("error col \n", np.sum(actual_table[:,col]-expected_table[:,col]))
        #     print("error row \n", np.sum(actual_table[col,:]-expected_table[col,:]))
        # print(np.sum(np.abs(aa[0,:])), np.sum(np.abs(aa[1,:])), np.max(np.abs(aa[0,:])), np.max(np.abs(aa[1,:])))
        np.testing.assert_allclose(expected_table, actual_table, rtol=self.tol)


if __name__ == "__main__":
    # rospy.init_node("sensor_model_test")
    # rostest.rosrun("localization", 'test_sensor_model', TestSensorModel)
    model = TestSensorModel()
    model.test_precompute_sensor_model()

