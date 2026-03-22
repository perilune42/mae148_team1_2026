import unittest
import numpy as np
import numpy.testing as npt

# Assuming your EKF class is saved in a file named ekf.py
# If it is named differently, update the import below:
from opp_ekf import RobocarEKF 

class TestRobocarEKF(unittest.TestCase):
    
    def setUp(self):
        """Initialize a fresh EKF instance before each test."""
        self.ekf = RobocarEKF(
            initial_x=0.0, 
            initial_y=0.0, 
            initial_yaw=0.0, 
            initial_vel=0.0
        )
        self.dt = 0.1

    def warm_start(self):
        """
        Helper function to simulate 2 timesteps.
        This populates the `_prev_state` and `_prev_cam_xz` buffers so that 
        finite difference calculations are actively used in the main tests.
        """
        # Timestep 1
        self.ekf.prediction(self.dt)
        # cam: [x=2.0, y=0.0, z=10.0], ego: [x=0.0, y=0.0, yaw=0.0, v=5.0]
        self.ekf.correction(self.dt, [2.0, 0.0, 10.0], [0.0, 0.0, 0.0, 5.0])
        
        # Timestep 2
        self.ekf.prediction(self.dt)
        # cam moved slightly, ego moved forward
        self.ekf.correction(self.dt, [2.1, 0.0, 9.5], [0.5, 0.0, 0.0, 5.0])

    def test_shapes_after_constructor(self):
        """Verify the initial shapes of the state and covariance matrices."""
        # Internal state vector should be strictly a (4, 1) column vector
        self.assertEqual(self.ekf._x.shape, (4, 1))
        
        # Public mean property returns a 1D array of shape (4,)
        self.assertEqual(self.ekf.mean.shape, (4,))
        
        # Covariance matrix should be (4, 4)
        self.assertEqual(self.ekf._P.shape, (4, 4))
        self.assertEqual(self.ekf.cov.shape, (4, 4))
        
        npt.assert_allclose(self.ekf.mean, np.array([0.0, 0.0, 0.0, 0.0]).reshape((4,)))

    def test_shapes_after_prediction(self):
        """Verify shapes are maintained after a prediction step with historical data."""
        self.warm_start()
        
        # Run the target prediction step
        self.ekf.prediction(self.dt)
        
        # Assert shapes remain consistent
        self.assertEqual(self.ekf._x.shape, (4, 1), "Internal state _x changed shape after prediction.")
        self.assertEqual(self.ekf.mean.shape, (4,), "Public mean changed shape after prediction.")
        self.assertEqual(self.ekf._P.shape, (4, 4), "Internal covariance _P changed shape after prediction.")
        self.assertEqual(self.ekf.cov.shape, (4, 4), "Public cov changed shape after prediction.")

    def test_shapes_after_correction(self):
        """Verify shapes are maintained after a correction step with historical data."""
        self.warm_start()
        
        # Standard workflow: predict then correct
        self.ekf.prediction(self.dt)
        
        cam_meas = [2.2, 0.0, 9.0]
        ego_state = [1.0, 0.0, 0.0, 5.0]
        self.ekf.correction(self.dt, cam_meas, ego_state)
        
        # Assert shapes remain consistent
        self.assertEqual(self.ekf._x.shape, (4, 1), "Internal state _x changed shape after correction.")
        self.assertEqual(self.ekf.mean.shape, (4,), "Public mean changed shape after correction.")
        self.assertEqual(self.ekf._P.shape, (4, 4), "Internal covariance _P changed shape after correction.")
        self.assertEqual(self.ekf.cov.shape, (4, 4), "Public cov changed shape after correction.")

if __name__ == '__main__':
    unittest.main()