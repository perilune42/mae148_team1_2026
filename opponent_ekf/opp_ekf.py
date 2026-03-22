import numpy as np
from numpy import linalg as la

class RobocarEKF:
    def __init__(self, initial_x: float, initial_y: float, 
                 initial_yaw: float, initial_vel: float, wheelbase: float = 2.5) -> None:
        
        # PARAMETERS
        self.n_states = 4 # [x, y, yaw, v]
        self.L = wheelbase # needed for bicycle model steering calculation
        
        # State vector: strictly maintained as a (4, 1) column vector
        self._x = np.array([initial_x, initial_y, initial_yaw, initial_vel], dtype=float).reshape(-1, 1)
        
        # Covariance of state
        self._P = np.eye(self.n_states)
        
        # Process noise covariance matrix (Q) 
        # Size is 4x4 because our noise is applied directly to the 4 state variables
        self.process_noise = np.eye(self.n_states) * 0.1
        
        # Measurement noise covariance matrix (R)
        # Size is 4x4 because our synthesized camera measurement yields 4 values
        self.R = np.eye(self.n_states) * 0.5 
        
        # History buffers for finite difference derivatives
        self._prev_state = None  
        self._prev_cam_xz = None 
        
    def prediction(self, dt: float) -> None:
        # 1. Estimate control inputs [steering_angle, acceleration] via finite difference
        """if self._prev_state is None:
            # First timestep: assume 0 acceleration and 0 steering
            theta_dot = 0.0
            v_dot = 0.0
            v_prev = self.vel[0] 
        else:
            # Finite difference using previous state
            theta_dot = (self.yaw[0] - self._prev_state[2, 0]) / dt
            v_dot = (self.vel[0] - self._prev_state[3, 0]) / dt
            v_prev = self.vel[0] """

        steering_angle = 0#np.atan2(theta_dot * self.L, v_prev)
        acceleration = 0#v_dot

        # Store current state for the *next* prediction step
        self._prev_state = self._x.copy()
            
        # 2. Project the state ahead (x = f(x, u))
        self.mean = self.f(steering_angle, acceleration, dt) 
        
        # 3. Project the error covariance (P = F P Ft + Q)
        F_k = self.F(steering_angle, acceleration, dt)
        
        # Optimization: We assume process noise is additive directly to the state, 
        # removing the need for the W input jacobian.
        self._P = F_k @ self._P @ F_k.T + self.process_noise
        
        
    def correction(self, dt: float, cam_meas: list[float], ego_state: list[float]) -> None:
        """
        cam_meas: [x, y, z] from the camera frame
        ego_state: [x, y, yaw, v] of the ego vehicle in world frame
        """
        cam_x, cam_y, cam_z = cam_meas
        ego_x, ego_y, ego_yaw, ego_v = ego_state
        
        # calculate derivatives for camera relative state using finite difference
        if self._prev_cam_xz is None:
            x_dot = 0.0
            z_dot = 0.0
        else:
            prev_x, prev_z = self._prev_cam_xz
            x_dot = (cam_x - prev_x) / dt
            z_dot = (cam_z - prev_z) / dt
            
        # update camera history buffer
        self._prev_cam_xz = (cam_x, cam_z)

        # relative state                                                                                          b60-degree quadrant resolution
        rel_x = cam_x * np.cos(ego_yaw-np.pi/2) - cam_z * np.sin(ego_yaw-np.pi/2)
        rel_y = cam_x * np.sin(ego_yaw-np.pi/2) + cam_z * np.cos(ego_yaw-np.pi/2)
        rel_vx = x_dot * np.cos(ego_yaw - np.pi/2) - z_dot * np.sin(ego_yaw - np.pi/2)
        rel_vy = x_dot * np.sin(ego_yaw - np.pi/2) + z_dot * np.cos(ego_yaw - np.pi/2)
        world_vx = ego_v * np.cos(ego_yaw) + rel_vx
        world_vy = ego_v * np.sin(ego_yaw) + rel_vy        
        
        # calculate raw speed and the angle of the velocity vector
        world_vel = np.sqrt(world_vx**2 + world_vy**2)
        world_yaw = np.arctan2(world_vy, world_vx)
        # decouple heading from vel
        # car is moving, decide if its forward or backward
        angle_diff = (world_yaw - self.yaw[0] + np.pi) % (2 * np.pi) - np.pi
        
        if abs(angle_diff) > np.pi / 2:
            # if vel is more than 90 degrees from heading, the car must be reversing
            world_yaw += np.pi
            world_yaw = (world_yaw + np.pi) % (2 * np.pi) - np.pi 
            world_vel = -world_vel
        
        z_meas = np.array([rel_x+ego_x, rel_y+ego_y, world_yaw, world_vel]).reshape(-1, 1)

        # 4. Standard EKF Update Steps
        H_k = self.H()
        
        # y = z - h(x) (Innovation)
        y = z_meas - self.h()
        
        y[2, 0] = (y[2, 0] + np.pi) % (2 * np.pi) - np.pi # normalize yaw
        
        # S = H P Ht + R
        S = H_k @ self._P @ H_k.T + self.R
        
        # K = P Ht S^-1 
        # Optimization: Using np.linalg.solve is faster and more numerically stable than explicit inverse
        K = self._P @ H_k.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        
        # x = x + K y
        self.mean = self._x + K @ y
        
        # P = (I - K H) P
        self._P = (np.eye(self.n_states) - K @ H_k) @ self._P
    
    
    # --- HELPER FUNCTIONS ---
    
    def f(self, delta: float, a: float, dt: float) -> np.ndarray:
        """
        Kinematic Bicycle Model (Constant Turn Rate and Acceleration):
        x_new = x + v * cos(yaw) * dt
        y_new = y + v * sin(yaw) * dt
        yaw_new = yaw + (v / L) * tan(delta) * dt
        v_new = v + a * dt
        """
        x, y, yaw, v = self._x.flatten()
        
        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        yaw_new = yaw + (v / self.L) * np.tan(delta) * dt
        v_new = v + a * dt
        
        return np.array([x_new, y_new, yaw_new, v_new]).reshape(-1, 1)
    
    def h(self) -> np.ndarray:
        """
        Measurement function:
        Since our measurement vector `z_meas` is already fully transformed into the 
        world frame state format [x, y, yaw, v], our expected measurement is simply 
        our current state estimate.
        """
        return self._x.copy()
    
    def F(self, delta: float, a: float, dt: float) -> np.ndarray:
        """
        Jacobian of the state transition function f(x, u) with respect to state x.
        """
        s = np.sin(self.yaw[0])
        c = np.cos(self.yaw[0])
        v = self.vel[0]
        return np.array([
            [1, 0, -v*s*dt, c*dt],
            [0, 1, v*c*dt,  s*dt],
            [0, 0, 1,       np.tan(delta)*dt/self.L],
            [0, 0, 0,       1]
        ])
    
    def H(self) -> np.ndarray:
        """
        Jacobian of the measurement function h(x) with respect to state x.
        
        Approach:
        Since our measurement `z` is exactly the same format as our state vector,
        and `self.h()` simply returns the state, the partial derivative of the state 
        with respect to itself is just the Identity matrix. 
        Size should be 4x4.
        """
        return np.eye(self.n_states)

    # --- PROPERTIES ---
    
    @property
    def mean(self) -> np.ndarray:
        return self._x[:, 0]
    
    @mean.setter
    def mean(self, value: np.ndarray):
        self._x = np.asarray(value).reshape(self.n_states, 1)

    @property
    def cov(self) -> np.ndarray:
        return self._P
     
    @property
    def pos_x(self) -> np.ndarray:
        return self._x[0]
        
    @property
    def pos_y(self) -> np.ndarray:
        return self._x[1]
        
    @property
    def yaw(self) -> np.ndarray:
        return self._x[2]
        
    @property   
    def vel(self) -> np.ndarray:
        return self._x[3]