import numpy as np

class RobocarKF:
    def __init__(self, initial_x: float, initial_y: float, 
                 initial_vx: float = 0.0, initial_vy: float = 0.0) -> None:
        
        self.n_states = 4
        
        self._x = np.array([initial_x, initial_y, initial_vx, initial_vy], dtype=float).reshape(-1, 1)
        
        self._P = np.eye(self.n_states) * 10.0
        
        self.Q = np.diag([0.1, 0.1, 2.0, 2.0])
        
        self.R = np.eye(2) * 0.5 
        
    def prediction(self, dt: float) -> None:
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self.Q
        
    def correction(self, cam_meas: list[float], ego_state: list[float]) -> None:
        cam_x, cam_y, cam_z = cam_meas
        ego_x, ego_y, ego_yaw, ego_v = ego_state
        
        rel_x = cam_x * np.cos(ego_yaw - np.pi/2) - cam_z * np.sin(ego_yaw - np.pi/2)
        rel_y = cam_x * np.sin(ego_yaw - np.pi/2) + cam_z * np.cos(ego_yaw - np.pi/2)
        
        target_x = rel_x + ego_x
        target_y = rel_y + ego_y
        
        z = np.array([target_x, target_y]).reshape(-1, 1)
        
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        y = z - H @ self._x
        
        S = H @ self._P @ H.T + self.R
        K = self._P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        
        self._x = self._x + K @ y
        self._P = (np.eye(self.n_states) - K @ H) @ self._P
    
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
    def vel_x(self) -> np.ndarray:
        return self._x[2]
        
    @property   
    def vel_y(self) -> np.ndarray:
        return self._x[3]