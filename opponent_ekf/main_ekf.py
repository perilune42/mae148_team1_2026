import numpy as np
import matplotlib.pyplot as plt
from opp_ekf import RobocarEKF as EKF

ekf = EKF(0.0, 0.0, 0.0, 0.0)

DT = 0.01
NUM_STEPS = 500

steps_per_meas = 5

camera_det = [0.0, 0.0, 1.0]
ego_state = [0.0, 0.0, np.pi/4, 0.0]

mus = []
covs = []


for step in range(NUM_STEPS):
    mus.append(ekf.mean)
    covs.append(ekf.cov)
    
    ekf.prediction(DT)
    
    if (step != 0) and (step % steps_per_meas == 0):
        ekf.correction(steps_per_meas*DT, [0.0, 0.0, 1.0 + 1.0 * np.sin(1.4*DT*step)], [step*DT*0.5, step*DT*0.5, np.pi/4, 0.5*1.41]) 

mus = np.array(mus)

plt.figure(1)
plt.suptitle('CTRV Assumptions')

plt.subplot(2, 2, 1)
plt.title('Position X')
plt.plot([mu[0] for mu in mus], 'b')
plt.plot([mu[0] + 2 * np.sqrt(cov[0,0]) for mu,cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[0] - 2 * np.sqrt(cov[0,0]) for mu,cov in zip(mus, covs)], 'r--') # -2 std

plt.subplot(2, 2, 2)
plt.title('Position Y')
plt.plot([mu[1] for mu in mus], 'b')
plt.plot([mu[1] + 2 * np.sqrt(cov[1,1]) for mu,cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[1] - 2 * np.sqrt(cov[1,1]) for mu,cov in zip(mus, covs)], 'r--') # -2 std

plt.subplot(2, 2, 3)
plt.title('Yaw')

plt.plot([mu[2] for mu in mus], 'b')
plt.plot([mu[2] + 2 * np.sqrt(cov[2,2]) for mu,cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[2] - 2 * np.sqrt(cov[2,2]) for mu,cov in zip(mus, covs)], 'r--') # -2 std

plt.subplot(2, 2, 4)
plt.title('Velocity')
plt.plot([mu[3] for mu in mus], 'b')
plt.plot([mu[3] + 2 * np.sqrt(cov[3,3]) for mu,cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[3] - 2 * np.sqrt(cov[3,3]) for mu,cov in zip(mus, covs)], 'r--') # -2 std
    
plt.tight_layout()

"""
plt.figure(2)

plt.subplot(2, 1, 1)
plt.title('Yaw')

yaws = np.arctan2(mus[:,3], mus[:,2])
plt.plot([yaw for yaw in yaws], 'b')

plt.subplot(2, 1, 2)
plt.title('Velocity')

vels = np.sqrt(mus[:,2]**2 + mus[:,3]**2)
plt.plot([v for v in vels], 'b')
plt.show()
"""

plt.show()