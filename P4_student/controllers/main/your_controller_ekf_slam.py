# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        self.Kp_lat = 5
        self.Ki_lat = 0.5
        self.Kd_lat = 1
        
        self.Kp_lon = 160
        self.Ki_lon = 0.1
        self.Kd_lon = 8
        
        self.index_step_lat = 170
        self.index_step_lon = 2000
        
        self.index_nxt_lat = 0
        self.index_nxt_lon = 0
        
        self.intPsiErr = 0
        self.intXdotErr = 0
        
        self.pervPsiErr = 0
        self.pervXdotErr = 0
        
        self.cum_lat = 0
        self.cum_lon = 0
        self.integral_limit = 100
        
        self.i = 0
        self.counter = 0
        np.random.seed(99)
         
       
    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        
        if self.counter == 0:
            
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)

            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, Y, psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        
        return y

    def update(self, timestep):
        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch states using EKF-SLAM
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True )
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        # Find the closest point on the trajectory
        sqindex, index = closestNode(X, Y, trajectory)

        # Adaptive LQR for Lateral Control
        if index + self.index_step_lat < len(trajectory):
            index_nxt_lat = index + self.index_step_lat
        else:
            index_nxt_lat = len(trajectory) - 1

        targetX, targetY = trajectory[index_nxt_lat, 0], trajectory[index_nxt_lat, 1]
        psi_nxt = np.arctan2(targetY - Y, targetX - X)
        psi_err = wrapToPi(psi_nxt - psi)

        # Adaptive Q and R matrices
        adaptive_Q = np.array([[1, 0, 0, 0],
                           [0, 0.1, 0, 0],
                           [0, 0, 0.01, 0],
                           [0, 0, 0, 0.001]])
        adaptive_R = np.array([[35 * (1 + abs(psi_err) * 10)]])

        A = np.array([[0, 1, 0, 0],
                  [0, -4 * Ca / (m * xdot), 4 * Ca / m, -2 * Ca * (lf - lr) / (m * xdot)],
                  [0, 0, 0, 1],
                  [0, -2 * Ca * (lf - lr) / (Iz * xdot), 2 * Ca * (lf - lr) / Iz, -2 * Ca * (lf**2 + lr**2) / (Iz * xdot)]])
        B = np.array([[0], [2 * Ca / m], [0], [2 * Ca * lf / Iz]])
        sys = signal.StateSpace(A, B, np.eye(4), np.zeros((4, 1))).to_discrete(delT)

        # Solve discrete LQR
        S = linalg.solve_discrete_are(sys.A, sys.B, adaptive_Q, adaptive_R)
        K = np.linalg.inv(sys.B.T @ S @ sys.B + adaptive_R) @ (sys.B.T @ S @ sys.A)

        # Calculate state errors
        e1 = (Y - targetY) * np.cos(psi_nxt) - (X - targetX) * np.sin(psi_nxt)
        e1Dot = ydot + xdot * wrapToPi(psi - psi_nxt)
        e2, e2Dot = wrapToPi(psi - psi_nxt), psidot
        states = np.array([[e1], [e1Dot], [e2], [e2Dot]])

        # Control input for lateral control
        delta = -K @ states
        delta = float(delta)  # Ensure scalar value

        # PID with Anti-Windup for Longitudinal Control
        if index + self.index_step_lon < len(trajectory):
            index_nxt_lon = index + self.index_step_lon
        else:
            index_nxt_lon = len(trajectory) - 1

        targetX_lon, targetY_lon = trajectory[index_nxt_lon, 0], trajectory[index_nxt_lon, 1]
        psi_nxt_lon = np.arctan2(targetY_lon - Y, targetX_lon - X)
        psi_err_lon = wrapToPi(psi_nxt_lon - psi)

        # Dynamic velocity adjustment
        ideal_velocity = 90
        dynamic_velocity = ideal_velocity / (1 + abs(psi_err_lon) * 4)

        # Velocity error
        xdot_err = dynamic_velocity - xdot

        # Anti-windup logic for integral term
        self.cum_lon = min(max(self.cum_lon + xdot_err * delT, -self.integral_limit), self.integral_limit)

        # PID control for force
        F = (self.Kp_lon * xdot_err +
         self.Ki_lon * self.cum_lon +
         self.Kd_lon * (xdot_err - self.pervXdotErr) / delT)

        # Update previous error for derivative term
        self.pervXdotErr = xdot_err

        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
