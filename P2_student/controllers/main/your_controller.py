# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):
    def __init__(self, trajectory):
        super().__init__(trajectory)
        # Define vehicle constants
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Controller tuning parameters
        self.index_step_lat = 170
        self.index_step_lon = 2000
        self.Kp_lon = 160
        self.Ki_lon = 0.1
        self.Kd_lon = 8

        # Initialize error terms for integral and derivative control
        self.cum_lat = 0
        self.cum_lon = 0
        self.pervPsiErr = 0
        self.pervXdotErr = 0
        self.i = 0

    def update(self, timestep):
        # Get vehicle states
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Calculate control actions
        F = self.longitudinal_control(X, Y, xdot, psi, delT)
        delta = self.lateral_control(X, Y, psi, xdot, ydot, psidot)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta

    def lateral_control(self, X, Y, psi, xdot, ydot, psidot):
        """ Computes the lateral control (steering angle delta) """
        # Find the target index for lateral control
        sqindex, index = closestNode(X, Y, self.trajectory)
        index_nxt_lat = min(index + self.index_step_lat, len(self.trajectory) - 1)
        
        # Calculate desired heading
        arr1 = self.trajectory[index_nxt_lat, 1] - Y
        arr2 = self.trajectory[index_nxt_lat, 0] - X
        psi_nxt = np.arctan2(arr1, arr2)
        psi_err = wrapToPi(psi_nxt - psi)

        # Define the state-space model
        A = np.array([[0, 1, 0, 0],
                      [0, -4 * self.Ca / (self.m * xdot), 4 * self.Ca / self.m, -2 * self.Ca * (self.lf - self.lr) / (self.m * xdot)],
                      [0, 0, 0, 1],
                      [0, -2 * self.Ca * (self.lf - self.lr) / (self.Iz * xdot), 2 * self.Ca * (self.lf - self.lr) / self.Iz, -2 * self.Ca * (self.lf**2 + self.lr**2) / (self.Iz * xdot)]])
        B = np.array([[0], [2 * self.Ca / self.m], [0], [2 * self.Ca * self.lf / self.Iz]])

        # Error in lateral states
        e1 = (Y - self.trajectory[index_nxt_lat, 1]) * np.cos(psi_nxt) - (X - self.trajectory[index_nxt_lat, 0]) * np.sin(psi_nxt)
        e2 = wrapToPi(psi - psi_nxt)
        e1Dot = ydot + xdot * e2
        e2Dot = psidot

        states = np.array([[e1], [e1Dot], [e2], [e2Dot]])
        poles = np.array([-4, -3, -2, -1])

        k = signal.place_poles(A, B, poles).gain_matrix
        delta = float(-k @ states)
        return delta

    def longitudinal_control(self, X, Y, xdot, psi, delT):
        """ Computes the longitudinal control (force F) """
        sqindex, index = closestNode(X, Y, self.trajectory)
        index_nxt_lon = min(index + self.index_step_lon, len(self.trajectory) - 1)
        
        # Calculate desired velocity based on curvature
        arr1_lon = self.trajectory[index_nxt_lon, 1] - Y
        arr2_lon = self.trajectory[index_nxt_lon, 0] - X
        psi_nxt_lon = np.arctan2(arr1_lon, arr2_lon)
        psi_err_lon = wrapToPi(psi_nxt_lon - psi)
        
        ideal_velocity = 90
        dynamic_velocity = ideal_velocity / (abs(psi_err_lon) * 6 + 1)
        
        # PID control for velocity
        xdot_err = dynamic_velocity - xdot
        self.cum_lon += xdot_err * delT

        F = (self.Kp_lon * (abs(psi_err_lon) * 3 + 1) * xdot_err +
             self.Ki_lon * self.cum_lon +
             self.Kd_lon * (abs(xdot_err - self.pervXdotErr)) / delT)

        self.pervXdotErr = xdot_err
        return F
