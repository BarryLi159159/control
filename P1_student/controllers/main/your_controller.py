# Import libraries
import numpy as np
from base_controller import BaseController
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Constants
        self.lr = 1.39
        self.lf = 1.55
        self.m = 1888.6  # mass of the vehicle (kg)
        self.g = 9.81    # gravity

        # PID Gains for lateral control
        self.Kp_lat = 0.8    # Proportional gain for lateral control
        self.Ki_lat = 0.01   # Integral gain for lateral control
        self.Kd_lat = 1   # Derivative gain for lateral control
        
        # PID Gains for longitudinal control
        self.Kp_lon = 50     # Proportional gain for longitudinal control
        self.Ki_lon = 0.01   # Integral gain for longitudinal control
        self.Kd_lon = 25     # Derivative gain for longitudinal control

        # Cumulative errors for integration in PID control
        self.cum_cte = 0     # Cumulative cross-track error (for lateral control)
        self.cum_speed_err = 0  # Cumulative speed error (for longitudinal control)

        # Previous errors for derivative term
        self.prev_cte = 0    # Previous cross-track error
        self.prev_speed_err = 0  # Previous speed error

    def update(self, timestep):

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        trajectory = self.trajectory

        # Find the closest point on the trajectory
        sqindex, index = closestNode(X, Y, trajectory)
        
        # ---------------|Lateral Controller (Steering Angle)|-------------------------
        
        # Calculate the desired heading (psi_nxt)
        if index + 1 < len(trajectory):
            next_index = index + 1
        else:
            next_index = index

        # Calculate the desired direction to the next point in the trajectory
        psi_desired = np.arctan2(trajectory[next_index, 1] - Y, trajectory[next_index, 0] - X)
        
        # Calculate the cross-track error (CTE)
        psi_err = wrapToPi(psi_desired - psi)

        # Integrate the cross-track error
        self.cum_cte += psi_err * delT

        # Derivative of cross-track error
        d_cte = (psi_err - self.prev_cte) / delT

        # PID controller for steering angle (delta)
        delta = (self.Kp_lat * psi_err) + (self.Ki_lat * self.cum_cte) + (self.Kd_lat * d_cte)

        # Update the previous CTE for the next derivative calculation
        self.prev_cte = psi_err
        
        # ---------------|Longitudinal Controller (Throttle/Braking Force)|-------------------------
        
        # Set a target speed based on current trajectory (assuming constant target speed)
        target_speed = 50  # Desired speed in m/s (this can be dynamically adjusted)
        
        # Calculate speed error
        speed_err = target_speed - xdot

        # Integrate the speed error
        self.cum_speed_err += speed_err * delT

        # Derivative of speed error
        d_speed_err = (speed_err - self.prev_speed_err) / delT

        # PID controller for throttle/braking force (F)
        F = (self.Kp_lon * speed_err) + (self.Ki_lon * self.cum_speed_err) + (self.Kd_lon * d_speed_err)

        # Update the previous speed error for the next derivative calculation
        self.prev_speed_err = speed_err

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
