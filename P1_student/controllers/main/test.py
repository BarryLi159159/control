# Fill the respective function to implement the PID controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# Custom Controller Class
class CustomController(BaseController):
    def __init__(self, trajectory):
        super().__init__(trajectory)
        # Initialize necessary variables
        self.integralPsiError = 0
        self.previousPsiError = 0
        self.integralXdotError = 0
        self.previousXdotError = 0

    def update(self, timestep):
        trajectory = self.trajectory
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        # ---------------|Lateral Controller|-------------------------
        
        # Find the closest node to the vehicle
        _, node = closestNode(X, Y, trajectory)
        
        # Choose a node that is ahead of our current node based on index
        forwardIndex = 50
        
        # Two distinct ways to calculate the desired heading angle:
        # 1. Find the angle between a node ahead and the car's current position
        # 2. Find the angle between two nodes - one ahead, and one closest
        # The first method has better overall performance, as the second method
        # can read zero error when the car is not actually on the trajectory.
        
        # Use a try-except to avoid grabbing an out-of-scope index
        try:
            psiDesired = np.arctan2(trajectory[node + forwardIndex, 1] - Y, 
                                    trajectory[node + forwardIndex, 0] - X)
        except IndexError:
            # If forwardIndex is out of bounds, use the last trajectory point
            psiDesired = np.arctan2(trajectory[-1, 1] - Y, 
                                    trajectory[-1, 0] - X)
        
        # PID gains for lateral control
        kp = 1
        ki = 0.005
        kd = 0.001
        
        # Calculate difference between desired and actual heading angle
        psiError = wrapToPi(psiDesired - psi)
        self.integralPsiError += psiError * delT
        derivativePsiError = (psiError - self.previousPsiError) / delT
        delta = kp * psiError + ki * self.integralPsiError + kd * derivativePsiError
        delta = wrapToPi(delta)
        
        # Store the current error as the previous error for the next update
        self.previousPsiError = psiError
        
        # ---------------|Longitudinal Controller|-------------------------
        
        # PID gains for speed control
        kp = 200
        ki = 10
        kd = 30
        
        # Reference velocity for longitudinal control
        desiredVelocity = 6
        xdotError = desiredVelocity - xdot
        self.integralXdotError += xdotError * delT
        derivativeXdotError = (xdotError - self.previousXdotError) / delT
        F = kp * xdotError + ki * self.integralXdotError + kd * derivativeXdotError
        
        # Update previous error for next timestep
        self.previousXdotError = xdotError
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
