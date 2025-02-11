import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

class CustomController(BaseController):
    def __init__(self, trajectory):
        super().__init__(trajectory)

        # Vehicle parameters
        self.lr, self.lf = 1.39, 1.55
        self.Ca, self.Iz, self.m, self.g = 20000, 25854, 1888.6, 9.81

        # LQR gains
        self.base_Q = np.diag([1, 0.1, 0.01, 0.001])  
        self.base_R = np.array([[50]])

        # PID gains
        self.Kp_lon, self.Ki_lon, self.Kd_lon = 160, 0.1, 8
        self.integral_limit = 100  
        self.index_step_lat, self.index_step_lon = 170, 2000
        self.pervXdotErr, self.cum_lon = 0, 0

    def update(self, timestep):
        trajectory = self.trajectory
        lr, lf, Ca, Iz, m = self.lr, self.lf, self.Ca, self.Iz, self.m

        # Get current vehicle state
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Find closest point on the trajectory
        _, index = closestNode(X, Y, trajectory)

        # Lateral Controller - Adaptive LQR
        index_nxt_lat = min(index + self.index_step_lat, len(trajectory) - 1)
        targetX, targetY = trajectory[index_nxt_lat, 0], trajectory[index_nxt_lat, 1]
        psi_nxt = np.arctan2(targetY - Y, targetX - X)
        psi_err = wrapToPi(psi_nxt - psi)

        # Adapt Q and R matrices based on the heading error
        adaptive_Q = self.base_Q.copy()
        adaptive_R = self.base_R * (1 + abs(psi_err) * 10)  # Increase R with larger errors

        A = np.array([[0, 1, 0, 0],
                      [0, -4*Ca/(m*xdot), 4*Ca/m, -2*Ca*(lf-lr)/(m*xdot)],
                      [0, 0, 0, 1],
                      [0, -2*Ca*(lf-lr)/(Iz*xdot), 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2 + lr**2)/(Iz*xdot)]])
        B = np.array([[0], [2*Ca/m], [0], [2*Ca*lf/Iz]])
        sys = signal.StateSpace(A, B, np.eye(4), np.zeros((4, 1))).to_discrete(delT)

        # Solve discrete LQR with adaptive Q and R
        S = linalg.solve_discrete_are(sys.A, sys.B, adaptive_Q, adaptive_R)
        K = np.linalg.inv(sys.B.T @ S @ sys.B + adaptive_R) @ (sys.B.T @ S @ sys.A)

        # Calculate state errors for lateral control
        e1 = (Y - targetY) * np.cos(psi_nxt) - (X - targetX) * np.sin(psi_nxt)
        e1Dot = ydot + xdot * wrapToPi(psi - psi_nxt)
        e2, e2Dot = wrapToPi(psi - psi_nxt), psidot
        delta = -K @ np.array([[e1], [e1Dot], [e2], [e2Dot]])
        # Ensure delta is a scalar float value
        delta = float(delta)

        # Longitudinal Controller - Dynamic Target Velocity with Anti-Windup
        index_nxt_lon = min(index + self.index_step_lon, len(trajectory) - 1)
        targetX_lon, targetY_lon = trajectory[index_nxt_lon, 0], trajectory[index_nxt_lon, 1]
        psi_nxt_lon = np.arctan2(targetY_lon - Y, targetX_lon - X)
        psi_err_lon = wrapToPi(psi_nxt_lon - psi)

        # Calculate dynamic target velocity
        ideal_velocity = 90  # base speed
        dynamic_velocity = ideal_velocity / (1 + abs(psi_err_lon) * 4)  # smoother slowdown

        # Calculate velocity error
        xdot_err = dynamic_velocity - xdot

        # Anti-windup for integral control
        self.cum_lon = min(max(self.cum_lon + xdot_err * delT, -self.integral_limit), self.integral_limit)
        
        # Calculate control force F with PID
        F = (self.Kp_lon * xdot_err +
             self.Ki_lon * self.cum_lon +
             self.Kd_lon * (xdot_err - self.pervXdotErr) / delT)

        # Update previous error for derivative term
        self.pervXdotErr = xdot_err

        # Return all states and control inputs (F, delta)
     
        return X, Y, xdot, ydot, psi, psidot, F, delta

