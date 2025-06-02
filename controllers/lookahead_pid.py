from . import BaseController
import numpy as np

class Controller(BaseController):
    """PID controller augmented with a simple lookahead feedforward term."""
    def __init__(self, kp=0.3, ki=0.05, kd=-0.1, kf=0.1, lookahead=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kf = kf
        self.lookahead = lookahead
        self.error_integral = 0.0
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        ff = 0.0
        if future_plan is not None and len(future_plan.lataccel) > 0:
            ff = np.mean(future_plan.lataccel[: self.lookahead]) - current_lataccel

        return (
            self.kp * error
            + self.ki * self.error_integral
            + self.kd * error_diff
            + self.kf * ff
        )
