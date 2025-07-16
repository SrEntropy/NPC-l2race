# === utils.py ===
import csv
import os

class RaceLogger:
    def __init__(self, log_file="log.csv"):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "x", "y", "vx", "vy", "steer", "throttle", "brake",
                                 "steer_corr", "throttle_corr", "centerline_error", "loss"])

        self.step = 0

    def log(self, car, action, residual, centerline_error, loss=None):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step,
                car.x, car.y, car.vx, car.vy,
                action.steer, action.throttle, action.brake,
                residual[0], residual[1],
                centerline_error,
                loss if loss is not None else ""
            ])
        self.step += 1
