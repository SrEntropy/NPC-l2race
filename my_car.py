# === Example car state ===
import numpy as np

class MyCar:
    def __init__(self):
        self.x = 0.0    # position x
        self.y = 0.0    # position y
        self.vx = 0.0   # velocity x
        self.vy = 0.0   # velocity y
        self.ax = 0.0   # acceleration x
        self.ay = 0.0   # acceleration y
        self.heading = 0.0  # heading angle [radians]
        self.wheelbase = 2.5  # meters, tune for realism
