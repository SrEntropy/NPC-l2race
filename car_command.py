# === CarCommand object the server expects ===
class CarCommand:
    def __init__(self, steer, throttle, brake):
        self.steer = steer   # [-1, 1]
        self.throttle = throttle  # [0, 1]
        self.brake = brake   # [0, 1]
