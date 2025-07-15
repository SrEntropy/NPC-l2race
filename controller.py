import numpy as np
from correction_net import CorrectionNet
from collection import deque

class CarController:
    def __init__(self, my_car=None):
        self.cat = my_car
        #adjust to your state feature dim
        self.correction_net = CorrectionNet(input_dim=10)
        #tiny replay buffer
        self.buffer = deque(maxlen=10000)
        self.prev_state = None
        self.prev_action = None
        self.lookahead_distance = 5.0

    def read(self):
        state = self._get_state_features()
        
        # === Pure Pursuit ===
        target = self._get_lookahead_target(state)
        alpha = self._compute_angle_to_target(state, target)
        base_steer = self._pure_pursuit_steer(alpha)

        # Curvature-base throttle:
        base_throttle = self._curvature_based_throttle(state)

        #Tiny MLP correctionNet residuals
        # tiny net, fast inference
        steer_corr, throttle_corr = self.correction_net.predict(state)
        steer = base_steer + steer_corr
        throttle = base_throttle + throttle_corr
        #add correction net output for brake if needed
        brake = 0.0
        #clip
        steer = np.clip(steer, -1, 1)
        throttle = np.clip(throttle, 0, 1)

        # Buffer & training hooks:
        #Save to buffer
        if self.prev_state is not None:
            self.buffer.append((self.prev_state,self.prev_action,state))
        self.prev_state = state
        self.prev_action = (steer_corr,throttle_corr)
        return CarCommand(steer, throttle, brake)

    def _get_state_features(self):
        """Build input vector: can include
        - car.x, car.y, car.vx, car.vy
        - angle to centerline
        - distance to centerline
        - curvature ahead (next vertex)
        - surface type encoding (one-hot or scalar)
        """
        # placeholder; replace with real data
        state_vec = np.zeros(10)
        return state_vec


    def _get_lookahead_target(self, state):
        # Using centerline vertices & current position
        # Find point L meters ahead along the centerline
        #TODO:
        #-placeholder; replace with real logic
        return np.array([10,10])
        pass

    def _compute_angle_to_target(self, state, target):
        # Vector from car to target
        # Dot product with heading vector
        #Compute signed angle bt cat heading and vector to target point
        car_pos = np.array([self.car.x, self.car.y])
        car_heading = np.array([np.cos(self.car.heading), np.sin(self.car.heading)])
        vec_to_target = target - car_pos
        vec_to_target /= np.linalg.norm(vec_to_target)

        dot = np.dot(car_heading, vec_to_target)
        cross = np.cross(car_heading, vec_to_target)
        alpha = np.arctan2(cross, dot)
        return alpha

    def _pure_pursuit_steer(self, alpha):
        # delta = atan2(2*L*sin(alpha)/L_d)
        #PP steering formula

        #fallback to 2.5m
        L = getattr(self.car,"wheelbase",2.5)
        delta = np.arctan2(2 * L * np.sin(alpha), self.lookahead_distance

    def _curvature_based_throttle(self, state):
        #Reduce throttle if curvature ahead is high or surface is low traction
        curvature = state[7]
        surface = state[8]

        base_throttle = 1.0
        if curvature > 0.2:
            base_throttle *= 0.5
        if surface == 1:
            base_throttle *= 0.7
        return base_throttle

    def _online_update(self)
    #TODO
    #Do small gradient step every N steps to train correction_net
