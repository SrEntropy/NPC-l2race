import numpy as np
from correction_net import CorrectionNet
from car_command import CarCommand
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
        """
        Build input vector: can include
        - car.x, car.y, car.vx, car.vy
        - angle to centerline
        - distance to centerline
        - curvature ahead (next vertex)
        - surface type encoding (one-hot or scalar)
        """
        #Build a feature vector:
         # [x, y, vx, vy, ax, ay, distance_to_centerline, angle_to_centerline, curvature_ahead, surface]

        x, y = self.car.x, self.car.y
        vx, vy = self.car.vx, self.car.vy
        ax, ay = self.car.ax, self.car.ay

        # Example placeholder for centerline helpers:
        dist_centerline = 0.0
        angle_centerline = 0.0
        curvature = 0.1
        surface = 0  # 0 = asphalt, 1 = gravel, for example

        # Fill these with real data from your helpers!
        return np.array([x, y, vx, vy, ax, ay, dist_centerline, angle_centerline, curvature, surface])

    def _load_centerline(self):
        """
        Loads ordered track centerline vertices.
        For now, simple straight path.
        """
        return np.array([
            [0, 0],
            [20, 0],
            [40, 10],
            [60, 30],
            [80, 60],
            [100, 100]
        ])    

    def _get_lookahead_target(self, state):
        """
        Find the closest point on the centerline.
        Then walk forward L meters along the segments.
        """
        car_pos = np.array([self.car.x, self.car.y])

        # Find closest segment
        closest_idx = None
        min_dist = float('inf')
        closest_proj = None

        for i in range(len(self.centerline) - 1):
            p1 = self.centerline[i]
            p2 = self.centerline[i+1]
            proj = self._project_point_on_segment(car_pos, p1, p2)
            dist = np.linalg.norm(car_pos - proj)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                closest_proj = proj

        # Walk forward until distance exceeds lookahead
        dist_accum = 0.0
        last_pt = closest_proj

        for j in range(closest_idx, len(self.centerline) - 1):
            seg_start = self.centerline[j] if j != closest_idx else closest_proj
            seg_end = self.centerline[j+1]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)

            if dist_accum + seg_len >= self.lookahead_distance:
                remain = self.lookahead_distance - dist_accum
                direction = seg_vec / seg_len
                target = seg_start + remain * direction
                return target

            dist_accum += seg_len
            last_pt = seg_end

        return self.centerline[-1]  # fallback
    def _project_point_on_segment(self, point, p1, p2):
        """
        Vector projection of point onto line segment [p1, p2]
        """
        v = p2 - p1
        u = point - p1
        len_sq = np.dot(v, v)
        if len_sq == 0:
            return p1
        t = np.dot(u, v) / len_sq
        t = np.clip(t, 0, 1)
        return p1 + t * v

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

    
    
    def _online_update(self):
        # Do small gradient step every N steps to train correction_net
        """Do one mini-batch update every N steps.
        """
        
        BATCH_SIZE = 16
        if len(self.buffer) < BATCH_SIZE:
            return  # not enough data

        # Random sample
        batch = random.sample(self.buffer, BATCH_SIZE)

        # Build (state, residual target)
        training_pairs = []
        for prev_state, prev_corr, next_state in batch:
            # Here’s a simple trick:
            # If car is drifting from centerline → increase steer_corr next time

            dist_now = next_state[6]  # assuming index 6 = distance to centerline
            # For a perfect centerline: residual steer correction should move car back

            # Super naive pseudo-target: push steer correction in opposite direction of drift
            steer_target = -0.1 * np.sign(dist_now)
            throttle_target = 0.0  # could be smarter: e.g. slow down if slip is high

            target_corr = np.array([steer_target, throttle_target])
            training_pairs.append((prev_state, target_corr))

        loss = self.correction_net.train_step(training_pairs)
        print(f"Online train loss: {loss:.4f}") 
