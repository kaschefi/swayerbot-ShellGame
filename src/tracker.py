import numpy as np
from scipy.optimize import linear_sum_assignment


class SimpleKalmanFilter:
    def __init__(self, dt=1.0):
        # State: [x, y, dx, dy]
        self.x = np.zeros((4, 1))
        # Covariance
        self.P = np.eye(4) * 500.0
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise (higher means we trust prediction more)
        self.R = np.eye(2) * 10.0
        
        # Process noise (lower means we expect constant velocity)
        self.Q = np.array([
            [1, 0,   0,   0],
            [0, 1,   0,   0],
            [0, 0, 50.0,  0],
            [0, 0,   0, 50.0]
        ])
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2, 0]
        
    def update(self, z):
        z = np.array(z).reshape((2, 1))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2, 0]


class Tracker:
    def __init__(self):
        # Configuration
        self.max_lost_frames = 60
        self.dist_threshold = 180
        self.velocity_decay = 0.85
        
        # cups dict now holds a Kalman Filter per id
        self.cups = {
            i: {'pos': None, 'kf': SimpleKalmanFilter(), 'lost_count': 0, 'tethered_to': None}
            for i in range(3)
        }
        self.winning_id = None

    def update(self, detections):
        # 1. PREDICT
        preds = {}
        for i in range(3):
            if self.cups[i]['pos'] is not None:
                # Get KF prediction
                pred_pos = self.cups[i]['kf'].predict()
                preds[i] = pred_pos
                
                # Decay velocity in the state vector if they were already lost
                if self.cups[i]['lost_count'] > 0:
                    self.cups[i]['kf'].x[2] *= self.velocity_decay
                    self.cups[i]['kf'].x[3] *= self.velocity_decay
            else:
                preds[i] = None

        # 2. MATCHING
        if detections:
            # We use an augmented distance matrix
            dist_matrix = np.full((3, len(detections)), 999.0)
            for i in range(3):
                if preds[i] is not None:
                    # Current predicted velocity [vx, vy]
                    pred_vel = self.cups[i]['kf'].x[2:4, 0]
                    vel_norm = np.linalg.norm(pred_vel)
                    vel_dir = pred_vel / vel_norm if vel_norm > 0 else np.zeros(2)

                    for j, det in enumerate(detections):
                        dist = np.linalg.norm(preds[i] - det)
                        
                        # Velocity Penalty (Momentum Continuity)
                        # Penalize matching to a detection that shifts the cup abruptly backwards
                        direction = det - np.array(self.cups[i]['pos'])
                        dir_norm = np.linalg.norm(direction)
                        direction_normed = direction / dir_norm if dir_norm > 0 else np.zeros(2)
                        
                        cosine_sim = np.dot(direction_normed, vel_dir) if vel_norm > 1.0 else 1.0
                        
                        # Normalize cosine_sim from [-1, 1] to [2, 0] where 2 means opposite direction
                        # Multiplying by a weight factor (e.g. 50) offsets distance cost.
                        vel_penalty = (1.0 - cosine_sim) * 50.0 
                        
                        dist_matrix[i, j] = dist + vel_penalty

            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_ids = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < self.dist_threshold:
                    det_pos = np.array(detections[c])

                    # 3. UPDATE
                    if self.cups[r]['pos'] is not None:
                        # KF Update
                        smoothed_pos = self.cups[r]['kf'].update(det_pos)
                        self.cups[r]['pos'] = tuple(smoothed_pos.astype(int))
                    else:
                        # Initialize KF state if just discovered
                        self.cups[r]['kf'].x[0:2, 0] = det_pos
                        self.cups[r]['kf'].x[2:4, 0] = 0
                        self.cups[r]['pos'] = tuple(det_pos.astype(int))

                    self.cups[r]['lost_count'] = 0
                    self.cups[r]['tethered_to'] = None
                    matched_ids.add(r)
                    matched_dets.add(c)

            # 4. RECOVERY & TETHERING
            unmatched_dets = [d for j, d in enumerate(detections) if j not in matched_dets]
            lost_ids = [i for i in range(3) if i not in matched_ids]

            for det in unmatched_dets:
                if lost_ids:
                    best_id = None
                    min_d = float('inf')

                    for l_id in lost_ids:
                        if self.cups[l_id]['pos'] is not None:
                            d = np.linalg.norm(np.array(self.cups[l_id]['pos']) - det)
                            if d < min_d:
                                min_d = d
                                best_id = l_id

                    if best_id is None:
                        best_id = lost_ids[0]

                    # Give it to the recovered ID
                    self.cups[best_id]['kf'].x[0:2, 0] = np.array(det)
                    self.cups[best_id]['kf'].x[2:4, 0] = 0  # reset momentum to prevent erratic jump
                    self.cups[best_id]['pos'] = tuple(np.array(det).astype(int))
                    self.cups[best_id]['lost_count'] = 0
                    self.cups[best_id]['tethered_to'] = None
                    lost_ids.remove(best_id)

            # Merge / Lost Logic
            for i in lost_ids:
                self.cups[i]['lost_count'] += 1

                # If we are lost, look for a visible cup that is very close
                if self.cups[i]['pos'] is not None:
                    for visible_id in range(3):
                        if self.cups[visible_id]['lost_count'] == 0 and self.cups[visible_id]['pos'] is not None:
                            dist = np.linalg.norm(
                                np.array(self.cups[i]['pos']) - np.array(self.cups[visible_id]['pos']))

                            if dist < 80:  # Overlap threshold
                                self.cups[i]['tethered_to'] = visible_id

                # If tethered, ride along
                leader = self.cups[i]['tethered_to']
                if leader is not None and self.cups[leader]['lost_count'] == 0:
                    leader_pos = self.cups[leader]['pos']
                    leader_vel = self.cups[leader]['kf'].x[2:4, 0]
                    
                    self.cups[i]['pos'] = leader_pos
                    # Copy leader momentum roughly so when they split, it holds shape
                    self.cups[i]['kf'].x[0:2, 0] = np.array(leader_pos)
                    self.cups[i]['kf'].x[2:4, 0] = leader_vel
                else:
                    # If not tethered, just drift with momentum from KF prediction
                    if self.cups[i]['pos'] is not None:
                        self.cups[i]['pos'] = tuple(preds[i].astype(int))

        return self.get_positions()

    def get_positions(self):
        return {i: c['pos'] for i, c in self.cups.items()
                if c['pos'] is not None and c['lost_count'] < self.max_lost_frames}

    def assign_winner(self, ball_pos):
        if not ball_pos: return None
        min_dist = float('inf')
        for i, c in self.cups.items():
            if c['pos']:
                dist = np.linalg.norm(np.array(ball_pos) - np.array(c['pos']))
                if dist < min_dist:
                    min_dist = dist
                    self.winning_id = i
        return self.winning_id