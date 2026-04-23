import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        # Configuration
        self.max_lost_frames = 60
        self.dist_threshold = 180
        self.velocity_decay = 0.85
        # 0.7 = 70% new detection, 30% old position
        self.smoothing = 0.7


        self.cups = {
            i: {'pos': None, 'vel': np.array([0.0, 0.0]), 'lost_count': 0, 'tethered_to': None}
            for i in range(3)
        }
        self.winning_id = None

    def update(self, detections):
        #PREDICT & DECAY
        preds = {}
        for i in range(3):
            if self.cups[i]['pos'] is not None:
                # Apply velocity
                new_pred = np.array(self.cups[i]['pos']) + self.cups[i]['vel']
                preds[i] = new_pred
                # Decay velocity if they were already lost
                if self.cups[i]['lost_count'] > 0:
                    self.cups[i]['vel'] *= self.velocity_decay
            else:
                preds[i] = None

        # MATCHING
        if detections:
            dist_matrix = np.full((3, len(detections)), 999.0)
            for i in range(3):
                if preds[i] is not None:
                    for j, det in enumerate(detections):
                        dist_matrix[i, j] = np.linalg.norm(preds[i] - det)

            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_ids = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < self.dist_threshold:
                    det_pos = np.array(detections[c])

                    # Apply Alpha-Beta Smoothing
                    if self.cups[r]['pos'] is not None:
                        old_pos = np.array(self.cups[r]['pos'])
                        smoothed_pos = (det_pos * self.smoothing) + (old_pos * (1 - self.smoothing))
                        self.cups[r]['vel'] = smoothed_pos - old_pos
                        self.cups[r]['pos'] = tuple(smoothed_pos.astype(int))
                    else:
                        self.cups[r]['pos'] = tuple(det_pos.astype(int))

                    self.cups[r]['lost_count'] = 0
                    self.cups[r]['tethered_to'] = None
                    matched_ids.add(r)
                    matched_dets.add(c)

            # RECOVERY & TETHERING
            unmatched_dets = [d for j, d in enumerate(detections) if j not in matched_dets]
            lost_ids = [i for i in range(3) if i not in matched_ids]

            for det in unmatched_dets:
                if lost_ids:
                    # Find the best lost ID to give this detection to
                    best_id = None
                    min_d = float('inf')

                    for l_id in lost_ids:
                        # FIX: Only calculate distance if the cup has a position
                        if self.cups[l_id]['pos'] is not None:
                            d = np.linalg.norm(np.array(self.cups[l_id]['pos']) - det)
                            if d < min_d:
                                min_d = d
                                best_id = l_id

                    # If we didn't find a "close" lost cup (or they are all None)
                    # just take the first available lost ID slot
                    if best_id is None:
                        best_id = lost_ids[0]

                    self.cups[best_id]['pos'] = tuple(np.array(det).astype(int))
                    self.cups[best_id]['lost_count'] = 0
                    self.cups[best_id]['tethered_to'] = None
                    lost_ids.remove(best_id)

            # Merge Logic
            for i in lost_ids:
                self.cups[i]['lost_count'] += 1

                # Check if we should tether to a visible cup
                # If we are lost, look for a visible cup that is very close
                if self.cups[i]['pos'] is not None:
                    for visible_id in range(3):
                        # Check if the other cup is visible AND has a position
                        if self.cups[visible_id]['lost_count'] == 0 and self.cups[visible_id]['pos'] is not None:

                            dist = np.linalg.norm(
                                np.array(self.cups[i]['pos']) - np.array(self.cups[visible_id]['pos']))

                            if dist < 80:  # Overlap threshold
                                self.cups[i]['tethered_to'] = visible_id
                # --- FIX ENDS HERE ---

                # If tethered, follow the leader
                leader = self.cups[i]['tethered_to']
                if leader is not None and self.cups[leader]['lost_count'] == 0:
                    self.cups[i]['pos'] = self.cups[leader]['pos']
                    self.cups[i]['vel'] = self.cups[leader]['vel']
                else:
                    # If not tethered, just drift with momentum
                    if self.cups[i]['pos'] is not None:
                        drift_pos = np.array(self.cups[i]['pos']) + self.cups[i]['vel']
                        self.cups[i]['pos'] = tuple(drift_pos.astype(int))

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