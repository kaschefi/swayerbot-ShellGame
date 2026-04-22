import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        # always keep exactly 3 slots
        self.cups = {
            0: {'pos': None, 'lost_count': 0, 'vel': np.array([0, 0])},
            1: {'pos': None, 'lost_count': 0, 'vel': np.array([0, 0])},
            2: {'pos': None, 'lost_count': 0, 'vel': np.array([0, 0])}
        }
        self.winning_id = None

    def update(self, detections):
        if not detections:
            for i in range(3): self.cups[i]['lost_count'] += 1
            return self.get_positions()

        # 1. PREDICT (Simple momentum)
        preds = {}
        for i in range(3):
            if self.cups[i]['pos'] is not None:
                preds[i] = np.array(self.cups[i]['pos']) + self.cups[i]['vel']
            else:
                preds[i] = None

        # MATCHING
        dist_matrix = np.full((3, len(detections)), 999.0)
        for i in range(3):
            if preds[i] is not None:
                for j, det in enumerate(detections):
                    dist_matrix[i, j] = np.linalg.norm(preds[i] - det)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_ids = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < 150:  # If it's reasonably close, it's a match
                new_pos = np.array(detections[c])
                old_pos = np.array(self.cups[r]['pos'])
                self.cups[r]['vel'] = new_pos - old_pos
                self.cups[r]['pos'] = tuple(new_pos.astype(int))
                self.cups[r]['lost_count'] = 0
                matched_ids.add(r)
                matched_dets.add(c)

        # LEFTOVER RECOVERY
        # If we have a blob that didn't match, and an ID that is 'lost', FORCE them together
        unmatched_dets = [d for j, d in enumerate(detections) if j not in matched_dets]
        lost_ids = [i for i in range(3) if i not in matched_ids]

        for det in unmatched_dets:
            if lost_ids:
                # Pick the 'most' lost ID or just the first available
                target_id = lost_ids.pop(0)
                self.cups[target_id]['pos'] = tuple(np.array(det).astype(int))
                self.cups[target_id]['lost_count'] = 0
                self.cups[target_id]['vel'] = np.array([0, 0])
                matched_ids.add(target_id)

        # AGEING
        for i in range(3):
            if i not in matched_ids:
                self.cups[i]['lost_count'] += 1
                # If lost, still apply velocity so it 'ghosts' behind other cups
                if self.cups[i]['pos'] is not None:
                    new_ghost_pos = np.array(self.cups[i]['pos']) + self.cups[i]['vel']
                    self.cups[i]['pos'] = tuple(new_ghost_pos.astype(int))

        return self.get_positions()

    def get_positions(self):
        # We return the position if the cup isn't 'too lost' (2 seconds)
        return {i: c['pos'] for i, c in self.cups.items() if c['pos'] is not None and c['lost_count'] < 60}

    def assign_winner(self, ball_pos):
        # Find which ID is currently closest to the ball
        min_dist = float('inf')
        for i, c in self.cups.items():
            if c['pos']:
                dist = np.linalg.norm(np.array(ball_pos) - np.array(c['pos']))
                if dist < min_dist:
                    min_dist = dist
                    self.winning_id = i
        return self.winning_id