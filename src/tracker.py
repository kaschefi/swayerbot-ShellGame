import numpy as np
from scipy.optimize import linear_sum_assignment  # Standard for matching


class Tracker:
    def __init__(self):
        # We pre-define 3 IDs
        self.ids = [0, 1, 2]
        self.tracked_cups = {
            0: {'pos': None, 'lost_count': 0},
            1: {'pos': None, 'lost_count': 0},
            2: {'pos': None, 'lost_count': 0}
        }
        self.winning_id = None
        self.max_lost_frames = 60  # Remember for 2 seconds at 30fps

    def update(self, detected_centers):
        # If it's the very first frame, just assign 0, 1, 2 to whatever we see
        if all(v['pos'] is None for v in self.tracked_cups.values()):
            for i, pos in enumerate(detected_centers[:3]):
                self.tracked_cups[i]['pos'] = pos
            return self.get_active_positions()

        # Create a Distance Matrix
        # Rows = Our 3 IDs, Cols = New Detections
        num_dets = len(detected_centers)
        if num_dets > 0:
            current_dets = detected_centers[:3]
            dist_matrix = np.zeros((3, len(current_dets)))

            for i in range(3):
                for j in range(len(current_dets)):
                    if self.tracked_cups[i]['pos'] is not None:
                        dist_matrix[i, j] = np.linalg.norm(
                            np.array(self.tracked_cups[i]['pos']) - np.array(current_dets[j])
                        )
                    else:
                        dist_matrix[i, j] = 9999  # Placeholder for lost cups

            # Use the Hungarian Algorithm to find the best global match
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_ids = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                # Only match if the distance isn't "teleportation" level
                # Unless the cup was lost
                threshold = 200 if self.tracked_cups[r]['lost_count'] > 0 else 100

                if dist_matrix[r, c] < threshold:
                    self.tracked_cups[r]['pos'] = current_dets[c]
                    self.tracked_cups[r]['lost_count'] = 0
                    matched_ids.add(r)
                    matched_dets.add(c)

            # Handle Unmatched IDs (Mark as Lost)
            for i in range(3):
                if i not in matched_ids:
                    self.tracked_cups[i]['lost_count'] += 1
        else:
            # No detections at all, increment lost_count for everyone
            for i in range(3):
                self.tracked_cups[i]['lost_count'] += 1

        return self.get_active_positions()

    def get_active_positions(self):
        # Only return positions for cups that aren't "permanently" gone
        return {i: data['pos'] for i, data in self.tracked_cups.items()
                if data['pos'] is not None and data['lost_count'] < self.max_lost_frames}

    def assign_winner(self, ball_pos):
        min_dist = float('inf')
        for i, data in self.tracked_cups.items():
            if data['pos']:
                dist = np.linalg.norm(np.array(ball_pos) - np.array(data['pos']))
                if dist < min_dist:
                    min_dist = dist
                    self.winning_id = i
        return self.winning_id