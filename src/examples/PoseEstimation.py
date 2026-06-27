import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        # Setup MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Constants
        self.RED = (0, 0, 255)
        self.BRIGHT_RED = (0, 100, 255)
        self.ALERT_BOX_COLOR = (0, 0, 255)
        self.DIAMOND_SIZE = 20
        self.matrix_toggle = True
        self.frame_counter = 0
        
        # Pre-allocate reusable objects
        self.overlay_frame = None
        
        # Landmarks to draw (nose removed for manual draw)
        self.LANDMARKS_TO_DRAW = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        ]

    def draw_diamond(self, img, x, y, size=None, color=None, thickness=2, animate=True):
        if size is None:
            size = self.DIAMOND_SIZE
        if color is None:
            color = self.RED
            
        x, y = int(x), int(y)
        pts = np.array([(x, y - size), (x + size, y), (x, y + size), (x - size, y)], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, thickness)

        if animate:
            offset_x = size + 4
            offset_y = size + 4
            matrix = [["1", "0"], ["0", "1"]] if self.matrix_toggle else [["0", "1"], ["1", "0"]]
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 0.7
            for r, row in enumerate(matrix):
                for c, val in enumerate(row):
                    tx = x + offset_x + c * 6
                    ty = y + offset_y + r * 10
                    cv2.putText(img, val, (tx, ty), font, font_scale, color, 1, cv2.LINE_AA)

    def draw_alert_box(self, frame, lm, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
        """Draw alert box relative to the bounding box coordinates"""
        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1
        
        ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Convert relative coordinates to absolute within the bounding box
        top_l = (int(bbox_x1 + ls.x * bbox_w), int(bbox_y1 + ls.y * bbox_h))
        top_r = (int(bbox_x1 + rs.x * bbox_w), int(bbox_y1 + rs.y * bbox_h))
        bot_l = (int(bbox_x1 + lh.x * bbox_w), int(bbox_y1 + lh.y * bbox_h))
        bot_r = (int(bbox_x1 + rh.x * bbox_w), int(bbox_y1 + rh.y * bbox_h))

        x1 = int(min(top_l[0], bot_l[0]) + (max(top_r[0], bot_r[0]) - min(top_l[0], bot_l[0])) * 0.1)
        y1 = int(min(top_l[1], top_r[1]) + (max(bot_l[1], bot_r[1]) - min(top_l[1], top_r[1])) * 0.1)
        x2 = int(max(top_r[0], bot_r[0]) - (max(top_r[0], bot_r[0]) - min(top_l[0], bot_l[0])) * 0.1)
        y2 = int(max(bot_l[1], bot_r[1]) - (max(bot_l[1], bot_r[1]) - min(top_l[1], top_r[1])) * 0.1)

        # Reuse overlay frame
        if self.overlay_frame is None or self.overlay_frame.shape != frame.shape:
            self.overlay_frame = frame.copy()
        else:
            self.overlay_frame[:] = frame
        
        cv2.rectangle(self.overlay_frame, (x1, y1), (x2, y2), self.ALERT_BOX_COLOR, thickness=-1)
        cv2.addWeighted(self.overlay_frame, 0.08, frame, 0.85, 0, frame)

        # Draw outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.BRIGHT_RED, thickness=2)

    def draw_head_marker(self, frame, lm, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
        """Draw head marker relative to the bounding box coordinates"""
        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1
        
        nose = lm[self.mp_pose.PoseLandmark.NOSE.value]
        nose_x = int(bbox_x1 + nose.x * bbox_w)
        nose_y = int(bbox_y1 + nose.y * bbox_h)
        self.draw_diamond(frame, nose_x, nose_y)

    def get_alertness_score(self, distance_squared, min_dist=200, max_dist=3500):
        min_dist_sq = min_dist * min_dist
        max_dist_sq = max_dist * max_dist
        
        if distance_squared <= min_dist_sq:
            return 10.0
        elif distance_squared >= max_dist_sq:
            return 1.0
        else:
            distance = np.sqrt(distance_squared)
            scaled = (np.log(max_dist / distance) / np.log(max_dist / min_dist))
            return round(scaled * 9 + 1, 1)

    def check_hand_to_hip_proximity(self, frame, lm, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
        """Check hand to hip proximity relative to the bounding box coordinates"""
        enlarged_hips = []
        max_alert = 0
        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1

        # Pre-compute coordinates relative to bounding box
        left_wrist = np.array([bbox_x1 + lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * bbox_w, 
                              bbox_y1 + lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * bbox_h])
        left_index = np.array([bbox_x1 + lm[self.mp_pose.PoseLandmark.LEFT_INDEX.value].x * bbox_w, 
                              bbox_y1 + lm[self.mp_pose.PoseLandmark.LEFT_INDEX.value].y * bbox_h])
        left_hip = np.array([bbox_x1 + lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * bbox_w, 
                            bbox_y1 + lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * bbox_h])
        
        right_wrist = np.array([bbox_x1 + lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * bbox_w, 
                               bbox_y1 + lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * bbox_h])
        right_index = np.array([bbox_x1 + lm[self.mp_pose.PoseLandmark.RIGHT_INDEX.value].x * bbox_w, 
                               bbox_y1 + lm[self.mp_pose.PoseLandmark.RIGHT_INDEX.value].y * bbox_h])
        right_hip = np.array([bbox_x1 + lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * bbox_w, 
                             bbox_y1 + lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * bbox_h])

        # Process both hands
        for hand_mid, hip_pos, hip_id in [
            ((left_wrist + left_index) * 0.5, left_hip, self.mp_pose.PoseLandmark.LEFT_HIP.value),
            ((right_wrist + right_index) * 0.5, right_hip, self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        ]:
            diff = hand_mid - hip_pos
            distance_squared = np.dot(diff, diff)
            alertness = self.get_alertness_score(distance_squared)
            max_alert = max(max_alert, alertness)

            cv2.line(frame, hand_mid.astype(int), hip_pos.astype(int), self.RED, 1)

            if alertness >= 7:
                enlarged_hips.append(hip_id)

        return enlarged_hips, max_alert

    def process_pose_in_bbox(self, frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
        """
        Process pose estimation within a specific bounding box.
        Returns the frame with pose annotations drawn.
        """
        self.frame_counter += 1
        
        # Extract region of interest from the frame
        roi = frame[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        if roi.size == 0:
            return frame
            
        # Convert to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_roi)

        if results and results.pose_landmarks and results.pose_landmarks.landmark:
            lm = results.pose_landmarks.landmark

            # Only process if we have sufficient landmark confidence
            if len(lm) >= 33:  # Ensure we have all required landmarks
                enlarged_hips, max_threat = self.check_hand_to_hip_proximity(frame, lm, bbox_x1, bbox_y1, bbox_x2, bbox_y2)

                # Draw landmarks relative to bounding box
                bbox_w = bbox_x2 - bbox_x1
                bbox_h = bbox_y2 - bbox_y1
                
                for idx in self.LANDMARKS_TO_DRAW:
                    landmark = lm[idx]
                    x = int(bbox_x1 + landmark.x * bbox_w)
                    y = int(bbox_y1 + landmark.y * bbox_h)
                    self.draw_diamond(frame, x, y)

                self.draw_head_marker(frame, lm, bbox_x1, bbox_y1, bbox_x2, bbox_y2)

                if max_threat >= 6:
                    self.draw_alert_box(frame, lm, bbox_x1, bbox_y1, bbox_x2, bbox_y2)

        # Update matrix toggle for animation
        self.matrix_toggle = self.frame_counter & 1
        
        return frame

    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()