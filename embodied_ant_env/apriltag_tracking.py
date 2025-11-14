import cv2
from pyapriltags import Detector
import numpy as np
import time
import matplotlib.pyplot as plt


class VisionTracker:
    @staticmethod
    def camera_matrix_from_fov(res_w_h, fov_diagonal):
        width, height = res_w_h
        diag = np.sqrt(width**2 + height**2)
        f = diag / (2 * np.tan(fov_diagonal / 2))
        cx = width / 2
        cy = height / 2
        return np.array(
            [[f, 0, cx], 
            [0, f, cy], 
            [0, 0, 1]], dtype=np.float32)

    def __init__(self, camera_id=0, fov_diagonal_deg=60, K=None, tag_sizes={}, tag_ids={}, flip_z_up=True):
        self.cap = cv2.VideoCapture(camera_id)
        self.detector = Detector(families='tagCircle21h7', nthreads=1, quad_decimate=1)
        if K is None:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.K = self.camera_matrix_from_fov((width, height), np.deg2rad(fov_diagonal_deg))
        else:
            self.K = K
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        self.camera_params = [fx, fy, cx, cy]
        self.tag_sizes = {tag_ids[name]: size for name, size in tag_sizes.items()}
        self.tag_labels = {tag_ids[name]: name for name in tag_ids.keys()}
        self.origin_tag_id = tag_ids['origin']
        self.last_origin_detection = None
        self.flip_z_up = flip_z_up

    def detect(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        time_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_sizes)
        detections = self.filter_detections(detections)
        detection_time = time.time() - time_start
        # print(f"detection time: {detection_time:.3f}s")
        vis_frame = frame.copy()
        self.draw_detections(vis_frame, detections, detection_time)
        return detections, frame, vis_frame

    def filter_detections(self, detections):
        # print(f"before filtering: {len(detections)} detections")
        # print('decision margin:', [det.decision_margin for det in detections])
        # print('hamming:', [det.hamming for det in detections])
        # filter out detections by decision margin
        detections = [det for det in detections if det.decision_margin > 2.0]
        # filter out detections by hamming distance
        detections = [det for det in detections if det.hamming <= 0]
        # print(f"after filtering: {len(detections)} detections")
        return detections

    def draw_axes(self, frame, R, t, K, axis_length=0.05):
        """
        Draw the coordinate axes for a given pose (R, t) on the image.
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 camera matrix
        """
        axis_3D = np.float32([
            [0, 0, 0],                # origin
            [axis_length, 0, 0],      # x-axis
            [0, axis_length, 0],      # y-axis
            [0, 0, axis_length],      # z-axis
        ]).reshape(-1, 3)

        # Project points to 2D
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        pts_2D, _ = cv2.projectPoints(axis_3D, rvec, tvec, K, None)
        pts_2D = pts_2D.reshape(-1, 2).astype(int)

        origin = tuple(pts_2D[0])
        cv2.line(frame, origin, tuple(pts_2D[1]), (0, 0, 255), 2)  # X - red
        cv2.line(frame, origin, tuple(pts_2D[2]), (0, 255, 0), 2)  # Y - green
        cv2.line(frame, origin, tuple(pts_2D[3]), (255, 0, 0), 2)  # Z - blue

    def draw_detections(self, frame, detections, detection_time, draw_axes=False, draw_text=False):
        for det in detections:
            for i in range(4):
                pt1 = tuple(det.corners[i].astype(int))
                pt2 = tuple(det.corners[(i+1)%4].astype(int))
                cv2.line(frame, pt1, pt2, (0,255,0), 2)

            center = tuple(det.center.astype(int))
            cv2.circle(frame, center, 5, (0,0,255), -1)

            if det.pose_t is not None:
                if draw_text:
                    cv2.putText(frame, f"ID: {det.tag_id}", (center[0]+5, center[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                    cv2.putText(frame, str(det.pose_R[0]), (center[0]+30, center[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                    cv2.putText(frame, str(det.pose_R[1]), (center[0]+30, center[1]+60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                    cv2.putText(frame, str(det.pose_R[2]), (center[0]+30, center[1]+90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                if draw_axes:
                    self.draw_axes(frame, det.pose_R, det.pose_t, self.K)

        cv2.putText(frame, f"detection time: {detection_time:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    def track(self):
        detections, frame, vis_frame = self.detect()
        grouped_detections = {}
        for det in detections:
            tag_id = det.tag_id
            if tag_id not in grouped_detections:
                grouped_detections[tag_id] = []
            grouped_detections[tag_id].append(det)
        # Sort detections by decision margin for each tag_id (to get the best detection)
        for tag_id in grouped_detections:
            grouped_detections[tag_id] = sorted(grouped_detections[tag_id], key=lambda d: d.decision_margin, reverse=True)

        # Check if origin tag is detected
        if self.origin_tag_id not in grouped_detections:
            print("Warning: No origin tag detected")
            if self.last_origin_detection is None:
                print("Error: No origin reference")
                return {}, frame, vis_frame
        else:
            if len(grouped_detections[self.origin_tag_id]) > 1:
                print("Warning: Multiple origin tags detected")
            self.last_origin_detection = grouped_detections[self.origin_tag_id][0]

        # report bodies with respect to origin tag
        bodies = {}
        R_OtoC = self.last_origin_detection.pose_R
        t_OinC = self.last_origin_detection.pose_t.flatten()
        for tag_id, detections in grouped_detections.items():
            if tag_id not in self.tag_labels:
                continue
            if len(detections) > 1:
                print(f"Warning: Multiple tags detected for ID {tag_id} ({self.tag_labels[tag_id]})")
            if detections[0].pose_t is not None: # only tags with specified width are tracked
                bodies[self.tag_labels[tag_id]] = {'detection': detections[0]}
                R_BtoC = detections[0].pose_R
                t_BinC = detections[0].pose_t.flatten()
                R_BtoO = R_OtoC.T @ R_BtoC
                t_BinO = R_OtoC.T @ (t_BinC - t_OinC)
                if self.flip_z_up:
                    # default is camera frame: x points right, y points down, z into the marker
                    # flipped frame: x points up on the marker, y points left, z points out of the marker
                    R_FBtoB = R_FOtoO = np.array(
                        [[0, -1, 0],
                         [-1, 0, 0],
                         [0, 0, -1]])
                    R_FBtoFO = R_FOtoO.T @ R_BtoO @ R_FBtoB
                    t_BinFO = R_FOtoO.T @ t_BinO
                    bodies[self.tag_labels[tag_id]]['position'] = t_BinFO
                    bodies[self.tag_labels[tag_id]]['orientation'] = R_FBtoFO
                else:
                    bodies[self.tag_labels[tag_id]]['position'] = t_BinO
                    bodies[self.tag_labels[tag_id]]['orientation'] = R_BtoO
                bodies[self.tag_labels[tag_id]]['image_pos'] = detections[0].center / np.array([frame.shape[1], frame.shape[0]])
        return bodies, frame, vis_frame


def show_image(frame):
    cv2.imshow("apriltag detections", frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    import sys
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tracker = VisionTracker(camera_id=camera_id,
                            fov_diagonal_deg=78,
                            tag_sizes={'origin': 0.1, 'body': 0.045},
                            tag_ids={'origin': 0, 'body': 12})

    plt.figure(dpi=150)
    while True:
        bodies, frame, vis_frame = tracker.track()
        # Resize the visualization frame to half its original size before showing
        small_vis_frame = cv2.resize(vis_frame, (vis_frame.shape[1] // 2, vis_frame.shape[0] // 2))
        cv2.imshow("apriltag detections", small_vis_frame)
        for tag_id, detection in bodies.items():
            print(detection)
            print(f"{tag_id}: {detection['position']}")
            print(f"{tag_id}: \n{detection['orientation']}")

        if 'body' in bodies:
            plt.plot(bodies['body']['position'][0], bodies['body']['position'][1], 'o', color='blue')

        # Plot a circle with the radius of the task
        radius = 0.61
        origin = np.array([0.14194049, -0.82257924])
        plt.plot(origin[0], origin[1], 'o', color='red')
        plt.plot(origin[0] + radius*np.cos(np.linspace(0, 2*np.pi, 100)), origin[1] + radius*np.sin(np.linspace(0, 2*np.pi, 100)), color='red')
        plt.plot(0, 0, 'o', color='black')
        plt.pause(0.01)
        # equal aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=False)
        plt.grid(True)
        # plt.clf()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
