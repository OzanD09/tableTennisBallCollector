import cv2
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class DetectorConfig:
    grid_size: int = 64
    grid_stride: int = 48  # overlap improves recall without full dense scan

    heuristic_threshold: float = 0.6
    yolo_conf_threshold: float = 0.45
    yolo_imgsz: int = 640

    # HSV filter for tennis-ball yellow (tune with your camera)
    h_low: int = 24
    h_high: int = 45
    s_low: int = 65
    s_high: int = 255
    v_low: int = 65
    v_high: int = 255

    min_contour_area: float = 30.0
    min_circularity: float = 0.45
    yellow_gain: float = 4.0

    dedupe_radius_cm: float = 9.0


class BallDetector:
    def __init__(self, pixel_coords, floor_coords, model_path="yolov8n.pt", config=None):
        self.cfg = config or DetectorConfig()

        self.pixel_coords = np.asarray(pixel_coords, dtype=np.float32)
        self.floor_coords = np.asarray(floor_coords, dtype=np.float32)
        if self.pixel_coords.shape != (4, 2) or self.floor_coords.shape != (4, 2):
            raise ValueError("pixel_coords and floor_coords must each be shape (4, 2)")

        self.M = cv2.getPerspectiveTransform(self.pixel_coords, self.floor_coords)
        self.model = YOLO(model_path)

        # Persistent map in floor coordinates: [(x_cm, y_cm, conf), ...]
        self.ball_map = []

    def _yellow_mask(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.cfg.h_low, self.cfg.s_low, self.cfg.v_low], dtype=np.uint8)
        upper = np.array([self.cfg.h_high, self.cfg.s_high, self.cfg.v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Reduce noise and fill small gaps for more stable contour/circularity.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _heuristic_roi(self, roi):
        mask = self._yellow_mask(roi)
        yellow_pct = float(cv2.countNonZero(mask)) / float(mask.size)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_contour_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.cfg.min_circularity:
                continue

            color_score = min(yellow_pct * self.cfg.yellow_gain, 1.0)
            conf = (0.6 * circularity) + (0.4 * color_score)

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            candidates.append(
                {
                    "center": (float(cx), float(cy)),
                    "radius": float(radius),
                    "conf": float(conf),
                }
            )
            best_conf = max(best_conf, conf)

        # If no contour survives, still expose low confidence for fallback behavior.
        if not candidates:
            best_conf = min(yellow_pct * self.cfg.yellow_gain, 1.0) * 0.4

        return candidates, float(best_conf)

    def _is_point_in_any_roi(self, px, py, rois):
        for (x1, y1, x2, y2) in rois:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        return False

    def _run_yolo_on_frame(self, frame, uncertain_rois):
        if not uncertain_rois:
            return []

        results = self.model.predict(
            frame,
            conf=self.cfg.yolo_conf_threshold,
            classes=[32],  # COCO sports ball
            imgsz=self.cfg.yolo_imgsz,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                # Keep YOLO detections tied to uncertain zones to reduce false positives.
                if self._is_point_in_any_roi(cx, cy, uncertain_rois):
                    detections.append(
                        {
                            "center": (float(cx), float(cy)),
                            "radius": float(max(x2 - x1, y2 - y1) / 2.0),
                            "conf": conf,
                        }
                    )

        return detections

    def _pixel_to_floor(self, pixel_x, pixel_y):
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        real_world_pos = cv2.perspectiveTransform(point, self.M)
        x_floor, y_floor = real_world_pos[0][0]
        return float(x_floor), float(y_floor)

    def _dedupe_floor_points(self, detections_floor):
        # Greedy merge by confidence, then suppress nearby detections in cm space.
        detections_floor = sorted(detections_floor, key=lambda d: d[2], reverse=True)
        kept = []

        for x_cm, y_cm, conf in detections_floor:
            keep = True
            for kx, ky, _ in kept:
                if (kx - x_cm) ** 2 + (ky - y_cm) ** 2 <= self.cfg.dedupe_radius_cm ** 2:
                    keep = False
                    break
            if keep:
                kept.append((x_cm, y_cm, conf))

        return kept

    def _merge_into_global_map(self, frame_points):
        merged = list(self.ball_map)

        for x_cm, y_cm, conf in frame_points:
            matched_idx = None
            for i, (mx, my, mconf) in enumerate(merged):
                if (mx - x_cm) ** 2 + (my - y_cm) ** 2 <= self.cfg.dedupe_radius_cm ** 2:
                    matched_idx = i
                    break

            if matched_idx is None:
                merged.append((x_cm, y_cm, conf))
            else:
                mx, my, mconf = merged[matched_idx]
                # Smooth position with confidence as weight.
                w_old = max(mconf, 1e-3)
                w_new = max(conf, 1e-3)
                nx = (mx * w_old + x_cm * w_new) / (w_old + w_new)
                ny = (my * w_old + y_cm * w_new) / (w_old + w_new)
                nconf = max(mconf, conf)
                merged[matched_idx] = (nx, ny, nconf)

        self.ball_map = merged

    def run_full_sweep(self, frame):
        h, w = frame.shape[:2]
        heuristic_detections_px = []
        uncertain_rois = []

        for y in range(0, h, self.cfg.grid_stride):
            for x in range(0, w, self.cfg.grid_stride):
                x2 = min(x + self.cfg.grid_size, w)
                y2 = min(y + self.cfg.grid_size, h)
                roi = frame[y:y2, x:x2]
                if roi.size == 0:
                    continue

                candidates, best_conf = self._heuristic_roi(roi)

                for c in candidates:
                    conf = c["conf"]
                    if conf >= self.cfg.heuristic_threshold:
                        cx_local, cy_local = c["center"]
                        heuristic_detections_px.append(
                            {
                                "center": (x + cx_local, y + cy_local),
                                "radius": c["radius"],
                                "conf": conf,
                            }
                        )

                if best_conf < self.cfg.heuristic_threshold:
                    uncertain_rois.append((x, y, x2, y2))

        yolo_detections_px = self._run_yolo_on_frame(frame, uncertain_rois)
        all_detections_px = heuristic_detections_px + yolo_detections_px

        detections_floor = []
        for det in all_detections_px:
            cx, cy = det["center"]
            x_cm, y_cm = self._pixel_to_floor(cx, cy)
            detections_floor.append((x_cm, y_cm, det["conf"]))

        frame_points = self._dedupe_floor_points(detections_floor)
        self._merge_into_global_map(frame_points)

        return self.ball_map


if __name__ == "__main__":
    # 1) Camera pixel coordinates: top-left, top-right, bottom-left, bottom-right
    # Replace these with corners of a floor rectangle visible in your camera.
    pixel_coords = np.float32([
        [100, 400],
        [500, 400],
        [0, 600],
        [600, 600],
    ])

    floor_coords = np.float32([
    [0, 0],
    [400, 0],
    [0, 300],
    [400, 300]
    ])

    detector = BallDetector(pixel_coords, floor_coords)

    # Example usage:
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ok, frame = cap.read()
    #     if not ok:
    #         break
    #     ball_map = detector.run_full_sweep(frame)
    #     print(ball_map)
