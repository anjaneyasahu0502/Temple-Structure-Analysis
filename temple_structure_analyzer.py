import os
import time
import cv2
import numpy as np
import json
from datetime import datetime

# Try to import YOLO, but fail soft if unavailable
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class TempleStructureAnalyzer:
    def __init__(self, yolo_model_path=None, use_windows=False):
        """
        Initialize the Temple Structure Analyzer.
        - yolo_model_path: path to YOLOv8 model (e.g., 'yolov8n.pt' or 'yolov8n-seg.pt').
          If None or ultralytics is unavailable, YOLO is skipped gracefully.
        - use_windows: True to show cv2.imshow windows (off by default for headless runs).
        """
        self.model = None
        self.model_names = None
        self.is_seg_model = False
        if YOLO_AVAILABLE and yolo_model_path:
            try:
                self.model = YOLO(yolo_model_path)
                # Detect if this is a segmentation model
                self.is_seg_model = hasattr(self.model, "task") and str(self.model.task).lower() == "segment"
                self.model_names = getattr(self.model, "names", None)
            except Exception:
                self.model = None  # fail soft

        self.use_windows = use_windows
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "yolo_loaded": bool(self.model),
            "is_seg_model": bool(self.is_seg_model),
            "frame_analyses": []
        }

    # ------------------------ SHAPE APPROXIMATION ------------------------
    def shape_approximation(self, frame, mask=None):
        # ROI focus if mask exists
        img = frame.copy()
        if mask is not None and np.any(mask):
            img = cv2.bitwise_and(img, img, mask=mask)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast boost to make contours pop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lower area threshold so you actually see structures
        min_area = frame.shape[0] * frame.shape[1] * 0.002  # 0.2% of frame
        significant = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Fallback if nothing passed
        if not significant:
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > min_area * 0.5:
                    significant = [largest]

        out = frame.copy()
        info = []

        if not significant:
            self._put_multiline_text(out, ["No significant structures"], (20, 60), 0.9, (255, 255, 255))
            return out, info

        for i, contour in enumerate(significant):
            cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(out, [hull], -1, (0, 0, 255), 2)

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull_area = cv2.contourArea(hull)

            epsilon = 0.02 * max(1.0, perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            convexity = (area / hull_area) if hull_area > 0 else 0.0
            shape_type = self._classify_shape(len(approx), convexity)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

            lines = [
                f"Structure {i+1}",
                f"Area: {area:.0f}",
                f"Perim: {perimeter:.0f}",
                f"Verts: {len(approx)}",
                f"Convx: {convexity:.2f}",
                f"Type: {shape_type}"
            ]
            self._put_multiline_text(out, lines, (cx - 60, max(20, cy - 50)), 0.5, (255, 255, 255))

            info.append({
                "id": i,
                "area": float(area),
                "perimeter": float(perimeter),
                "vertices": int(len(approx)),
                "convexity": float(convexity),
                "shape_type": shape_type,
                "centroid": [int(cx), int(cy)]
            })

        return out, info

    def _classify_shape(self, vertices, convexity):
        if vertices == 3:
            return "Triangular Roof/Spire"
        elif vertices == 4:
            return "Rectangular/Base" if convexity > 0.85 else "Complex Quad"
        elif vertices > 8:
            return "Circular/Dome" if convexity > 0.9 else "Complex"
        else:
            return f"Polygon ({vertices})"

    # ------------------------ FEATURE MATCHING ------------------------
    def temple_feature_matching(self, frame1, frame2, roi_mask1=None, roi_mask2=None):
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if roi_mask1 is not None and np.any(roi_mask1):
            g1 = cv2.bitwise_and(g1, roi_mask1)
        if roi_mask2 is not None and np.any(roi_mask2):
            g2 = cv2.bitwise_and(g2, roi_mask2)

        # SIFT -> ORB fallback
        sift = None
        try:
            sift = cv2.SIFT_create()
        except Exception:
            pass
        if sift is not None:
            kp1, des1 = sift.detectAndCompute(g1, None)
            kp2, des2 = sift.detectAndCompute(g2, None)
            descriptor = "SIFT"
        else:
            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(g1, None)
            kp2, des2 = orb.detectAndCompute(g2, None)
            descriptor = "ORB"

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            vis = np.hstack([frame1, frame2])
            self._put_multiline_text(vis, ["No robust keypoints"], (10, 30), 0.8, (255, 255, 255))
            return vis, []

        norm = cv2.NORM_L2 if descriptor == "SIFT" else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm, crossCheck=False)
        matches_knn = bf.knnMatch(des1, des2, k=2)

        good = []
        features = []
        for m_n in matches_knn:
            if len(m_n) < 2:
                continue
            m, n = m_n[:2]
            if m.distance < 0.7 * n.distance:
                good.append(m)
                ftype = self._classify_temple_feature(kp1[m.queryIdx], g1)
                features.append({
                    "keypoint1": [float(kp1[m.queryIdx].pt[0]), float(kp1[m.queryIdx].pt[1])],
                    "keypoint2": [float(kp2[m.trainIdx].pt[0]), float(kp2[m.trainIdx].pt[1])],
                    "distance": float(m.distance),
                    "feature_type": ftype
                })

        good = sorted(good, key=lambda x: x.distance)[:50]
        vis = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        type_counts = {}
        for f in features:
            t = f["feature_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        summary = [f"Matches: {len(good)}", "Types:"]
        for k, v in sorted(type_counts.items(), key=lambda kv: -kv[1])[:4]:
            summary.append(f" - {k}: {v}")
        self._put_multiline_text(vis, summary, (10, 30), 0.7, (255, 255, 255))

        return vis, features

    def _classify_temple_feature(self, keypoint, gray_image):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        h, w = gray_image.shape
        roi_size = 20
        x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
        x2, y2 = min(w, x + roi_size), min(h, y + roi_size)
        roi = gray_image[y1:y2, x1:x2]
        if roi.size == 0:
            return "Unknown"

        variance = float(np.var(roi))
        mean_intensity = float(np.mean(roi))

        if variance > 1000 and mean_intensity > 100:
            return "Architectural Detail"
        elif variance > 500:
            return "Structural Edge"
        elif mean_intensity > 150:
            return "Surface Feature"
        else:
            return "Shadow/Depth Feature"

    # ------------------------ SEGMENTATION ------------------------
    def backtracking_segmentation(self, frame, target_class_mask=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if target_class_mask is not None and np.any(target_class_mask):
            gray = cv2.bitwise_and(gray, target_class_mask)

        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

        optimal_mask = None
        min_error = float('inf')
        best_thr = 0

        for thr in range(50, 200, 25):
            _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            error = np.sum(np.abs(mask.astype(np.int32) - adaptive.astype(np.int32)))

            density = float(np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255))
            if density < 0.1 or density > 0.9:
                error *= 2

            if error < min_error:
                min_error = error
                optimal_mask = mask
                best_thr = thr

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        optimal_mask = cv2.morphologyEx(optimal_mask, cv2.MORPH_CLOSE, kernel)
        optimal_mask = cv2.morphologyEx(optimal_mask, cv2.MORPH_OPEN, kernel)

        # Colorize the mask for visibility and alpha-blend on original
        seg = frame.copy()
        color_mask = np.zeros_like(seg)
        color_mask[:, :, 1] = optimal_mask  # green channel
        seg = cv2.addWeighted(seg, 1.0, color_mask, 0.35, 0.0)

        info = {
            "optimal_threshold": int(best_thr),
            "segmentation_error": float(min_error),
            "mask_density": float(np.sum(optimal_mask) / (optimal_mask.shape[0] * optimal_mask.shape[1] * 255))
        }

        self._put_multiline_text(seg, [
            "Segmentation",
            f"Threshold: {info['optimal_threshold']}",
            f"Density: {info['mask_density']:.3f}",
            f"Error: {info['segmentation_error']:.0f}"
        ], (10, 30), 0.7, (255, 255, 255))

        return seg, optimal_mask, info

    # ------------------------ YOLO DETECTION ------------------------
    def detect_temple_structures(self, frame):
        """
        YOLO (if available). If segmentation model, union masks.
        Otherwise, use dilated box-union as coarse ROI.
        """
        h, w = frame.shape[:2]
        temple_mask = np.zeros((h, w), dtype=np.uint8)
        detections = []
        draw = frame.copy()

        if not self.model:
            cv2.putText(draw, "Detections: 0 (YOLO disabled)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return draw, temple_mask, detections

        results = self.model(frame)[0]

        if self.is_seg_model and hasattr(results, "masks") and results.masks is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0]); conf = float(box.conf[0])
                xyxy = list(map(int, box.xyxy[0].tolist()))
                label = self.model_names[cls_id] if self.model_names else str(cls_id)
                detections.append({
                    "class": label, "confidence": conf, "bbox": xyxy,
                    "area": int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                })
            m = results.masks.data.cpu().numpy()  # (N, H, W)
            for mi in m:
                mm = (mi * 255).astype(np.uint8)
                temple_mask = cv2.bitwise_or(temple_mask, mm)
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 220, 220), 2)
                cv2.putText(draw, f"{d['class']} {d['confidence']:.2f}",
                            (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
        else:
            for box in results.boxes:
                cls_id = int(box.cls[0]); conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.model_names[cls_id] if self.model_names else str(cls_id)
                if conf < 0.3:
                    continue
                cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(draw, f"{label} {conf:.2f}",
                            (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.rectangle(temple_mask, (x1, y1), (x2, y2), 255, -1)
                detections.append({
                    "class": label, "confidence": conf, "bbox": [x1, y1, x2, y2],
                    "area": int((x2 - x1) * (y2 - y1))
                })
            if np.any(temple_mask):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                temple_mask = cv2.dilate(temple_mask, k)

        # Also tint the detection mask for visibility
        if np.any(temple_mask):
            tint = draw.copy()
            cm = np.zeros_like(draw); cm[:, :, 2] = temple_mask  # red channel
            tint = cv2.addWeighted(tint, 1.0, cm, 0.25, 0.0)
            draw = tint

        cv2.putText(draw, f"Detections: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return draw, temple_mask, detections

    # ------------------------ TEXT UTILS ------------------------
    def _put_multiline_text(self, img, lines, org, scale, color, thickness=2, line_h=22):
        x, y = org
        for i, line in enumerate(lines):
            yy = y + i * line_h
            cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # ------------------------ WRITER HELPERS ------------------------
    def _resolve_output_path(self, output_path):
        """Accept a directory or a file path; ensure parent exists; default .mp4 if no ext."""
        if os.path.isdir(output_path):
            ts = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_path, f"temple_analysis_output_{ts}.mp4")
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        root, ext = os.path.splitext(output_path)
        if not ext:
            output_path = root + ".mp4"
        return output_path

    def _open_video_writer(self, path, fps, frame_size):
        """Try MP4; if it fails, fallback to AVI."""
        path = self._resolve_output_path(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, float(fps), frame_size)
        if out.isOpened():
            print(f"[OK] Writing MP4 to: {path} size={frame_size} fps={fps}")
            return out, path

        root, _ = os.path.splitext(path)
        alt_path = root + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(alt_path, fourcc, float(fps), frame_size)
        if out.isOpened():
            print(f"[OK] MP4 failed; writing AVI instead: {alt_path} size={frame_size} fps={fps}")
            return out, alt_path

        raise RuntimeError(f"Could not open VideoWriter for '{path}' or '{alt_path}'. Check codecs/permissions.")

    # ------------------------ MAIN PIPE ------------------------
    def process_video(self, video_path, output_path="temple_analysis_output.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file '{video_path}' not found or cannot be opened.")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        out_path_final = None

        frame_count = 0
        prev_frame = None

        print(f"Processing {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # --- Always run full analysis on every frame (VISIBLE RESULTS) ---
            detected_frame, temple_mask, detections = self.detect_temple_structures(frame.copy())
            shape_frame, shape_info = self.shape_approximation(frame.copy(),
                                                               temple_mask if np.any(temple_mask) else None)

            if prev_frame is not None:
                matched_frame, match_info = self.temple_feature_matching(
                    prev_frame, frame.copy(),
                    roi_mask1=temple_mask if np.any(temple_mask) else None,
                    roi_mask2=temple_mask if np.any(temple_mask) else None
                )
            else:
                matched_frame, match_info = frame.copy(), []

            segmented_frame, seg_mask, seg_info = self.backtracking_segmentation(
                frame.copy(), target_class_mask=temple_mask if np.any(temple_mask) else None
            )

            # Store analysis (for JSON)
            self.analysis_results["frame_analyses"].append({
                "frame_number": int(frame_count),
                "detections": detections,
                "structures": shape_info,
                "matches": match_info,
                "segmentation": seg_info
            })
            prev_frame = frame.copy()

            # Compose 2×2 grid
            half_w, half_h = width // 2, height // 2
            def rsz(x): return cv2.resize(x, (half_w, half_h))
            top_row = np.hstack([rsz(detected_frame), rsz(shape_frame)])
            bottom_row = np.hstack([rsz(matched_frame), rsz(segmented_frame)])
            combined = np.vstack([top_row, bottom_row])

            # Panel titles + progress
            self._put_multiline_text(combined, ["Temple Detection"], (10, 30), 0.9, (255, 255, 255))
            self._put_multiline_text(combined, ["Shape Analysis"], (half_w + 10, 30), 0.9, (255, 255, 255))
            self._put_multiline_text(combined, ["Feature Matching"], (10, half_h + 30), 0.9, (255, 255, 255))
            self._put_multiline_text(combined, ["Segmentation"], (half_w + 10, half_h + 30), 0.9, (255, 255, 255))
            progress = (frame_count / max(1, total_frames)) * 100.0
            cv2.putText(combined, f"Progress: {progress:.1f}%",
                        (combined.shape[1] - 260, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Lazy-open writer with actual combined size
            if out is None:
                h_out, w_out = combined.shape[:2]
                out, out_path_final = self._open_video_writer(output_path, fps, (w_out, h_out))

            # Sanity check: size must remain constant
            if (combined.shape[1], combined.shape[0]) != (w_out, h_out):
                raise ValueError(
                    f"Frame size changed from {(w_out, h_out)} to {(combined.shape[1], combined.shape[0])}"
                )

            out.write(combined)

            if self.use_windows:
                cv2.imshow("Temple Structure Analysis", cv2.resize(combined, (1200, 800)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

        cap.release()
        if out is not None:
            out.release()
        if self.use_windows:
            cv2.destroyAllWindows()

        with open('temple_analysis_results.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2)

        print("\nAnalysis complete!")
        print(f"Output video saved: {out_path_final if out_path_final else output_path}")
        print("Analysis results saved: temple_analysis_results.json")
        print(f"Total frames processed: {frame_count}")
        print(f"Detailed analysis performed on: {len(self.analysis_results['frame_analyses'])} frames")

        return self.analysis_results


def main():
    # Update these to your paths. Raw strings recommended on Windows.
    analyzer = TempleStructureAnalyzer(yolo_model_path="yolov8n.pt", use_windows=False)

    video_path = "The mystery of Brihadeesvara Temple, Thanjavur. #cholaempire.mp4"
    output_path = "temple_structure_analysis_output.mp4"

    try:
        results = analyzer.process_video(video_path, output_path)

        print("\n=== TEMPLE STRUCTURE ANALYSIS SUMMARY ===")
        total_detections = sum(len(f["detections"]) for f in results["frame_analyses"])
        total_structures = sum(len(f["structures"]) for f in results["frame_analyses"])
        total_matches = sum(len(f["matches"]) for f in results["frame_analyses"])

        print(f"Total object detections: {total_detections}")
        print(f"Total structural elements identified: {total_structures}")
        print(f"Total feature matches: {total_matches}")

        structure_types = {}
        for f in results["frame_analyses"]:
            for s in f["structures"]:
                t = s["shape_type"]
                structure_types[t] = structure_types.get(t, 0) + 1
        print("\nStructure Types Detected:")
        for t, c in sorted(structure_types.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {t}: {c}")

    except FileNotFoundError as e:
        print(str(e))
        print("Please ensure the video path is correct.")
    except Exception as e:
        print(f"Error during processing: {str(e)}")



if __name__ == "__main__":
    main()