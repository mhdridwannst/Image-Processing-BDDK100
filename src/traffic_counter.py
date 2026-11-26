import argparse
import cv2
from ultralytics import YOLO

COLORS = {
    "car": (0, 255, 127),
    "motor": (255, 200, 0),
    "truck": (0, 165, 255),
    "bus": (255, 0, 255),
    "person": (255, 255, 0),
}


def run_traffic_analytics(weights_path: str, source_path: str, output_path: str, conf: float = 0.35):
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counting_line_x = int(width * 0.50)
    counted_ids = set()
    class_counts = {"car": 0, "motor": 0, "truck": 0, "bus": 0, "person": 0}
    direction_counts = {"Left -> Right": 0, "Right -> Left": 0}
    track_history_x = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)
        res = results

        cv2.line(frame, (counting_line_x, 0), (counting_line_x, height), (0, 255, 255), 2)
        cv2.putText(frame, "SENSOR LINE", (counting_line_x + 8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            track_ids = res.boxes.id.int().cpu().numpy()
            cls_ids = res.boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[cls_id]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = COLORS.get(cls_name, (200, 200, 200))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                label = f"#{track_id} {cls_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                prev_x = track_history_x.get(track_id, cx)
                if track_id not in counted_ids:
                    if prev_x < counting_line_x <= cx:
                        counted_ids.add(track_id)
                        direction_counts["Left -> Right"] += 1
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                    elif prev_x > counting_line_x >= cx:
                        counted_ids.add(track_id)
                        direction_counts["Right -> Left"] += 1
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                track_history_x[track_id] = cx

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (265, 225), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        cv2.putText(frame, "TRAFFIC FLOW ANALYTICS", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y_pos = 58
        for k, v in class_counts.items():
            cv2.putText(frame, f"{k.upper():<7}: {v}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[k], 2)
            y_pos += 21

        cv2.line(frame, (20, y_pos - 5), (250, y_pos - 5), (100, 100, 100), 1)
        y_pos += 15
        for d_label, d_val in direction_counts.items():
            cv2.putText(frame, f"{d_label}: {d_val}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            y_pos += 20

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done. Output saved to {output_path}")
    print("Vehicle Counts:", class_counts)
    print("Directional Flow:", direction_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bidirectional Traffic Analytics with YOLOv8 + ByteTrack")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    run_traffic_analytics(args.weights, args.source, args.output, args.conf)
