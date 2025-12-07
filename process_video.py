import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from graphmemory.scene_graph_processor import SceneGraphProcessor


def main():
    input_path = Path("test_video.mp4")
    output_video_path = Path("tracked_output.mp4")
    output_json_path = Path("video_objects.json")

    if not input_path.exists():
        sys.exit(f"Input video not found: {input_path}")

    # Use a slightly larger model for better small-object performance.
    model = YOLO("yolo11s.pt")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps if fps and fps > 0 else None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        sys.exit("Unable to read video dimensions.")

    writer_fps = fps if fps and fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, writer_fps, (width, height))

    processor = SceneGraphProcessor(distance_threshold=100.0, end_buffer=5, focus_class="person")
    frames_data = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Lower confidence and increase input size to help with small objects (e.g., cup).
        results = model.track(frame, persist=True, conf=0.15, imgsz=960, verbose=False)
        if not results:
            break
        result = results[0]

        plotted_frame = result.plot()
        writer.write(plotted_frame)

        if frame_index % 50 == 0:
            print(f"Processing frame {frame_index}...")

        timestamp = frame_index * frame_duration if frame_duration else float(frame_index)

        logic_detections = []
        objects = []
        for box in result.boxes:
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item()) if box.cls is not None else -1
            label = model.names.get(class_id, str(class_id))
            confidence = round(float(box.conf.item()), 2) if box.conf is not None else 0.0
            x_center, y_center, w, h = map(float, box.xywhn[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            if track_id is not None:
                logic_detections.append(
                    {
                        "id": track_id,
                        "label": label,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

            objects.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "confidence": confidence,
                    "bbox": {
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": w,
                        "height": h,
                    },
                }
            )

        events = processor.update(logic_detections, timestamp, frame_index)
        for event in events:
            print(event)

        frames_data.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "objects": objects,
            }
        )

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(frames_data, f, indent=2)

    # Persist semantic events and timeline visualization for the paper.
    processor.save_events_to_json("event_log.json")
    processor.generate_timeline_plot("interaction_timeline.png")

    print(f"Saved tracked video to {output_video_path}")
    print(f"Saved detection data to {output_json_path}")


if __name__ == "__main__":
    main()
