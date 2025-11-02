from ultralytics import YOLO
import cv2
import os

def process_video(input_path, output_path):
    """
    Track baseball in the uploaded video using YOLOv8.
    Saves the processed video to output_path.
    Shows live tracking window (only works on local machine).
    """

    # Load YOLO model
    model = YOLO("yolov8m.pt")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open video: {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    trajectory_points = []

    # Create resizable window
    cv2.namedWindow("Baseball Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Baseball Tracking", 960, 540)

    print("Tracking baseball... Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                conf = float(box.conf[0])

                if label in ["sports ball", "baseball"] and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trajectory_points.append((cx, cy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

        # Draw trajectory
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)

        # Write frame to output video
        out.write(frame)

        # Show live tracking
        cv2.imshow("Baseball Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tracking stopped by user")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed video saved to: {output_path}")

    return output_path