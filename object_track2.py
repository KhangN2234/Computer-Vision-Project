from ultralytics import YOLO
import cv2

def process_video(input_path, output_path):
    """
    Track baseball in the uploaded video using YOLOv8.
    User selects start (thrown) and end (caught/hit) frames with arrow keys + spacebar.
    """

    model = YOLO("yolov8l.pt")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # ---------- STEP 1: Select throw start and end frames ----------
    cv2.namedWindow("Frame Selection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame Selection", 960, 540)

    def select_frame(message):
        frame_index = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(display, f"{message}\nFrame: {frame_index}/{total_frames}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Frame Selection", display)

            key = cv2.waitKey(0) & 0xFF

            if key == 32:  # Space bar -> confirm
                print(f"✅ Selected frame {frame_index} for: {message}")
                return frame_index
            elif key == ord('d') or key == 83:  # Right arrow or 'd'
                frame_index = min(frame_index + 1, total_frames - 1)
            elif key == ord('a') or key == 81:  # Left arrow or 'a'
                frame_index = max(frame_index - 1, 0)
            elif key == ord('q'):
                print("❌ Selection cancelled by user.")
                return None

    print("🎯 Use A/D to navigate, SPACE to confirm.")
    start_frame = select_frame("Select frame where ball is THROWN ")
    if start_frame is None:
        return
    end_frame = select_frame("Select frame where ball is CAUGHT or HIT")
    if end_frame is None:
        return

    cv2.destroyWindow("Frame Selection")

    # ---------- STEP 2: ROI selection ----------
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read selected start frame.")
        return

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", 960, 540)
    roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x_roi, y_roi, w_roi, h_roi = roi
    print(f"📍 ROI selected: x={x_roi}, y={y_roi}, w={w_roi}, h={h_roi}")

    # ---------- STEP 3: YOLO tracking ----------
    trajectory_points = []
    cv2.namedWindow("Baseball Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Baseball Tracking", 960, 540)

    print("🚀 Tracking baseball between selected frames... Press 'q' to quit early.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
        results = model(roi_frame, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                conf = float(box.conf[0])

                if label in ["sports ball", "baseball"] and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 += x_roi; x2 += x_roi; y1 += y_roi; y2 += y_roi
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trajectory_points.append((cx, cy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Baseball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Tracking stopped by user.")
            break

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed video saved to: {output_path}")

# --- For quick testing ---
if __name__ == "__main__":
    video_path = input("Enter path to video: ").strip()
    output_path = "tracked_output.mp4"
    process_video(video_path, output_path)
