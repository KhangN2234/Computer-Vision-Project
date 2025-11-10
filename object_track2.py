from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np

def process_video(input_path, output_path):
    """
    Detect a baseball using YOLOv8 within a user-defined ROI.
    Interpolates missing detections using Pandas.
    Allows manual frame control with A/D and stops YOLO detection after the end frame.
    """

    model = YOLO("yolov8l.pt")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open video: {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # ---------- STEP 1: Frame selection ----------
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
            cv2.putText(display, f"{message} | Frame: {frame_index}/{total_frames}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Frame Selection", display)
            key = cv2.waitKey(0) & 0xFF
            if key == 32:  # Space bar -> confirm
                print(f"Selected frame {frame_index} for: {message}")
                return frame_index
            elif key in [ord('d'), ord('D')]:
                frame_index = min(frame_index + 1, total_frames - 1)
            elif key in [ord('a'), ord('A')]:
                frame_index = max(frame_index - 1, 0)
            elif key == ord('q'):
                print("Selection cancelled by user.")
                return None

    print("Use A/D to navigate frames, SPACE to confirm selection.")
    start_frame = select_frame("Select frame where ball is THROWN")
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
        print("Could not read selected start frame.")
        return

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", 960, 540)
    roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x_roi, y_roi, w_roi, h_roi = roi
    print(f"ROI selected: x={x_roi}, y={y_roi}, w={w_roi}, h={h_roi}")

    # ---------- STEP 3: YOLO Detection ----------
    data = []  # will store [frame_idx, cx, cy]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
        cx, cy = np.nan, np.nan  # default = missing

        results = model(roi_frame, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                conf = float(box.conf[0])
                if label in ["sports ball", "baseball"] and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2 + x_roi, (y1 + y2) // 2 + y_roi
                    break
            if not np.isnan(cx):
                break
        data.append([frame_idx, cx, cy])

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ---------- STEP 4: Interpolate missing detections ----------
    df = pd.DataFrame(data, columns=["frame", "x", "y"])
    df[["x", "y"]] = df[["x", "y"]].interpolate(method='linear')
    df[["x", "y"]] = df[["x", "y"]].fillna(method="bfill").fillna(method="ffill")

    # ---------- STEP 5: Playback ----------
    print("Press A/D to move frames, SPACE to toggle play/pause, Q to quit.")
    cv2.namedWindow("Baseball Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Baseball Tracking", 960, 540)

    frame_index = start_frame
    playing = True

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Draw ROI
        cv2.rectangle(display, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 255, 0), 2)

        # Draw interpolated ball position + trajectory
        if frame_index in df["frame"].values:
            idx = df.index[df["frame"] == frame_index][0]
            cx, cy = int(df.loc[idx, "x"]), int(df.loc[idx, "y"])
            cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)

            # Draw trajectory up to current frame
            for j in range(1, idx + 1):
                if not np.isnan(df.loc[j - 1, "x"]) and not np.isnan(df.loc[j, "x"]):
                    p1 = (int(df.loc[j - 1, "x"]), int(df.loc[j - 1, "y"]))
                    p2 = (int(df.loc[j, "x"]), int(df.loc[j, "y"]))
                    cv2.line(display, p1, p2, (0, 0, 255), 2)

        cv2.putText(display, f"Frame {frame_index}/{end_frame}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(display)
        cv2.imshow("Baseball Tracking", display)

        key = cv2.waitKey(0 if not playing else 30) & 0xFF
        if key in [ord('q'), 27]:  # quit
            break
        elif key in [ord('a'), ord('A')]:
            frame_index = max(start_frame, frame_index - 1)
            playing = False
        elif key in [ord('d'), ord('D')]:
            frame_index = min(end_frame, frame_index + 1)
            playing = False
        elif key == 32:  # space
            playing = not playing
        elif playing:
            frame_index = min(end_frame, frame_index + 1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")


if __name__ == "__main__":
    video_path = input("Enter path to video: ").strip()
    output_path = "tracked_output.mp4"
    process_video(video_path, output_path)
