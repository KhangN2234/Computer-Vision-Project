import cv2
import numpy as np

def calibrate_distance_from_video(input_path, real_distance_meters=18.44):
    """
    Allows user to draw a calibration line (pitcher -> batter) on the first frame.
    Returns meters per pixel ratio for real-world distance conversion.
    """
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read video for calibration.")
        return None

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 960, 540)
    print("Draw a line from the pitcher to the batter (L-click start, R-click end).")
    cv2.putText(frame, "Select the point from the pitcher's base to first base",
                (30,40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255),2)
    points = []
    cv2.putText(frame, "LEFT-CLICK for starting point RIGHT-CLICK for endpoint",
                (30,100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255),2)
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.clear()
            points.append((x, y))
            # Draw a visible circle for the first point
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow("Calibration", frame)
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) == 1:
            points.append((x, y))
            # Draw a visible circle for the second point
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            # Draw a thick line connecting the points
            cv2.line(frame, points[0], points[1], (0, 255, 0), 3)
            cv2.imshow("Calibration", frame)

    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        print("Calibration failed: two points (pitcher and batter) required.")
        return None

    pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    meters_per_pixel = real_distance_meters / pixel_distance
    print(f"Calibration complete: {meters_per_pixel:.6f} meters/pixel")

    return meters_per_pixel
