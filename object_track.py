## How to Run
#1. Download the video as "YOUR_VIDEO_FILENAME.mp4" or just use the name in github.
#2. Create and activate your Python venv:
  #  - Linux/macOS/WSL: `python3 -m venv venv && source venv/bin/activate`
 #   - Windows: `python -m venv venv && venv\Scripts\activate`
#3. Install dependencies:
 #   - `pip install numpy opencv-python`
#4. Edit the code to set your video filename in `cap = cv2.VideoCapture(...)`.
#5. Run the code:
 #   - `python object_track.py`
#PS i did this on linux.
import cv2
import numpy as np

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    ret, prev_frame = cap.read() #initailize previous frame

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    
    while True:
        ret,frame = cap.read()
        if not ret: break
    
        #gaussian blur for motion robustness
        blurred_prev = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
        #Grayscale motion detection
        prev_gray = cv2.cvtColor(blurred_prev, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, movemask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    
    
        #HSV white detection    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0,0, 160]) #[h,s,v]
        upper = np.array([180, 30, 255])#[h,s,v]
        mask = cv2.inRange(hsv, lower, upper)
    
        #combine motion and color mask
        combined_mask = cv2.bitwise_or(mask, movemask)
    
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        contours,_ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_blob = None
        best_area = 0
    
        #set the perimeter to track
        for C in contours:
            area = cv2.contourArea(C)
            perimeter = cv2.arcLength(C, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            (x,y), radius = cv2.minEnclosingCircle(C)
            #area, circulatory, and radius filter
            if 15 < area < 450 and 0.5 < circularity < 1.3 and 5 < radius < 20:
                    if area > best_area:
                        best_blob = (x, y, radius, area)
            if best_blob:
                x, y, radius, area = best_blob
    
                cv2.circle(frame, (int (x), int (y)), int(radius), (0, 255, 0), 1)
        cv2.imshow("Ball Tracking", frame)
        cv2.imshow("Combined mask", combined_mask)
        #cv2.imshow("HSV mask", mask)
        #cv2.imshow("Motion mask", movemask)
        #print(area, circularity, radius)
    
    
        prev_frame = frame.copy() #update previous frame
    
    
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    