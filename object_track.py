import cv2
import numpy as np
import os

def process_video(input_path, output_path):
    """
    Process video to track baseball regardless of size.
    Optimized for baseball games with varying camera angles and distances.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open input video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try multiple codecs - fallback if one doesn't work
    codecs_to_try = [
        ('mp4v', 'MP4V'),  # Most reliable on Windows
        ('XVID', 'XVID'),  # Good alternative
        ('MJPG', 'MJPG'),  # Motion JPEG fallback
    ]
    
    out = None
    codec_used = None
    for fourcc_str, codec_name in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        test_path = output_path
        out = cv2.VideoWriter(test_path, fourcc, fps, (frame_width, frame_height))
        if out.isOpened():
            codec_used = codec_name
            print(f"✅ Using codec: {codec_name}")
            break
        else:
            if out:
                out.release()
    
    if not out or not out.isOpened():
        print("❌ Could not create output file with any codec.")
        cap.release()
        return

    print("✅ Processing video ...")
    
    # Tracking variables
    prev_gray = None
    trail = []
    frame_count = 0
    lost_frames = 0
    predicted_pos = None
    
    # Adaptive parameters based on video resolution
    video_area = frame_width * frame_height
    min_area = max(5, int(video_area / 100000))  # Very small for distant balls
    max_area = min(2000, int(video_area / 1000))  # Larger for close-up balls
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Motion detection - compare with previous frame
        motion_mask = np.zeros_like(blurred)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, blurred)
            # Adaptive threshold - lower for fast motion
            _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect nearby motion regions
            motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.dilate(motion_mask, motion_kernel, iterations=1)
        
        prev_gray = blurred.copy()
        
        # Color detection - white/off-white for baseball
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Expanded range to catch various lighting conditions
        white_lower = np.array([0, 0, 130])
        white_upper = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine motion and color masks
        combined = cv2.bitwise_and(motion_mask, white_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        
        # Adaptive max distance based on video resolution and fps
        max_jump = int(fps * 0.15 * (frame_width / 800))  # ~100-200 pixels
        
        for c in contours:
            area = cv2.contourArea(c)
            
            # Size filter - very permissive for various ball sizes
            if area < min_area or area > max_area:
                continue
            
            # Calculate circularity (baseballs are roughly circular)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            
            # Lower circularity threshold to handle motion blur
            if circularity < 0.3:
                continue
            
            # Get bounding properties
            (x, y), radius = cv2.minEnclosingCircle(c)
            x, y, radius = int(x), int(y), int(radius)
            
            # Calculate score for candidate selection
            score = area * circularity
            
            # Bonus for being near predicted position or previous trail
            if trail:
                last_pos = trail[-1]
                distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                
                # Reject if too far from last known position
                if distance > max_jump:
                    continue
                
                # Favor candidates closer to trail
                proximity_bonus = 100 / (1 + distance / 10)
                score += proximity_bonus
            elif predicted_pos is not None:
                # Use predicted position if we lost the ball
                distance = np.sqrt((x - predicted_pos[0])**2 + (y - predicted_pos[1])**2)
                if distance > max_jump * 2:
                    continue
                proximity_bonus = 50 / (1 + distance / 10)
                score += proximity_bonus
            
            # Bonus for aspect ratio (prefer circular)
            if len(c) >= 5:
                _, (w, h), _ = cv2.fitEllipse(c)
                if h > 0:
                    aspect_ratio = min(w, h) / max(w, h)
                    score += aspect_ratio * 50
            
            # Select best candidate
            if score > best_score:
                best_score = score
                best_candidate = (x, y, radius, area)
        
        # Update tracking
        if best_candidate:
            x, y, r, area = best_candidate
            trail.append((x, y))
            
            # Limit trail length
            if len(trail) > 150:
                trail.pop(0)
            
            # Reset lost frames and prediction
            lost_frames = 0
            predicted_pos = None
            
        else:
            # Ball not detected - use prediction if available
            lost_frames += 1
            
            if trail and len(trail) >= 2:
                # Predict next position based on velocity
                dx = trail[-1][0] - trail[-2][0]
                dy = trail[-1][1] - trail[-2][1]
                predicted_pos = (trail[-1][0] + dx, trail[-1][1] + dy)
                
                # Draw predicted position (faded)
                if 0 <= int(predicted_pos[0]) < frame_width and 0 <= int(predicted_pos[1]) < frame_height:
                    cv2.circle(frame, (int(predicted_pos[0]), int(predicted_pos[1])), 5, (128, 128, 128), 2)
                    cv2.putText(frame, "Predicted", (int(predicted_pos[0]) + 10, int(predicted_pos[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Stop tracking if lost for too long
            if lost_frames > int(fps * 0.5):  # 0.5 seconds
                if trail:
                    trail.clear()
                predicted_pos = None
        
        # Draw trail
        if trail:
            # Draw trail lines with gradient
            for i in range(len(trail) - 1):
                pt1 = trail[i]
                pt2 = trail[i + 1]
                
                # Color gradient: blue (old) -> green -> yellow (new)
                alpha = i / len(trail) if len(trail) > 0 else 0
                if alpha < 0.5:
                    # Blue to green
                    color = (int(255 * (1 - alpha * 2)), int(255 * alpha * 2), 0)
                else:
                    # Green to yellow
                    color = (0, 255, int(255 * (alpha - 0.5) * 2))
                
                thickness = max(1, int(4 * (1 - alpha * 0.7)))
                cv2.line(frame, pt1, pt2, color, thickness)
            
            # Draw current ball position
            if best_candidate:
                x, y, r, area = best_candidate
                cv2.circle(frame, (x, y), max(3, r), (0, 255, 0), 2)  # Green circle
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red center
                
                # Draw trail points
                for i, point in enumerate(trail[-20:]):  # Only show last 20 points clearly
                    alpha = i / 20 if 20 > 0 else 0
                    color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                    cv2.circle(frame, point, 2, color, -1)
        
        # Display info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if trail:
            cv2.putText(frame, f"Ball Tracked ({len(trail)} pts)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif lost_frames > 0:
            cv2.putText(frame, f"Searching... ({lost_frames} frames)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    print(f"✅ Processed video saved to {output_path}")
