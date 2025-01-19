import cv2
import numpy as np

# Load the video
# 4,7,8
video_path = 'recordings/recording-10.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Tracker to store previous frame detections
detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (optional for faster processing)
    # frame = cv2.resize(frame, (640, 480))
    # frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract the Saturation channel
    saturation = hsv_frame[:, :, 1]

    # Threshold based on saturation values
    _, sat_mask = cv2.threshold(saturation, 125, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up the mask
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []
    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # Adjust size threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            current_detections.append((x, y, w, h))
            # Slightly expand bounding box
            pad_x, pad_y = int(w * 0.05), int(h * 0.05)
            x, y, w, h = x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update detections across frames
    if len(current_detections) == 0 and len(detections) > 0:
        # If no new detections, reuse previous ones
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    else:
        detections = current_detections

    # Display results
    cv2.imshow('Parking Lot', frame)
    cv2.imshow('Saturation Mask', sat_mask)

    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
