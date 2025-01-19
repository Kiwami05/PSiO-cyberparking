import cv2
import os
import csv
from datetime import datetime
import pytesseract
import numpy as np

# Ścieżka do Tesseract (jeśli wymaga ustawienia własnej ścieżki)
# Linux
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# Windows
# ???

# Ścieżka do filmu i pliku CSV
video_path = 'recordings/recording-01.mp4'
csv_parking_path = 'data/parking_spots.csv'
csv_gates_path = 'data/gates.csv'

log_path = video_path.replace('.mp4', '.log')
output_video_path = video_path.replace('.mp4', '-przetworzony.mp4')

# Tworzy tło za pomocą BackgroundSubtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)


# Car detection functions

def detect_cars_by_saturation(frame, min_saturation=150, area_threshold=2500, padding=None):
    """
    Detect cars based on high saturation regions in the HSV color space.

    :param frame: Input video frame (BGR format).
    :param min_saturation: Minimum saturation value for thresholding.
    :param area_threshold: Minimum contour area to be considered a car.
    :param padding: Optional tuple (pad_x, pad_y) to expand bounding boxes.
    :return: List of bounding boxes [(x, y, w, h)].
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Convert frame to HSV and extract saturation channel
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv_frame[:, :, 1]

    # Threshold to isolate high-saturation areas
    _, sat_mask = cv2.threshold(saturation, min_saturation, 255, cv2.THRESH_BINARY)

    # Clean the mask
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            if padding:
                pad_x, pad_y = padding
                x, y, w, h = x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y
            detections.append((x, y, w, h))
    return detections


def draw_detected_cars(frame, detections):
    """
    Draws bounding boxes for detected cars on the frame.

    :param frame: Input video frame.
    :param detections: List of bounding boxes [(x, y, w, h)].
    """
    for x, y, w, h in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Samochod", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def log_event(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry, end='')
    with open(log_path, 'a') as log_file:
        log_file.write(log_entry)


def load_csv(file_path):
    elements = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y, w, h = map(int, row)
                elements.append((x, y, w, h))
    return elements


def draw_parking_spots(frame, spots, state):
    for i, (x, y, w, h) in enumerate(spots):
        color = (0, 255, 0) if state.get(i) is None else (0, 0, 255)
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_gates(frame, gates, state):
    for i, (x, y, w, h) in enumerate(gates):
        color = (0, 0, 255) if not state.get(i, False) else (0, 255, 0)
        label = f"Bramka {'wjazdowa' if i == 0 else 'wyjazdowa'} - {'otwarta' if state.get(i, False) else 'zamknieta'}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def load_parking_spots(csv_file):
    spots = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y, w, h = map(int, row)
                spots.append((x, y, w, h))
    return spots


def check_gate_occupation(contours, gates, max_distance=50):
    states = {}
    for i, (gate_x, gate_y, gate_width, gate_height) in enumerate(gates):
        car_x, car_y, car_width, car_height = contours
        # 1. Bramka wjazdowa
        # 2. Tył auta przed bramką
        # 3. Przód auta
        if i == 0 and car_y <= gate_y + gate_height and abs(gate_y - car_y) <= max_distance:  # Bramka wjazdowa
            states[i] = True
            break
        if i == 1 and car_y + car_height >= gate_y and abs(gate_y - car_y) <= max_distance:  # Bramka wyjazdowa
            states[i] = True
            break
        states.setdefault(i, False)
    return states


def intersection_over_union(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height
    rect1_area = w1 * h1
    if rect1_area == 0:
        return 0
    return intersection_area / rect1_area


def update_parking_status(contours, spots, state, iou_threshold=0.5):
    new_state = state.copy()
    for i, spot in enumerate(spots):
        spot_occupied = False
        for contour in contours:
            x, y, w, h = contour
            iou = intersection_over_union(spot, (x, y, w, h))
            if iou > iou_threshold:
                spot_occupied = True
                new_state[i] = (x, y, w, h)
                if state.get(i) is None:
                    log_event(f"Samochód {new_state[i]} zajął Miejsce {i + 1}")
                break
        if not spot_occupied and state.get(i) is not None:
            log_event(f"Miejsce {i + 1} zostało zwolnione.")
            new_state[i] = None
    return new_state


def detect_license_plate(frame, contour, padding=10):
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = max(0, x - padding), max(0, y - padding), w + 2 * padding, h + 2 * padding
    cropped = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, (x, y, w, h)


def read_license_plate(roi):
    try:
        text = pytesseract.image_to_string(roi, config='--psm 6')
        return text.strip()
    except Exception as e:
        log_event(f"Nie udało się odczytać tablicy rejestracyjnej: {e}")
        return None


# Ładowanie wideo
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Ładowanie współrzędnych miejsc parkingowych i bramek
parking_spots = load_csv(csv_parking_path)
gates = load_csv(csv_gates_path)
print(f"Wczytano {len(parking_spots)} miejsc parkingowych i {len(gates)} bramki.")
print(f"Przetwarzanie {video_path}")

parking_state = {i: False for i in range(len(parking_spots))}
gate_state = {i: False for i in range(len(gates))}

# Wczytywanie wideo i przygotowanie pierwszej klatki
ret, first_frame = cap.read()
if not ret:
    log_event("Nie można wczytać klatki wideo!")
    exit()

# Rozszerzona pętla główna z zastosowaniem analizy koloru i kontrastu
# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Background subtraction for motion detection (optional)
    fg_mask = back_sub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Detect cars using saturation
    car_detections = detect_cars_by_saturation(frame)

    # Draw detected cars
    draw_detected_cars(frame, car_detections)

    # Update parking status
    parking_state = update_parking_status(car_detections, parking_spots, parking_state)

    # Draw parking spots and gates
    draw_parking_spots(frame, parking_spots, parking_state)
    draw_gates(frame, gates, gate_state)

    # Write and display the processed frame
    out.write(frame)
    cv2.imshow('Parking - Przetwarzanie Wideo', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
log_event("Przetwarzanie zakończone.")
