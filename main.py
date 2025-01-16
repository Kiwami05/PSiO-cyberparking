import cv2
import os
import csv
from datetime import datetime

# Ścieżka do filmu i pliku CSV
video_path = 'data/Film_2.mp4'
csv_path = 'data/parking_spots.csv'
log_path = video_path.replace('.mp4', '.log')
output_video_path = video_path.replace('.mp4', '-przetworzony.mp4')

# Tworzy tło za pomocą BackgroundSubtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# Dodajemy słownik do śledzenia stanu miejsc parkingowych
parking_state = {}


def log_event(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry, end='')
    with open(log_path, 'a') as log_file:
        log_file.write(log_entry)


def draw_parking_spots(frame, spots):
    for i, (x, y, w, h) in enumerate(spots):
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def load_parking_spots(csv_file):
    spots = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y, w, h = map(int, row)
                spots.append((x, y, w, h))
    return spots


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
            x, y, w, h = cv2.boundingRect(contour)
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


cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

if os.path.exists(csv_path):
    parking_spots = load_parking_spots(csv_path)
    log_event(f"Wczytano {len(parking_spots)} miejsc parkingowych.")
    parking_state = {i: None for i in range(len(parking_spots))}
else:
    log_event("Brak pliku CSV z miejscami parkingowymi.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = back_sub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    parking_state = update_parking_status(contours, parking_spots, parking_state)
    draw_parking_spots(frame, parking_spots)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Samochod", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Parking - Przetwarzanie Wideo', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
log_event("Przetwarzanie zakończone.")
