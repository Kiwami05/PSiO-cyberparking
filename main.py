import cv2
import os
import csv
from datetime import datetime
import pytesseract
import numpy as np

# Ścieżka do Tesseract (jeśli wymaga ustawienia własnej ścieżki)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Ścieżka do filmu i pliku CSV
video_path = 'data/Film_18.mp4'
csv_path = 'data/parking_spots.csv'
log_path = video_path.replace('.mp4', '-alt.log')
output_video_path = video_path.replace('.mp4', '-przetworzony-alt.mp4')

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


def draw_parking_spots(frame, spots, state):
    for i, (x, y, w, h) in enumerate(spots):
        color = (0, 255, 0) if state.get(i) is None else (0, 0, 255)
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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


# Funkcja do wyznaczenia średniego koloru tła z pierwszej klatki
def calculate_background_color(frame):
    # Zamiana na przestrzeń HSV dla lepszej analizy koloru
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv_frame)[:3]  # Średnia wartość koloru HSV
    return mean_color  # Zwracamy (H, S, V)


# Funkcja do obliczenia średniego koloru i kontrastu w bounding boxie
def analyze_bounding_box(frame, bounding_box):
    x, y, w, h = bounding_box
    cropped = frame[y:y + h, x:x + w]
    hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Obliczanie średniego koloru w HSV
    mean_color = cv2.mean(hsv_cropped)[:3]

    # Obliczanie kontrastu jako wariancji jasności (kanał V w HSV)
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    contrast = np.var(gray_cropped)

    return mean_color, contrast


# Funkcja do sprawdzania, czy bounding box zawiera samochód
def is_car_detected(frame, bounding_box, background_color, color_tolerance=60, contrast_threshold=100):
    mean_color, contrast = analyze_bounding_box(frame, bounding_box)

    # Obliczanie różnicy między kolorem boxa a dominującym kolorem tła (na przestrzeni HSV)
    color_difference = np.linalg.norm(np.array(mean_color) - np.array(background_color))

    # Decyzja: samochód wykryty, jeśli różnica kolorów jest duża lub kontrast jest wysoki
    if color_difference > color_tolerance or contrast > contrast_threshold:
        return True
    return False


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

# Wczytywanie wideo i przygotowanie pierwszej klatki
ret, first_frame = cap.read()
if not ret:
    log_event("Nie można wczytać klatki wideo!")
    exit()

# Wyznaczanie średniego koloru tła na podstawie pierwszej klatki
background_color = calculate_background_color(first_frame)
log_event(f"Dominujący kolor tła (HSV): {background_color}")

# Rozszerzona pętla główna z zastosowaniem analizy koloru i kontrastu
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = back_sub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    parking_state = update_parking_status(contours, parking_spots, parking_state)
    draw_parking_spots(frame, parking_spots, parking_state)

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue

        # Weryfikacja detekcji poprzez analizę koloru i kontrastu
        bounding_box = cv2.boundingRect(contour)
        if not is_car_detected(frame, bounding_box, background_color):
            continue  # Ignorujemy detekcję, jeśli bounding box zawiera tło

        # Rysowanie wykrytego samochodu
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Samochod", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Zapis i wyświetlenie wyników
    out.write(frame)
    # cv2.imshow('Parking - Przetwarzanie Wideo', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
log_event("Przetwarzanie zakończone.")
