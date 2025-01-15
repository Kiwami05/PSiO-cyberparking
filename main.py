import cv2
import os
import csv

# Ścieżka do filmu i pliku CSV
video_path = 'data/Film_3.mp4'
csv_path = 'data/parking_spots.csv'

# Tworzy tło za pomocą BackgroundSubtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# Dodajemy słownik do śledzenia stanu miejsc parkingowych
parking_state = {}

# Funkcja do rysowania istniejących miejsc parkingowych
def draw_parking_spots(frame, spots):
    for i, (x, y, w, h) in enumerate(spots):
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Funkcja do wczytywania miejsc parkingowych z pliku CSV
def load_parking_spots(csv_file):
    spots = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y, w, h = map(int, row)
                spots.append((x, y, w, h))
    return spots

# Funkcja do zapisywania miejsc parkingowych do pliku CSV
def save_parking_spots(csv_file, spots):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(spots)

# Funkcja do interaktywnego zaznaczania miejsc parkingowych
def select_parking_spots(frame):
    spots = []
    print("Zaznacz miejsca parkingowe. Wciśnij ENTER po wyborze miejsca, a ESC aby zakończyć.")
    while True:
        roi = cv2.selectROI("Wybierz miejsca parkingowe", frame, fromCenter=False, showCrosshair=True)
        if roi == (0, 0, 0, 0):
            break
        spots.append(roi)
    cv2.destroyWindow("Wybierz miejsca parkingowe")
    return spots

# Funkcja do sprawdzania procentowego pokrycia prostokątów
def intersection_over_union(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Współrzędne prostokąta przecięcia
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Obliczenie powierzchni przecięcia
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height

    # Powierzchnie prostokątów
    rect1_area = w1 * h1

    # Uniknięcie dzielenia przez zero
    if rect1_area == 0:
        return 0

    return intersection_area / rect1_area

# Funkcja do wykrywania zajętości miejsc parkingowych
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
                    print(f"Samochód {new_state[i]} zajął Miejsce {i + 1}")
                break
        if not spot_occupied and state.get(i) is not None:
            print(f"Miejsce {i + 1} zostało zwolnione.")
            new_state[i] = None
    return new_state

# Wczytaj wideo
cap = cv2.VideoCapture(video_path)

# Sprawdzenie, czy plik CSV istnieje i wczytanie miejsc parkingowych
if os.path.exists(csv_path):
    parking_spots = load_parking_spots(csv_path)
    print(f"Wczytano {len(parking_spots)} miejsc parkingowych z pliku {csv_path}.")
    parking_state = {i: None for i in range(len(parking_spots))}
else:
    print("Plik CSV nie istnieje. Wczytuję pierwszy frame do zaznaczenia miejsc parkingowych...")
    ret, frame = cap.read()
    if not ret:
        print("Nie można wczytać klatki wideo!")
        exit()

    parking_spots = select_parking_spots(frame)
    save_parking_spots(csv_path, parking_spots)
    print(f"Zapisano {len(parking_spots)} miejsc parkingowych do pliku {csv_path}.")
    parking_state = {i: None for i in range(len(parking_spots))}

# Przetwarzanie wideo
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

    cv2.imshow('Parking - Przetwarzanie Wideo', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
