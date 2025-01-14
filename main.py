import cv2
import os
import csv

# Ścieżka do filmu i pliku CSV
video_path = 'data/Film_11.mp4'
csv_path = 'data/parking_spots.csv'

# Tworzy tło za pomocą BackgroundSubtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)


# Funkcja do rysowania istniejących miejsc parkingowych
def draw_parking_spots(frame, spots):
    for i, (x, y, w, h) in enumerate(spots):
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Funkcja do wczytywania miejsc parkingowych z pliku CSV
def load_parking_spots(csv_file):
    spots = []
    if os.path.exists(csv_file):  # Sprawdzenie, czy plik istnieje
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
        if roi == (0, 0, 0, 0):  # ESC lub brak zaznaczenia — zakończenie
            break
        spots.append(roi)
    cv2.destroyWindow("Wybierz miejsca parkingowe")
    return spots


# Wczytaj wideo
cap = cv2.VideoCapture(video_path)

# Sprawdzenie, czy plik CSV istnieje i wczytanie miejsc parkingowych
if os.path.exists(csv_path):
    parking_spots = load_parking_spots(csv_path)
    print(f"Wczytano {len(parking_spots)} miejsc parkingowych z pliku {csv_path}.")
else:
    print("Plik CSV nie istnieje. Wczytuję pierwszy frame do zaznaczenia miejsc parkingowych...")
    ret, frame = cap.read()
    if not ret:
        print("Nie można wczytać klatki wideo!")
        exit()

    # Umożliwiamy ręczne zaznaczenie miejsc parkingowych
    parking_spots = select_parking_spots(frame)
    save_parking_spots(csv_path, parking_spots)
    print(f"Zapisano {len(parking_spots)} miejsc parkingowych do pliku {csv_path}.")

# Przetwarzanie wideo i rysowanie miejsc parkingowych
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Zmiana rozmiaru klatki (opcjonalnie, by poprawić wydajność)
    # frame = cv2.resize(frame, (800, 600))

    # Zastosowanie modelu odejmowania tła
    fg_mask = back_sub.apply(frame)

    # Czyszczenie maski usuwając szumy
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Znajdowanie konturów w masce
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignorowanie małych konturów
        if cv2.contourArea(contour) < 500:
            continue

        # Rysowanie prostokątów wokół wykrytych obiektów
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Samochod", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Rysuj zapisane miejsca parkingowe
    draw_parking_spots(frame, parking_spots)

    # Wyświetlanie wideo w czasie rzeczywistym
    cv2.imshow('Parking - Przetwarzanie Wideo', frame)

    # Zatrzymywanie programu przez klawisz 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
