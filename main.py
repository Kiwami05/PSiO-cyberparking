# Importy lokalne
from src.car_detection import *
from src.gate_handling import *
from src.misc import *
from src.parking_spot_handling import *

# Ścieżki do danych wejściowych
video_path = 'recordings/recording-10.mp4'
csv_parking_path = 'data/parking_spots.csv'  # Plik csv ze współrzędnymi miejsc parkingowych
csv_gates_path = 'data/gates.csv'  # Plik csv ze współrzędnymi bramek parkingowych

# Ścieżki do danych wyjściowych
log_path = os.path.join('logs', os.path.basename(video_path).replace('.mp4', '.log'))
output_video_path = os.path.join('outputs', os.path.basename(video_path).replace('.mp4', '_output.mp4'))

# Tworzy tło za pomocą BackgroundSubtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

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

# Tworzenie słowników na stany miejsc i bramek parkingowych
parking_state = {i: False for i in range(len(parking_spots))}
gate_states = {i: False for i in range(len(gates))}

# Wczytywanie wideo i przygotowanie pierwszej klatki
ret, first_frame = cap.read()
if not ret:
    log_event("Nie można wczytać klatki wideo!", log_path)
    exit()

# Główna pętla programu
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Odejmowanie tła dla wykrywania ruchu (opcjonalnie)
    fg_mask = back_sub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Wykrywanie samochodów przy użyciu nasycenia HSV
    car_detections = detect_cars_by_saturation(frame)

    # Aktualizacja statusu parkingu
    update_parking_status(frame, car_detections, parking_spots, parking_state, log_path)
    check_gate_occupation(frame, gate_states, car_detections, gates, log_path, max_distance=20)

    # Rysowanie bounding box-ów
    draw_detected_cars(frame, car_detections)
    draw_parking_spots(frame, parking_spots, parking_state)
    draw_gates(frame, gates, gate_states, draw_detection_range=True, max_distance=20)

    # Zapisz i wyświetlą przetworzoną ramkę
    out.write(frame)
    cv2.imshow('Parking - Przetwarzanie Wideo', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
log_event("Przetwarzanie zakończone.", log_path)
