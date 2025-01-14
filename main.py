import cv2

# Ścieżka do filmu
video_path = 'data/Film_11.mp4'
cap = cv2.VideoCapture(video_path)

# Tworzy tło za pomocą BackgroundSubtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Zmiana rozmiaru klatki (opcjonalnie, by poprawić wydajność)
    frame = cv2.resize(frame, (800, 600))

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

    # Wyświetlanie obrazu w czasie rzeczywistym
    cv2.imshow('Parking - Wykrywanie Samochodow', frame)

    # Zatrzymywanie wideo klawiszem 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()