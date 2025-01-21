"""
Moduł: car_detection.py
Opis: Zawiera funkcje do detekcji aut.
Utworzono: 19-01-2025
"""
import cv2


def detect_cars_by_saturation(frame, min_saturation=150, area_threshold=2500, padding=None):
    """
    Wykrywa samochody na bazie wysokiego nasycenia koloru (ang. Saturation) modelu barw HSV.

    :param frame: Ramka z filmu w formacie BGR.
    :param min_saturation: Minimalna wartość nasycenia do progowania.
    :param area_threshold: Minimalny obszar konturu uznawany za samochód.
    :param padding: Opcjonalna krotka (pad_x, pad_y) dla rozszerzenia wymiarów bounding box-a.
    :return: Lista bounding box-ów reprezentujących samochody [(x, y, w, h)].
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Konwersja ramki do HSV i wyodrębnienie kanału nasycenia
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv_frame[:, :, 1]

    # Próg izolujący obszary o wysokim nasyceniu
    _, sat_mask = cv2.threshold(saturation, min_saturation, 255, cv2.THRESH_BINARY)

    # Czyszczenie maski
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    # Znajdywanie konturów
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)

            # Ustawianie padding-u (jeśli podano)
            if padding:
                pad_x, pad_y = padding
                x, y, w, h = x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y

            detections.append((x, y, w, h))
    return detections


def draw_detected_cars(frame, detections):
    """
    Rysuje bounding box-y dla samochodów wykrytych w ramce.

    :param frame: Ramka z filmu w formacie BGR.
    :param detections: Lista bounding box-ów reprezentujących samochody [(x, y, w, h)].
    :return:
    """

    for x, y, w, h in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Samochod", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
