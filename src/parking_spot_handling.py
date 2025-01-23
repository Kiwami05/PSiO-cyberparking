"""
Moduł: parking_spot_handling.py
Opis: Zawiera funkcje do obsługi i rysowania miejsc parkingowych.
Utworzono: 21-01-2025
"""
import cv2
from src.misc import intersection_over_union, log_event


def update_parking_status(contours, spots, state, log_path, iou_threshold=0.3):
    """
    TODO: do poprawy z przekazywaniem zmiennej state
    """
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
                    log_event(f"Samochód {new_state[i]} zajął Miejsce {i + 1}", log_path)
                break
        if not spot_occupied and state.get(i) is not None:
            log_event(f"Miejsce {i + 1} zostało zwolnione.",log_path)
            new_state[i] = None
    return new_state


def draw_parking_spots(frame, spots, state):
    """
    Rysuje miejsca parkingowe.

    :param frame: Ramka z filmu w formacie BGR.
    :param spots: Lista współrzędnych miejsc parkingowych.
    :param state: TODO to mają być wartości True/False a nie jakiegoś gówno jak teraz.
    :return: None
    """
    for i, (x, y, w, h) in enumerate(spots):
        color = (0, 255, 0) if state.get(i) is None else (0, 0, 255)
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
