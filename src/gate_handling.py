"""
Moduł: gate_handling.py
Opis: Zawiera funkcje do obsługi i rysowania bramek parkingowych.
Utworzono: 21-01-2025
"""
import cv2
from src.license_plate_handling import *
from src.misc import intersection_over_union


def check_gate_occupation(frame, gate_states, contours, gates, max_distance=50, iou_threshold=0.01):
    """
    Funkcja decyduje, czy bramka powinna być zamknięta, czy otwarta.

    :param frame: Ramka z filmu w formacie BGR.
    :param gate_states: Słownik reprezentująca czy dana bramka jest otwarta, czy nie.
    :param contours: Lista reprezentująca współrzędne wykrytych aut.
    :param gates: Lista reprezentująca współrzędne bramki wjazdowej i wyjazdowej.
    :param max_distance: Maksymalna odległość samochodu od bramki, na jakiej samochód zostanie wykryty.
    :param iou_threshold: Próg, przy którym uznajemy, że samochód jest w strefie bycia wykrytym przez bramkę.
    :return: Słownik z wartościami prawda/fałsz, czy dana bramka jest otwarta, czy nie.
    """

    for contour in contours:
        for i, (gate_x, gate_y, gate_w, gate_h) in enumerate(gates):
            # Ustalamy współrzędne strefy wykrywania auta
            detection_area = [gate_x, gate_y, gate_w + max_distance, gate_h]

            # Przesuwamy współrzędne prostokąta dla bramki wjazdowej, by zaczynał się na lewo od bramki
            if i == 0:
                detection_area[0] -= max_distance

            # Sprawdzamy, czy auto nie znajduje się w strefie wykrywania bramki
            if intersection_over_union(contour, detection_area) > iou_threshold:

                # Jeśli status bramki się zmieni, to próbujemy wykryć tablicę
                if not gate_states[i]:
                    read_license_plate(frame, contour)

                gate_states[i] = True
            else:
                gate_states[i] = False


def draw_gates(frame, gates, state, draw_detection_range=False, max_distance=50):
    """
    Funkcja rysująca bramkę wjazdową i wyjazdową i opcjonalnie strefy wykrywania.

    :param frame: Ramka z filmu w formacie BGR.
    :param gates: Lista reprezentująca współrzędne bramki wjazdowej i wyjazdowej.
    :param state: Słownik z wartościami prawda/fałsz, czy dana bramka jest otwarta, czy nie.
    :param draw_detection_range: Wartość prawda/fałsz, czy chcemy rysować strefy wykrywania.
    :param max_distance: Maksymalna odległość samochodu od bramki, na jakiej samochód zostanie wykryty.
    :return:
    """

    # Rysujemy bramki
    for i, (x, y, w, h) in enumerate(gates):
        color = (0, 0, 255) if not state.get(i, False) else (0, 255, 0)
        label = f"Bramka {'wjazdowa' if i == 0 else 'wyjazdowa'} - {'otwarta' if state.get(i, False) else 'zamknieta'}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Rysujemy strefy detekcji
    if draw_detection_range:
        for i, (x, y, w, h) in enumerate(gates):
            color = (255, 0, 255)
            label = 'Strefa wykrywania auta'
            if i == 0:
                x -= max_distance
            cv2.rectangle(frame, (x, y), (x + w + max_distance, y + h), color, 2)
            cv2.putText(frame, label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
