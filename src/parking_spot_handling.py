"""
Moduł: parking_spot_handling.py
Opis: Zawiera funkcje do obsługi i rysowania miejsc parkingowych.
Utworzono: 21-01-2025
"""
import cv2
from src.misc import intersection_over_union, log_event
from src.license_plate_handling import read_license_plate


def update_parking_status(frame, cars, spots, spot_states, log_path, iou_threshold=0.3):
    """
    Aktualizuje status miejsca parkingowego i loguje informacje o autach.

    :param frame: Ramka z filmu w formacie BGR.
    :param cars: Współrzędne aut.
    :param spots: Współrzędne miejsc parkingowych.
    :param spot_states: Stany miejsc parkingowych (wolne/zajęte)
    :param log_path: Ścieżka do pliku log.
    :param iou_threshold: Próg, przecięcia bounding box-ów auta i miejsca, przy którym uznajemy, że auto zajmuje to miejsce.
    :return:
    """
    # Tworzymy nowy słownik do zbiorczego ustalenia stanów miejsc parkingowych
    new_spot_states = {i: False for i in range(len(spots))}

    for i, spot in enumerate(spots):
        for car in cars:
            if intersection_over_union(car, spot) > iou_threshold:
                new_spot_states[i] = True

                # Jeśli miejsce było wolne, a teraz jest zajęte, logujemy zdarzenie
                if not spot_states[i]:
                    plate = read_license_plate(frame, car)
                    log_event(f'Samochód o rejestracji `{plate}` zajmuje miejsce {i + 1}.', log_path)
                break  # Przerwij, jeśli miejsce już zostało oznaczone jako zajęte

    # Aktualizujemy stany miejsc parkingowych i logujemy zwolnienia
    for i, is_occupied in new_spot_states.items():
        for car in cars:
            if not is_occupied and spot_states[i]:  # Jeśli miejsce było zajęte, a teraz jest wolne
                plate = read_license_plate(frame, car)  # Spróbuj odczytać tablicę rejestracyjną z miejsca
                log_event(f'Samochód o rejestracji `{plate}` zwolnił miejsce {i + 1}.', log_path)

            # Zaktualizuj status miejsca
            spot_states[i] = is_occupied


def draw_parking_spots(frame, spots, state):
    """
    Rysuje miejsca parkingowe.

    :param frame: Ramka z filmu w formacie BGR.
    :param spots: Lista współrzędnych miejsc parkingowych.
    :param state: TODO to mają być wartości True/False a nie jakiegoś gówno jak teraz.
    """
    for i, (x, y, w, h) in enumerate(spots):
        color = (0, 255, 0) if not state.get(i) else (0, 0, 255)
        label = f"Miejsce {i + 1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
