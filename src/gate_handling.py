"""
Moduł: gate_handling.py
Opis: Zawiera funkcje do obsługi i rysowania bramek parkingowych.
Utworzono: 21-01-2025
"""
from src.license_plate_handling import *
from src.misc import intersection_over_union, log_event


def check_gate_occupation(frame, gate_states, cars, gates, log_path, max_distance=50, iou_threshold=0.01):
    """
    Decyduje, czy bramka powinna być zamknięta, czy otwarta.

    :param frame: Ramka z filmu w formacie BGR.
    :param gate_states: Słownik reprezentujący czy dana bramka jest otwarta, czy nie.
    :param cars: Lista reprezentująca współrzędne wykrytych aut.
    :param gates: Lista reprezentująca współrzędne bramki wjazdowej i wyjazdowej.
    :param log_path: Ścieżka do pliku log.
    :param max_distance: Maksymalna odległość samochodu od bramki, na jakiej samochód zostanie wykryty.
    :param iou_threshold: Próg, przy którym uznajemy, że samochód jest w strefie bycia wykrytym przez bramkę.
    """
    # Przechowujemy nowy stan bramek na podstawie wszystkich samochodów
    new_gate_states = {i: False for i in range(len(gates))}

    for car in cars:
        for i, (gate_x, gate_y, gate_w, gate_h) in enumerate(gates):
            # Ustalamy współrzędne strefy wykrywania auta
            detection_area = [gate_x, gate_y, gate_w + max_distance, gate_h]

            # Przesuwamy współrzędne prostokąta dla bramki wjazdowej, by zaczynał się na lewo od bramki
            if i == 0:
                detection_area[0] -= max_distance

            # Sprawdzamy, czy auto znajduje się w strefie wykrywania bramki
            if intersection_over_union(car, detection_area) > iou_threshold:
                new_gate_states[i] = True

                # Wykrywamy tablicę samochodu, jeśli bramka została otwarta
                if not gate_states[i]:
                    plate = read_license_plate(frame, car)
                    log_event(
                        f'Samochód o rejestracji `{plate}` {"wjeżdża na parking" if i == 0 else "wyjeżdża z parkingu"}.',
                        log_path)

    # Aktualizujemy stan bramek tylko na podstawie zbiorczego wyniku
    for i in range(len(gates)):
        gate_states[i] = new_gate_states[i]


def draw_gates(frame, gates, state, draw_detection_range=False, max_distance=50):
    """
    Rysuje bramkę wjazdową i wyjazdową i opcjonalnie strefy wykrywania.

    :param frame: Ramka z filmu w formacie BGR.
    :param gates: Lista reprezentująca współrzędne bramki wjazdowej i wyjazdowej.
    :param state: Słownik z wartościami prawda/fałsz, czy dana bramka jest otwarta, czy nie.
    :param draw_detection_range: Wartość prawda/fałsz, czy chcemy rysować strefy wykrywania.
    :param max_distance: Maksymalna odległość samochodu od bramki, na jakiej samochód zostanie wykryty.
    """
    # Rysujemy bramki
    for i, (x, y, w, h) in enumerate(gates):
        color = (0, 0, 255) if not state.get(i, False) else (0, 255, 0)
        label = f"Bramka {'wjazdowa' if i == 0 else 'wyjazdowa'}"
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
