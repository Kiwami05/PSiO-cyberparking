"""
Moduł: license_plate_handling.py
Opis: Zawiera funkcje do obsługi tablic rejestracyjnych.
Utworzono: 22-01-2025
"""
import cv2
import easyocr

import re


def is_valid_license_plate(plate):
    """
    Sprawdza, czy tekst pasuje do formatu dopuszczonych tablic rejestracyjnych.

    :param plate: Tekst do sprawdzenia.
    :return: Wartość prawda/fałsz czy tablica pasuje do wzorca.
    """
    # Wzorzec dla tablic: 2 litery + spacja + 5 znaków alfanumerycznych
    pattern = r'^[A-Z]{2} [A-Z0-9]{5}$'

    # Sprawdzamy, czy tekst pasuje do wzorca
    return bool(re.match(pattern, plate))


def read_license_plate(frame, car, debug=True, fill=True):
    """
    Funkcja wykrywa i zwraca wykrytą tablicę rejestracyjną.

    :param frame: Ramka z filmu w formacie BGR.
    :param car: Współrzędne reprezentujące samochód.
    :param debug: Wartość prawda/fałsz czy chcemy wyświetlić informacje do debugowania.
    :param fill: Parametr do "pomagania" OCR-owi. Nie musi dawać rzeczywistych rezultatów. Czasami działa, czasami nie.
    :return: Wykryta tablica lub wiadomość o jej niewykryciu.
    """
    reader = easyocr.Reader(['pl'])

    # Szukamy tekstu tylko tam, gdzie jest autko
    x, y, w, h = car
    roi = frame[y:y + h, x:x + w]

    # Konwertujemy do skali szarości
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Domyślny tekst, jeśli nie uda się odczytać tablicy
    best_text = 'NIE ROZPOZNANO'

    if debug:
        print('Przetwarzanie rejestracji:')

    found_plates = []

    # Testujemy różne kąty rotacji
    for angle in [0, 90, 180, 270]:
        rotated = cv2.rotate(roi_gray, {
            0: cv2.ROTATE_90_CLOCKWISE,
            90: cv2.ROTATE_180,
            180: cv2.ROTATE_90_COUNTERCLOCKWISE,
            270: None
        }[angle]) if angle != 0 else roi_gray

        # Wykrywanie tekstu
        plate = reader.readtext(rotated)

        # Przetwarzanie i czyszczenie tekstu do poprawnego formatu
        text = [_[1] for _ in plate]
        license_plate = ''.join(text).strip().upper()
        license_plate = re.sub(r'[^a-zA-Z0-9]', '', license_plate)

        found_plates.append(license_plate)

    for plate in found_plates:
        # Dzielenie rejestracji na dwie części
        plate = plate[:2] + ' ' + plate[2:]

        if debug:
            print(f'- `{plate}`')

        # Jeśli funkcja zwraca prawdę, to znaczy, że najprawdopodobniej udało się nam znaleźć tablicę.
        if is_valid_license_plate(plate):
            best_text = plate
            break

    if fill and best_text == 'NIE ROZPOZNANO':
        # Trochę oszukiwanie, ale w trakcie testowania zauważyłem, że OCR z jakiegoś powodu czasami nie sczytuje pierwszej
        # części rejestracji. Nie jest to najlepsze rozwiązanie, ale czasami pomaga.
        if debug:
            print('NIE ROZPOZNANO ŻADNEJ Z REJESTRACJI, PRÓBOWANIE ZNOWU Z DOPEŁNIANIEM!')

        for plate in found_plates:
            if len(plate) == 5:
                plate = 'EL' + plate

                # Dzielenie rejestracji na dwie części
                plate = plate[:2] + ' ' + plate[2:]

                if debug:
                    print(f'- `{plate}`')

                # Jeśli funkcja zwraca prawdę, to znaczy, że najprawdopodobniej udało się nam znaleźć tablicę.
                if is_valid_license_plate(plate):
                    best_text = plate
                    break

    return best_text
