"""
Moduł: license_plate_handling.py
Opis: Zawiera funkcje do obsługi tablic rejestracyjnych.
Utworzono: 22-01-2025
"""
import easyocr

def read_license_plate(frame, car):
    """
    Funkcja wykrywa i printuje wykrytą tablicę rejestracyjną.

    :param frame: Ramka z filmu w formacie BGR.
    :param car: Współrzędne reprezentujące samochód.
    :return: None
    """
    reader = easyocr.Reader(['pl'])

    x, y, w, h = car
    roi = frame[y:y + h, x:x + w]

    plate = reader.readtext(roi)
    texts = [combo[1] for combo in plate]
    print(f'Wykryto tablicę {texts}.')