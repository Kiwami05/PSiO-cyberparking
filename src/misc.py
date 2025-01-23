"""
Moduł: misc.py
Opis: Funkcje bliżej nieokreślone/ogólne.
Utworzono: 21-01-2025
"""
import csv
import os
from datetime import datetime


def log_event(message, log_path):
    """
    Zapisuje wiadomość do pliku log wraz ze znacznikiem czasu.

    :param message: Wiadomość do pliku log.
    :param log_path: Ścieżka do pliku log.
    :return: None
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"

    print(log_entry, end='')
    with open(log_path, 'a') as log_file:
        log_file.write(log_entry)


def load_csv(file_path):
    """
    Ładuje współrzędne x, y, w, h z pliku csv.

    :param file_path: Ścieżka do pliku csv.
    :return: Lista wszystkich sczytanych krotek ze współrzędnymi.
    """
    elements = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y, w, h = map(int, row)
                elements.append((x, y, w, h))
    return elements


def intersection_over_union(rect1, rect2):
    """
    Oblicza stosunek przecięcia dwóch regionów/prostokątów.

    :param rect1: Krotka (x, y, w, h) reprezentująca pierwszy prostokąt o szerokości w i długości h, zaczynający się we współrzędnych x i y.
    :param rect2: Krotka (x, y, w, h) reprezentująca pierwszy prostokąt o szerokości w i długości h, zaczynający się we współrzędnych x i y.
    :return: Stosunek przecięcia pierwszego i drugiego prostokąta w skali [0, 1].
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height
    rect1_area = w1 * h1
    if rect1_area == 0:
        return 0
    return intersection_area / rect1_area
