# Wymagania dla projektu: Analiza Parkingu

***Wykonali Jakub Kołacha 240708 i Bartosz Tyczyński 245952***

## Cel projektu

Celem projektu jest stworzenie systemu do analizy obrazów parkingu wykonanych z góry, który umożliwi:

- Identyfikację miejsc parkingowych

- Określenie liczby zajętych i wolnych miejsc parkingowych

- Wykrywanie błędnie zaparkowanych pojazdów

## Założenia

1. System nie będzie wykorzystywał modeli AI (np. sieci neuronowych, uczenia maszynowego).

2. Przetwarzanie obrazów będzie oparte wyłącznie na klasycznych metodach przetwarzania obrazów i algorytmach analizy
   obrazu.

3. Obrazy parkingów będą pochodziły z kamer zamontowanych na wysokości, zapewniających widok na cały obszar parkingu.

## Wymagania funkcjonalne

### Identyfikacja miejsc parkingowych:

- System powinien umożliwiać ręczne lub półautomatyczne oznaczanie miejsc parkingowych na obrazie podczas konfiguracji.

- Oznaczone miejsca powinny być zapisywane w formie współrzędnych (np. prostokątów obejmujących poszczególne miejsca
  parkingowe).

### Detekcja zajętości miejsc parkingowych:

- System powinien analizować zajętość miejsc parkingowych na podstawie kontrastu, krawędzi i kolorów.

- Dla każdego miejsca parkingowego powinna zostać określona jego zajętość jako "wolne" lub "zajęte".

### Licznik miejsc parkingowych:

- System powinien zliczać liczbę miejsc:

- Całkowitych

- Zajętych

- Wolnych

### Wykrywanie błędów w parkowaniu:

- System powinien wykrywać samochody, które:

- Wystają poza wyznaczone miejsce parkingowe,

- Zajmują więcej niż jedno miejsce parkingowe.

## Wymagania niefunkcjonalne

### Przenośność:

- Aplikacja powinna być uruchamiana na standardowym komputerze PC bez potrzeby dedykowanego sprzętu.

### Dokładność:

- Dokładność detekcji miejsc zajętych i wolnych powinna wynosić co najmniej 90% w standardowych warunkach oświetlenia.

### Łatwość utrzymania:

- Kod systemu powinien być napisany w sposób czytelny i zgodny z dobrymi praktykami programistycznymi, umożliwiający
  łatwe modyfikacje.

## Technologie

### Przetwarzanie obrazów:

- Opencv lub inne biblioteki wspierające klasyczne metody przetwarzania obrazów.

### Format danych:

- Obsługa obrazów w formatach takich jak JPEG, PNG.

- Wyniki analizy zapisywane w formacie CSV, JSON lub innym użytecznym dla użytkownika.

### Interfejs użytkownika:

- Prosty interfejs graficzny lub narzędzia wiersza poleceń do wprowadzania konfiguracji i przeglądu wyników analizy. 