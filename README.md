# Akordio Core
Tento repozitář obsahuje sdílené zdroje pro trénování modelů pro rozpoznávání akordů a jejich využití v jednoduché webové aplikaci.

## Obsah
- __Classes__ obsahuje definiční třídy konfiguračního souboru a PyTorch datesetu.
- __Models__ obsahuje vytrénované modely pro použití ve webové aplikaci. Modely jsou tříděny pro online a offline využití.
- __Tools__ obsahuje základní nástroje pro zpracování zvukových souborů a práci s akordy.

## Aktuální modely
Tato sekce obsahuje základní informace o aktuálně uložených modelech používaných ve webové aplikaci.
Detailnější informace je možné nalézt v konfiguračních souborech modelů.

### Online rozpoznávání
- __Typ:__ Konvoluční Rekurentní Neuronová síť (CR2)
- __Data:__ CQT Dataset
- __Typ trénování:__ Víceúlohové

### Offline rozpoznávání
- __Typ__: Konvoluční Rekurentní Neuronová síť (CR2)
- __Data__: CQT Dataset
- __Typ trénování:__ Víceúlohové