# Seeurn: Ricostruzione 3D del Tip Vortex & Dinamica di Cavitazione

Sistema modulare ad oggetti in Python per l'elaborazione ad alte prestazioni, la calibrazione multi-camera, la segmentazione deep learning e la ricostruzione 3D volumetrica della cavitazione di tip vortex in un'elica navale a 4 pale.

---

## 📌 Documentazione & Integrazione con Obsidian
La teoria matematica, l'architettura delle classi, le guide didattiche di OOP e i prompt per LLM sono integrati con il vault **Obsidian**:
- **Indice Master Obsidian:** [`Efford/Seeurn/00_Index.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/00_Index.llm.md)
- **Registro della Conversazione & Decisioni Fondative:** [`docs/CONVERSATION_BRIEFING.md`](docs/CONVERSATION_BRIEFING.md)

---

## 🏗️ Struttura del Progetto

```text
Activities/Seeurn/
├── pyproject.toml               # Configurazione pacchetto, dipendenze e pytest
├── .gitignore                   # Esclusione rigorosa di dati pesanti (data/) e cache
├── README.md                    # Questo documento
├── docs/
│   └── CONVERSATION_BRIEFING.md # Registro completo di briefing e requisiti
├── src/
│   └── seeurn/
│       ├── __init__.py
│       ├── data/                # Modulo 01: Catalog, Lazy Loading e Synchronizer
│       ├── calibration/         # Modulo 02: Rilevamento, Distorsioni, Epipolare, BBA
│       ├── segmentation/        # Modulo 03: Entropia, Active Learning, U-Net / SAM-2
│       ├── reconstruction_3d/   # Modulo 04: Visual Hull, Voxel Carving e Mesh
│       └── tracking/            # Modulo 05: Centerline (Dijkstra), Spline 3D, Raggio e 4D Track
└── tests/
    ├── conftest.py              # Fixture comuni di test e mock
    ├── test_data/
    ├── test_calibration/
    ├── test_segmentation/
    ├── test_reconstruction_3d/
    └── test_tracking/
```

---

## 🧪 Metodologia: Test-Driven Development (TDD)
Ogni componente viene sviluppato affiancato dal rispettivo test unitario con `pytest`.

### Esecuzione dei Test
Con l'ambiente virtuale attivo (`source .venv/bin/activate`):
```bash
# Esegue tutti i test
pytest -v

# Esegue solo i test del modulo dati
pytest tests/test_data/ -v
```

---

## ⚙️ Configurazione Ambiente
```bash
# Attiva il virtual environment esistente
source .venv/bin/activate

# Installa il pacchetto in modalità editabile con dipendenze di sviluppo
pip install -e ".[dev]"
```
