# Modulo 01: Gestione Dati, Sincronizzazione e Normalizzazione Immagini (`seeurn.data`)

Questo sottomodulo gestisce l'accesso, l'allineamento temporale e la normalizzazione dell'orientamento geometrico per **tutte le immagini del progetto Seeurn** (sia i dati di calibrazione che le 72 condizioni di cavitazione).

---

## 🔗 Collegamento con Obsidian (Second Brain)
- **Specifica di Dettaglio:** [`01_Data_Management/Data_Management.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/01_Data_Management/Data_Management.llm.md)
- **Logica di Sincronizzazione:** [`01_Data_Management/Frame_Synchronizer.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/01_Data_Management/Frame_Synchronizer.llm.md)
- **Lezione Didattica OOP:** [`01_Data_Management/OOP_Concepts_Data.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/01_Data_Management/OOP_Concepts_Data.llm.md)

---

## 🏛️ Architettura delle Classi

| Classe | Scopo & Responsabilità |
| :--- | :--- |
| [`CameraHardwareConfig`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/hardware.py) | Incapsula le proprietà del montaggio fisico (rotazione di 180° per la Camera 3, 0° per Cam 1 e 2). Caricabile da [`configs/hardware_config.yaml`](file:///home/gzosi/Documents/Activities/Seeurn/configs/hardware_config.yaml). |
| [`ImagePreprocessor`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/preprocessor.py) | Decodifica le immagini da disco applicando la rotazione automatica. Restituisce immagini già raddrizzate e orientate coerentemente nello spazio mondo. |
| [`FrameSynchronizer`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/synchronizer.py) | Corregge la latenza temporale hardware di Cam 3 (offset = +2 frame) e allinea i flussi temporali. |
| [`SynchronizedFrameTriplet`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/synchronizer.py) | Dataclass immutabile (`frozen=True`) che contiene i percorsi su disco delle 3 viste sincronizzate per un dato istante $t$. |
| [`DatasetCatalog`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/catalog.py) | Scandaglia e indicizza le 72 condizioni di cavitazione in `data/raw/runs/` senza caricare dati pesanti in memoria. |
| [`CalibrationCatalog`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/catalog.py) | Scandaglia e indicizza i dataset di calibrazione in `data/raw/calibration/{wake}/{cam}/{dataset}`. |
| [`ConditionImageStreamer`](file:///home/gzosi/Documents/Activities/Seeurn/src/seeurn/data/streamer.py) | Iteratore con Lazy Loading che carica in memoria solo la tripletta richiesta dal ciclo corrente e libera la RAM a ogni iterazione, integrando `ImagePreprocessor`. |

---

## 💡 Esempio di Utilizzo

```python
from seeurn.data import CalibrationCatalog, ImagePreprocessor

# 1. Inizializza il catalogo di calibrazione
catalog = CalibrationCatalog("data/raw/calibration")

# 2. Ottieni lo streamer per un dataset specifico (immagini caricate on-demand e già raddrizzate)
streamer = catalog.get_image_streamer("wake2", "dataset3")

# 3. Itera sulle triplette senza rischiare Out-Of-Memory (OOM)
for triplet in streamer:
    img_cam1 = triplet[1]  # Alta risoluzione (1400, 2176), dritta
    img_cam2 = triplet[2]  # Alta risoluzione (1400, 2176), dritta
    img_cam3 = triplet[3]  # Sensore laterale (800, 1280), ruotata automaticamente di 180°!
    break
```

---

## 🧪 Testing Associato (`tests/test_data/`)

La suite di test per il Modulo 01 copre:
1. `test_preprocessor.py`: Verifica della corretta rotazione di 180° per Cam 3, invarianza per Cam 1/2, e verifica su fotogramma reale `data.7vt5sy3k.000100.jpg`.
2. `test_synchronizer.py`: Parsing degli indici dei frame, offset=2 e rilevamento frame mancanti.
3. `test_synchronizer_real_data.py`: Verifica dell'allineamento sui file reali di `wake2`.
4. `test_catalog.py`: Riconoscimento automatico delle directory reali di run.
5. `test_calibration_catalog.py`: Riconoscimento dei dataset di calibrazione reali e verifica dimensioni dei sensori ($1400 \times 2176$ per Cam 1/2, $800 \times 1280$ per Cam 3).

### Come eseguire i test:
```bash
# Esegui tutti i test del modulo dati con output dettagliato
pytest tests/test_data/ -v

# Esegui con report di coverage
pytest tests/test_data/ -v --cov=seeurn.data
```
