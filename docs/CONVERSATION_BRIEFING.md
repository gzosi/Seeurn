# Seeurn: Briefing di Progetto e Registro della Conversazione Fondativa

Questo documento registra integralmente le decisioni, i requisiti fisici e sperimentali, i dettagli architetturali e la roadmap tecnica stabiliti durante la conversazione fondativa del progetto **Seeurn**.

---

## 1. Contesto Sperimentale e Dati Reali

- **Obiettivo Fisico:** Ricostruzione 3D volumetrica e caratterizzazione temporale del cavo di vortice di estremità (*tip vortex cavitation*) di un'elica navale in scala modello operante in un tunnel di cavitazione.
- **Geometria dell'Elica:** Elica a **4 pale** $\implies$ in ciascun fotogramma possono comparire contemporaneamente **fino a 4 vortici distinti**.
- **Condizioni Sperimentali:** 72 condizioni operative complessive:
  - **4 Scie** (condizioni di flusso a monte)
  - **3 Coefficienti di Spinta ($K_T$)**
  - **6 Numeri di Cavitazione ($\sigma$)**
- **Sistema di Acquisizione Ottica:**
  - 3 telecamere ad alta velocità (Cam 1 e Cam 2 ortogonali/frontali, Cam 3 laterale fortemente angolata e soggetta a rifrazione attraverso le finestre del tunnel).
  - Circa 2000 immagini in formato `.jpg` per telecamera per ciascuna condizione $\implies \approx 432.000$ immagini totali.
  - **Shift Temporale Hardware:** La Telecamera 3 presenta un ritardo fisso di avvio pari a 2 frame. Le registrazioni di Cam 3 partono dal frame `...000002.jpg`, corrispondente al frame `...000002.jpg` di Cam 1 e Cam 2 (che hanno i frame iniziali `...000000.jpg` e `...000001.jpg` orfani).
- **Setup di Calibrazione:**
  - 4 dataset di calibrazione dedicati registrati orientando la scacchiera per ottimizzare specifiche coppie di telecamere (dataset 1-2 per Cam 1-2; dataset 3-4 per includere Cam 3).
  - Illuminazione non uniforme e saturazione locale dovuta a riflessi su plexiglass/vetro del tunnel.
- **Hardware di Calcolo:**
  - GPU NVIDIA GeForce RTX 4070 (8 GB VRAM).
  - Ambiente Python 3.13.5 configurato in `.venv`.

---

## 2. Requisiti e Decisioni Metodologiche

### A. Gestione Dati & Streaming
- Impossibilità di caricare tutti i file in memoria RAM (rischio immediato di OOM).
- Soluzione: Pattern **Iterator** e **Lazy Loading** con dataclass immutabili (`ExperimentCondition`) per caricare on-demand solo le triplette di frame necessarie.

### B. Calibrazione Multi-Camera Robusta
- Target a scacchiera con pre-processing adattivo (CLAHE) e algoritmo **Sector-Based (`findChessboardCornersSB`)** per superare zone sature.
- Modello di distorsione avanzato a **14 parametri di OpenCV**:
  - Radiale razionale ($k_1..k_6$)
  - Tangenziale ($p_1, p_2$)
  - Thin-prism ($s_1..s_4$)
  - Tilted sensor ($\tau_x, \tau_y$)
  *(utili per assorbire matematicamente le non-linearità introdotte dalla rifrazione piana aria-vetro-acqua).*
- Costruzione di un **Grafo Epipolare** con verifica a ciclo chiuso ($R_{13} \approx R_{23} R_{12}$) e **Global Bundle Adjustment (BBA)** con perdita robusta di Huber su tutti e 4 i dataset.
- Protocollo Train/Test (80/20) per certificare che l'errore medio di riproiezione sia sub-pixel ($< 0.5 \text{ px}$).

### C. Segmentazione & Campionamento Entropico
- Basso rapporto segnale/rumore (SNR) e aspetto fortemente variabile (da filamento sottile a tubo spesso di vapore).
- **Active Learning con Entropia Locale:** Calcolo dell'entropia di Shannon sui frame per identificare le zone di transizione turbolenta liquido-vapore ed estrarre ~200 fotogrammi chiave da etichettare con poligoni in CVAT.
- Modello di deep learning (Attention U-Net o SAM-2 LoRA) con loss composita (Focal + Dice + Boundary) addestrato in precisione mista FP16 per rimanere entro gli 8GB di VRAM.

### D. Ricostruzione 3D (Shape-from-Silhouette / Visual Hull)
- Associazione geometrica dei 4 vortici tra le viste con vincolo epipolare e Algoritmo Ungherese.
- Discretizzazione metrica dello spazio in una griglia di voxel solidi in millimetri (**Voxel Carving**).
- Estrazione della mesh poligonale chiusa con Marching Cubes e calcolo del volume 3D reale $V \ [\text{mm}^3]$.

### E. Centerline, Raggio Locale Normale e Dinamica
- **Metodo dei grafi per la spina dorsale:** Nube di voxel trasformata in grafo di adiacenza 3D; estrazione del cammino massimo (diametro del grafo) con doppio algoritmo di Dijkstra.
- **Fitting continuo con B-Spline 3D cubica:** Calcolo dei vettori tangenti $\mathbf{T}(s)$ per ascissa curvilinea $s \in [0, L]$.
- **Affettamento Normale (Normal Slicing):** Divisione del volume in fette cilindriche ortogonali a $\mathbf{T}(s)$ di spessore $\Delta s$. Calcolo del raggio equivalente:
  $$r(s) = \sqrt{\frac{\Delta V(s)}{\pi \Delta s}}$$
- **Tracking 4D:** Tracciamento della traiettoria delle 4 pale lungo la sequenza temporale per estrarre $V(t)$, le derivate $\frac{dV}{dt}$ e $\frac{d^2V}{dt^2}$ (acustica sottomarina) e il passo elicoidale.

---

## 3. Nuovi Pilastri Operativi Concordati

1. **Programmazione ad Oggetti (OOP):**
   - Transizione dalla programmazione procedurale a classi ben strutturate (incapsulamento, `@dataclass`, Strategy pattern, Factory, macchine a stati).
   - Documentazione teorica integrata nell'Obsidian vault in `Efford/Seeurn/`.
2. **Programmazione per Testing (TDD):**
   - Ogni modulo, classe e funzione deve essere accompagnata da test unitari scritti con `pytest`.
   - Utilizzo di test sintetici e fixture per verificare la logica prima di eseguire sui dati reali.
3. **Documentazione Integrata:**
   - Struttura di `README.md` modulari nel codice sorgente (`src/seeurn/...`) collegati bidirezionalmente alle note in Obsidian.
4. **Controllo Versione Git:**
   - Repository collegato a GitHub (`https://github.com/gzosi/Seeurn.git`).
   - `.gitignore` rigoroso che esclude i dataset pesanti (`data/`), pesi, cache e ambienti virtuali.

---

## 4. Riferimenti Incrociati
- **Vault Obsidian:** `SecondBrain/Efford/Seeurn/00_Index.llm.md`
- **Specifiche Tecniche:** Moduli 01 (Data), 02 (Calibration), 03 (Segmentation), 04 (3D Recon), 05 (Tracking).
