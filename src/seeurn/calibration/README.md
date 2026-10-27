# Modulo 02: Calibrazione Multi-Camera (`seeurn.calibration`)

Questo modulo gestisce l'estrazione robusta dei keypoint della scacchiera $15 \times 9$ ($5\text{ mm}$), il campionamento temporale distanziato e la partizione in gruppi disgiunti per calibrazione monoculare, stereoscopica e Bundle Block Adjustment.

---

## 🔗 Collegamento con Obsidian
- **Pipeline Generale:** [`02_Calibration/Calibration_Pipeline.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/02_Calibration/Calibration_Pipeline.llm.md)
- **Rilevamento Target & Campionamento:** [`02_Calibration/Target_Detection.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/02_Calibration/Target_Detection.llm.md)
- **Lezione Didattica OOP:** [`02_Calibration/OOP_Concepts_Calibration.llm.md`](file:///home/gzosi/Documents/SecondBrain/Efford/Seeurn/02_Calibration/OOP_Concepts_Calibration.llm.md)

---

## 🔍 Specifiche Geometriche e Scoperte Sperimentali
1. **Geometria Scacchiera:**
   - Spigoli interni: **`15 x 9`** ($135$ corner totali, corrispondenti a $16 \times 10$ quadrati).
   - Passo metrico: **`5.0 mm`** (dimensione griglia utile: $70\text{ mm} \times 40\text{ mm}$).
2. **Canonicalizzazione dell'Orientamento ($180^\circ$):**
   - Poiché $15$ e $9$ sono entrambi dispari, la scacchiera è simmetrica per rotazione di $180^\circ$ (sia l'angolo in alto a sinistra sia quello in basso a destra hanno il quadrato esterno nero).
   - Inoltre, la **Telecamera 3** è montata capovolta di $180^\circ$ (il bullone di fissaggio nero compare in basso anziché in alto come in Cam 1 e Cam 2).
   - Il metodo `RobustCheckerboardDetector.canonicalize_corners()` forza sempre `PT_0` (indice 0) sullo stesso angolo fisico 3D (lato bullone: $y_0 < y_{134}$ per Cam 1 e 2; $y_0 > y_{134}$ per Cam 3).
3. **Campionamento Distanziato e Gruppi Disgiunti (`CalibrationFrameSampler`):**
   - Evita di selezionare frame consecutivi o troppo vicini imponendo $|\text{idx}_i - \text{idx}_j| \ge \text{min\_frame\_spacing}$ tramite la biiezione combinatoria di Kaplansky.
   - Suddivide i frame estratti in **6 gruppi totalmente disgiunti** (`mono_train`, `mono_test`, `stereo_train`, `stereo_test`, `ba_train`, `final_test`) per azzerare il *data leakage* tra addestramento e validazione.

---

## 🧪 Testing Associato (`tests/test_calibration/`)
```bash
pytest tests/test_calibration/ -v
```
