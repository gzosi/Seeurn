"""
Modulo per lo streaming e il caricamento pigro (Lazy Loading) delle immagini.
"""

from typing import Dict, Iterator, Sequence, Optional
from pathlib import Path
import numpy as np

from seeurn.data.synchronizer import SynchronizedFrameTriplet
from seeurn.data.preprocessor import ImagePreprocessor


class ConditionImageStreamer:
    """
    Iteratore che carica le immagini da disco solo al momento dell'utilizzo (Lazy Loading).

    Perché è fondamentale per Seeurn?
    Con oltre 430.000 file JPG, caricarli tutti in RAM causerebbe un crash immediato (OOM).
    Questo oggetto incapsula la lista dei puntatori su disco e legge in memoria esclusivamente
    la tripletta di frame richiesta nel ciclo corrente, rilasciandola al frame successivo.
    Tutte le immagini passano attraverso ImagePreprocessor per garantire l'orientamento corretto.
    """

    def __init__(
        self,
        triplets: Sequence[SynchronizedFrameTriplet],
        grayscale: bool = True,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        """
        Inizializza lo streamer.

        Args:
            triplets: Sequenza di triplette di frame sincronizzate.
            grayscale: Se True, carica le immagini come matrici a 1 canale (uint8 in scala di grigi).
                       Se False, carica a 3 canali BGR.
            preprocessor: Preprocessore per la normalizzazione dell'orientamento hardware.
                          Se None, usa la configurazione predefinita (Cam 3 ruotata di 180°).
        """
        self.triplets = list(triplets)
        self.grayscale = grayscale
        self.preprocessor = preprocessor or ImagePreprocessor()

    def __len__(self) -> int:
        """Restituisce il numero totale di triplette temporali disponibili."""
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[int, np.ndarray]:
        """
        Carica da disco e restituisce la tripletta di immagini per l'indice specificato,
        con l'orientamento normalizzato (Camera 3 raddrizzata).

        Args:
            idx: Indice temporale relativo [0, len - 1].

        Returns:
            Dizionario {cam_id: np.ndarray} con le immagini decodificate e orientate.

        Raises:
            IndexError: Se l'indice è fuori intervallo.
            FileNotFoundError: Se uno dei file non può essere letto da disco.
        """
        if idx < 0 or idx >= len(self.triplets):
            raise IndexError(f"Indice {idx} fuori intervallo [0, {len(self.triplets) - 1}]")

        triplet = self.triplets[idx]
        return self.preprocessor.load_triplet(triplet, grayscale=self.grayscale)

    def __iter__(self) -> Iterator[Dict[int, np.ndarray]]:
        """
        Generatore Python (yield) per scorrere sequenzialmente tutte le triplette.

        La memoria di ciascuna tripletta viene liberata automaticamente al ciclo successivo.
        """
        for i in range(len(self)):
            yield self[i]

