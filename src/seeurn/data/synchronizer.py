"""
Modulo per la sincronizzazione temporale e l'allineamento dei frame multi-camera.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class SynchronizedFrameTriplet:
    """
    Rappresenta una tupla di fotogrammi temporalmente coincidenti registrati dalle telecamere.

    Perché 'frozen=True'? (Principio OOP di Immutabilità)
    Un fotogramma scattato al tempo t è un dato storico sperimentale. Non deve poter
    essere alterato accidentalmente durante la pipeline (es. sovrascrivendo un percorso).
    """

    global_frame_idx: int
    cam_paths: Dict[int, Path]

    def get_path(self, cam_id: int) -> Optional[Path]:
        """Restituisce il percorso dell'immagine per la telecamera richiesta (1, 2 o 3)."""
        return self.cam_paths.get(cam_id)

    @property
    def has_all_three_views(self) -> bool:
        """Indica se il frame è stato registrato contemporaneamente da tutte e 3 le telecamere."""
        return len(self.cam_paths) == 3


class FrameSynchronizer:
    """
    Gestisce l'estrazione degli indici temporali e l'allineamento dei flussi di immagini.

    Nel setup sperimentale di Seeurn:
    - Cam 1 e Cam 2 registrano istantaneamente.
    - Cam 3 ha una latenza di accensione hardware: inizia a registrare dal frame progressivo 2.
    - Di conseguenza, il frame K di Cam 3 corrisponde al frame K di Cam 1 e Cam 2.
    """

    def __init__(
        self,
        cam3_offset: int = 2,
        regex_pattern: str = r"(\d+)\.jpe?g$",
    ):
        """
        Inizializza il sincronizzatore.

        Args:
            cam3_offset: Il numero di frame di ritardo della Telecamera 3.
            regex_pattern: Espressione regolare per estrarre l'indice numerico dal nome file.
        """
        self.cam3_offset = cam3_offset
        self.regex = re.compile(regex_pattern, re.IGNORECASE)

    def extract_index(self, path: Path) -> int:
        """
        Estrae il numero sequenziale di fotogramma dal nome del file.

        Args:
            path: Percorso del file immagine.

        Returns:
            L'indice numerico progressivo.

        Raises:
            ValueError: Se il formato del nome file non contiene un numero valido.
        """
        match = self.regex.search(path.name)
        if not match:
            raise ValueError(
                f"Impossibile estrarre l'indice temporale dal nome file: '{path.name}'. "
                f"Pattern regex atteso: '{self.regex.pattern}'"
            )
        return int(match.group(1))

    def find_missing_frames(self, files: Sequence[Path]) -> List[int]:
        """
        Identifica eventuali fotogrammi persi (buchi di campionamento) in una sequenza.

        Args:
            files: Lista di file immagine della stessa telecamera.

        Returns:
            Lista di indici di frame mancanti.
        """
        if not files:
            return []

        indices = sorted(self.extract_index(p) for p in files)
        full_range = set(range(indices[0], indices[-1] + 1))
        missing = sorted(list(full_range - set(indices)))
        return missing

    def align(
        self,
        cam1_files: Sequence[Path],
        cam2_files: Sequence[Path],
        cam3_files: Sequence[Path],
        require_all_views: bool = True,
    ) -> List[SynchronizedFrameTriplet]:
        """
        Allinea i file delle tre telecamere in triplette temporalmente coerenti.

        Args:
            cam1_files: File registrati dalla Telecamera 1.
            cam2_files: File registrati dalla Telecamera 2.
            cam3_files: File registrati dalla Telecamera 3.
            require_all_views: Se True, include solo le triplette complete (tutte e 3 le viste).
                               Se False, include anche coppie parziali.

        Returns:
            Lista ordinata di SynchronizedFrameTriplet.
        """
        # Mappiamo ogni file con il proprio indice numerico
        map_cam1 = {self.extract_index(p): p for p in cam1_files}
        map_cam2 = {self.extract_index(p): p for p in cam2_files}
        map_cam3 = {self.extract_index(p): p for p in cam3_files}

        aligned_triplets: List[SynchronizedFrameTriplet] = []

        # Determiniamo l'intervallo temporale utile (il minimo comune indice)
        # Cam 3 parte da cam3_offset (es. 2), quindi l'indice coincidente parte da lì
        all_candidate_indices = sorted(set(map_cam3.keys()) | set(map_cam1.keys()) | set(map_cam2.keys()))

        for idx in all_candidate_indices:
            paths: Dict[int, Path] = {}

            if idx in map_cam1:
                paths[1] = map_cam1[idx]
            if idx in map_cam2:
                paths[2] = map_cam2[idx]
            if idx in map_cam3:
                paths[3] = map_cam3[idx]

            if require_all_views:
                if len(paths) == 3:
                    aligned_triplets.append(
                        SynchronizedFrameTriplet(global_frame_idx=idx, cam_paths=paths)
                    )
            else:
                if len(paths) >= 2:
                    aligned_triplets.append(
                        SynchronizedFrameTriplet(global_frame_idx=idx, cam_paths=paths)
                    )

        return aligned_triplets

