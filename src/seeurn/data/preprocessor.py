"""
Pipeline di pre-elaborazione e normalizzazione dell'orientamento delle immagini (Modulo 01).
Garantisce che tutte le immagini provenienti dal tunnel sperimentale (sia calibrazione che run)
siano coerentemente orientate ('in piedi') nello spazio di laboratorio.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import cv2
import numpy as np

from seeurn.data.hardware import CameraHardwareConfig
from seeurn.data.synchronizer import SynchronizedFrameTriplet


class ImagePreprocessor:
    """
    Gestisce la decodifica e la normalizzazione geometrica (rotazione hardware) delle immagini.

    Perché è un'invariante di sistema in Seeurn?
    La Camera 3 è montata fisicamente ruotata di 180° per ragioni di alloggiamento sul tunnel.
    Questo preprocessore applica la correzione geometrica direttamente alla decodifica del file,
    così che nessun modulo a valle (calibrazione, segmentazione dei vortici, triangolazione 3D)
    debba mai preoccuparsi di coordinate invertite o capovolte.
    """

    def __init__(self, hardware_config: Optional[CameraHardwareConfig] = None):
        """
        Inizializza il preprocessore.

        Args:
            hardware_config: Istanza di CameraHardwareConfig (default: rotazione 180° per Cam 3).
        """
        self.hardware_config = hardware_config or CameraHardwareConfig()

    def preprocess_image(self, img: np.ndarray, camera_id: int) -> np.ndarray:
        """
        Applica la correzione di orientamento richiesta dall'hardware della specifica telecamera.

        Args:
            img: Matrice NumPy dell'immagine (1 canale o 3 canali).
            camera_id: Identificativo della telecamera (es. 1, 2, 3).

        Returns:
            Matrice dell'immagine orientata correttamente (eventualmente ruotata).
        """
        flag = self.hardware_config.get_rotation_flag(camera_id)
        if flag is not None:
            return cv2.rotate(img, flag)
        return img

    def load_image(
        self,
        file_path: Union[str, Path],
        camera_id: int,
        grayscale: bool = True,
    ) -> np.ndarray:
        """
        Legge un'immagine da disco e ne normalizza l'orientamento hardware.

        Args:
            file_path: Percorso del file immagine.
            camera_id: ID della telecamera (necessario per applicare la rotazione corretta).
            grayscale: Se True carica a 1 canale (uint8), se False carica a 3 canali BGR.

        Returns:
            Matrice NumPy dell'immagine decodificata e orientata.

        Raises:
            FileNotFoundError: Se il file non esiste o OpenCV non riesce a decodificarlo.
        """
        path_str = str(file_path)
        cv_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path_str, cv_flag)

        if img is None:
            raise FileNotFoundError(
                f"Impossibile decodificare l'immagine su disco: {path_str}"
            )

        return self.preprocess_image(img, camera_id=camera_id)

    def load_triplet(
        self,
        triplet: SynchronizedFrameTriplet,
        grayscale: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Carica e orienta simultaneamente tutte le viste disponibili in una tripletta sincronizzata.

        Args:
            triplet: Istanza di SynchronizedFrameTriplet.
            grayscale: Se True carica a 1 canale (uint8), se False a 3 canali BGR.

        Returns:
            Dizionario {cam_id: np.ndarray} con tutte le immagini raddrizzate.
        """
        return {
            cam_id: self.load_image(path, camera_id=cam_id, grayscale=grayscale)
            for cam_id, path in triplet.cam_paths.items()
        }
