"""
Rilevatore robusto di scacchiera 15x9 (5mm) con pre-processing adattivo CLAHE,
fallback multi-stadio e canonicalizzazione dell'orientamento 3D tra le viste.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class CheckerboardConfig:
    """
    Configurazione geometrica e metrica della scacchiera di calibrazione.

    Nel setup sperimentale di Seeurn:
    - Griglia: 15x9 spigoli interni (16x10 quadrati).
    - Dimensione lato quadrato: 5.0 mm.
    """

    pattern_size: Tuple[int, int] = (15, 9)
    square_size_mm: float = 5.0

    @property
    def num_corners(self) -> int:
        """Restituisce il numero totale di spigoli interni (15 * 9 = 135)."""
        return self.pattern_size[0] * self.pattern_size[1]

    def generate_object_points(self) -> np.ndarray:
        """
        Genera le coordinate 3D ideali (N, 3) nel sistema di riferimento della tavola (Z=0).
        Le coordinate sono espresse in millimetri.
        """
        nx, ny = self.pattern_size
        objp = np.zeros((nx * ny, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * self.square_size_mm
        return objp


@dataclass(frozen=True)
class DetectionResult:
    """Risultato dell'estrazione degli spigoli su un singolo fotogramma."""

    is_valid: bool
    image_points: Optional[np.ndarray]   # (135, 2) float32
    object_points: Optional[np.ndarray]  # (135, 3) float32 in mm
    method_used: str = "none"


class RobustCheckerboardDetector:
    """
    Rileva i 135 spigoli della scacchiera 15x9 gestendo:
    1. Zone localmente sature e riflessi tramite fallback CLAHE e Adaptive Thresholding.
    2. L'ambiguità di rotazione a 180° tipica delle griglie dispari x dispari (15x9),
       garantendo che il punto 0 (PT_0) corrisponda sempre allo stesso angolo fisico 3D
       (in alto a sinistra vicino al bullone per Cam 1 e Cam 2, in basso a destra
       vicino al bullone per Cam 3 che è montata capovolta).
    """

    def __init__(
        self,
        config: Optional[CheckerboardConfig] = None,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        inverted_cameras: Tuple[int, ...] = (3,),
    ):
        self.config = config or CheckerboardConfig()
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        self.inverted_cameras = inverted_cameras
        self._ideal_obj_pts = self.config.generate_object_points()

    def canonicalize_corners(self, corners: np.ndarray, camera_id: int = 1) -> np.ndarray:
        """
        Normalizza l'ordine dei 135 corner in modo che l'indice 0 sia sempre lo stesso
        punto fisico 3D (lato bullone, quadrato esterno nero).

        Args:
            corners: Array di coordinate 2D di forma (N, 2) o (N, 1, 2).
            camera_id: ID della telecamera (1, 2 sono dritte; 3 è montata capovolta di 180°).

        Returns:
            Array (N, 2) ordinato in modo canonico.
        """
        pts = corners.reshape(-1, 2).astype(np.float32)

        if camera_id in self.inverted_cameras:
            # Telecamera capovolta (bullone in basso -> y0 deve essere > y_end)
            if pts[0, 1] < pts[-1, 1]:
                pts = pts[::-1].copy()
        else:
            # Telecamera dritta (bullone in alto -> y0 deve essere < y_end)
            if pts[0, 1] > pts[-1, 1]:
                pts = pts[::-1].copy()

        return pts

    def detect(self, image: np.ndarray, camera_id: int = 1) -> DetectionResult:
        """
        Rileva gli spigoli della scacchiera usando una strategia multi-stadio:
        1. Sector-Based (SB) diretto sull'immagine in scala di grigi.
        2. Sector-Based (SB) sull'immagine equalizzata con CLAHE.
        3. Algoritmo classico con Adaptive Threshold + raffinamento sub-pixel (cornerSubPix).
        """
        if image is None or image.size == 0:
            return DetectionResult(False, None, None, "invalid_image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        p_size = self.config.pattern_size
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

        # Stadio 1: Sector-Based su immagine grezza
        found, corners = cv2.findChessboardCornersSB(gray, p_size, flags=sb_flags)
        if found and corners is not None:
            pts = self.canonicalize_corners(corners, camera_id=camera_id)
            return DetectionResult(True, pts, self._ideal_obj_pts.copy(), "sb_raw")

        # Stadio 2: Sector-Based su immagine migliorata con CLAHE
        enhanced = self.clahe.apply(gray)
        found, corners = cv2.findChessboardCornersSB(enhanced, p_size, flags=sb_flags)
        if found and corners is not None:
            pts = self.canonicalize_corners(corners, camera_id=camera_id)
            return DetectionResult(True, pts, self._ideal_obj_pts.copy(), "sb_clahe")

        # Stadio 3: Fallback classico Adaptive Threshold + SubPix
        std_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, p_size, flags=std_flags)
        if found and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            pts = self.canonicalize_corners(refined, camera_id=camera_id)
            return DetectionResult(True, pts, self._ideal_obj_pts.copy(), "adaptive_subpix")

        return DetectionResult(False, None, None, "not_found")

    def detect_from_path(self, image_path: Path, camera_id: int = 1) -> DetectionResult:
        """Carica un'immagine da disco ed esegue il rilevamento."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Impossibile leggere l'immagine: {image_path}")
        return self.detect(img, camera_id=camera_id)
