"""
Calibrazione monoculare con modelli di distorsione complessi (Modulo 02, Fase 2).
Supporta i modelli estesi di OpenCV fino a 14 parametri (radiale razionale, tangenziale,
thin-prism e tilted sensor) per modellare con accuratezza sub-pixel le deformazioni
introdotte dalla rifrazione piana aria-vetro-acqua nelle finestre del tunnel idrodinamico.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np


@dataclass(frozen=True)
class DistortionModelFlags:
    """
    Flag di controllo per i modelli di distorsione di OpenCV.

    Permette di abilitare selettivamente:
    - Modello Razionale (k4, k5, k6): frazione razionale radiale adatta a distorsioni forti.
    - Modello Thin-Prism (s1, s2, s3, s4): modella l'effetto cuneo ottico della lastra di vetro.
    - Modello Tilted Sensor (tau_x, tau_y): modella l'eventuale inclinazione del piano del sensore.
    - Vincoli addizionali (aspect ratio fisso, distorsione tangenziale nulla).
    """

    use_rational_model: bool = True
    use_thin_prism: bool = True
    use_tilted: bool = True
    fix_aspect_ratio: bool = False
    zero_tangent_dist: bool = False

    def to_opencv_flags(self) -> int:
        """Converte i flag booleani nella maschera di bit intera richiesta da OpenCV."""
        flags = 0
        if self.use_rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if self.use_thin_prism:
            flags |= cv2.CALIB_THIN_PRISM_MODEL
        if self.use_tilted:
            flags |= cv2.CALIB_TILTED_MODEL
        if self.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if self.zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        return flags


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Parametri intrinseci e coefficienti di distorsione stimati per una singola telecamera.
    """

    camera_id: int
    K: np.ndarray                         # Matrice intrinseca 3x3 (fx, fy, cx, cy)
    dist_coeffs: np.ndarray               # Vettore di distorsione (1, N) con N in {5, 8, 12, 14}
    image_size: Tuple[int, int]           # (width, height) in pixel
    mean_reprojection_error: float        # Errore RMS medio sui fotogrammi di calibrazione (px)

    def project_points(
        self,
        object_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        """
        Proietta punti 3D nello spazio telecamera sulle coordinate pixel 2D del sensore.

        Args:
            object_points: Coordinate 3D (N, 3) espresse in millimetri.
            rvec: Vettore di rotazione Rodrigues (3, 1) o (3,).
            tvec: Vettore di traslazione (3, 1) o (3,) in millimetri.

        Returns:
            Coordinate pixel (N, 2) float32 o float64.
        """
        pts_3d = object_points.reshape(-1, 3).astype(np.float64)
        r = rvec.reshape(3, 1).astype(np.float64)
        t = tvec.reshape(3, 1).astype(np.float64)
        proj_pts, _ = cv2.projectPoints(pts_3d, r, t, self.K, self.dist_coeffs)
        return proj_pts.reshape(-1, 2)

    def undistort(
        self,
        image: np.ndarray,
        new_camera_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Rimuove la distorsione ottica da un'immagine usando i parametri stimati.

        Args:
            image: Immagine uint8 originale (1 o 3 canali).
            new_camera_matrix: Matrice intrinseca opzionale per l'immagine rettificata.

        Returns:
            Immagine rettificata senza distorsione ottica.
        """
        new_k = new_camera_matrix if new_camera_matrix is not None else self.K
        return cv2.undistort(image, self.K, self.dist_coeffs, None, new_k)

    def to_dict(self) -> Dict[str, Any]:
        """Serializza i parametri in un dizionario compatibile con JSON."""
        return {
            "camera_id": self.camera_id,
            "image_size": list(self.image_size),
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "K": self.K.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraIntrinsics":
        """Deserializza un'istanza da dizionario."""
        img_size = data["image_size"]
        return cls(
            camera_id=int(data["camera_id"]),
            K=np.array(data["K"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(1, -1),
            image_size=(int(img_size[0]), int(img_size[1])),
            mean_reprojection_error=float(data["mean_reprojection_error"]),
        )

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Salva i parametri intrinseci su file JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> "CameraIntrinsics":
        """Carica i parametri intrinseci da file JSON."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MonocularCalibrator:
    """
    Esegue la stima dei parametri intrinseci (K, dist_coeffs) e la validazione out-of-sample.
    """

    def __init__(
        self,
        flags: Optional[DistortionModelFlags] = None,
        criteria: Optional[Tuple[int, int, float]] = None,
    ):
        self.flags = flags or DistortionModelFlags()
        self.criteria = criteria or (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-7,
        )

    def calibrate(
        self,
        object_points_list: Sequence[np.ndarray],
        image_points_list: Sequence[np.ndarray],
        image_size: Tuple[int, int],
        camera_id: int = 1,
    ) -> CameraIntrinsics:
        """
        Calcola la matrice K e i coefficienti di distorsione sui dati di training.

        Args:
            object_points_list: Lista di array (N, 3) di punti 3D nello spazio target (Z=0).
            image_points_list: Lista di array (N, 2) di coordinate pixel corrispondenti.
            image_size: (width, height) del sensore in pixel.
            camera_id: Identificativo della telecamera calibrata.

        Returns:
            Istanza di CameraIntrinsics con i parametri stimati.
        """
        if len(object_points_list) == 0 or len(image_points_list) == 0:
            raise ValueError("Le liste di punti per la calibrazione non possono essere vuote.")
        if len(object_points_list) != len(image_points_list):
            raise ValueError("Il numero di object_points e image_points deve coincidere.")

        obj_pts_f32 = [np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in object_points_list]
        img_pts_f32 = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points_list]

        cv_flags = self.flags.to_opencv_flags()

        ret, K, dist, rvecs, tvecs, std_intr, std_extr, per_view = cv2.calibrateCameraExtended(
            objectPoints=obj_pts_f32,
            imagePoints=img_pts_f32,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=cv_flags,
            criteria=self.criteria,
        )

        return CameraIntrinsics(
            camera_id=camera_id,
            K=K,
            dist_coeffs=dist,
            image_size=image_size,
            mean_reprojection_error=float(ret),
        )

    def evaluate_on_test_set(
        self,
        object_points_list: Sequence[np.ndarray],
        image_points_list: Sequence[np.ndarray],
        intrinsics: CameraIntrinsics,
    ) -> Tuple[float, List[float]]:
        """
        Valida la calibrazione stimando la posa (solvePnP) su fotogrammi mai visti durante il training
        e calcolando l'errore di riproiezione out-of-sample per ciascuna vista.

        Args:
            object_points_list: Lista di punti 3D di test (N, 3).
            image_points_list: Lista di punti 2D di test (N, 2).
            intrinsics: Istanza di CameraIntrinsics precedentemente stimata.

        Returns:
            Tupla (mean_rms_error, per_frame_rms_errors) in pixel.
        """
        per_frame_errors: List[float] = []

        for obj_p, img_p in zip(object_points_list, image_points_list):
            pts_3d = np.asarray(obj_p, dtype=np.float64).reshape(-1, 3)
            pts_2d = np.asarray(img_p, dtype=np.float64).reshape(-1, 2)

            # Stima della posa su frame non visto con solvePnP
            success, rvec, tvec = cv2.solvePnP(
                pts_3d,
                pts_2d,
                intrinsics.K,
                intrinsics.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                continue

            projected = intrinsics.project_points(pts_3d, rvec, tvec)
            # Calcolo errore RMS euclideo sul frame
            residuals = np.linalg.norm(projected - pts_2d, axis=1)
            rms = float(np.sqrt(np.mean(residuals ** 2)))
            per_frame_errors.append(rms)

        if not per_frame_errors:
            return float("inf"), []

        mean_error = float(np.mean(per_frame_errors))
        return mean_error, per_frame_errors
