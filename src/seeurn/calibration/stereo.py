"""
Calibrazione Stereo e Grafo Epipolare Multi-Camera (Modulo 02, Fase 3).

Fornisce:
1. Dataclass StereoExtrinsics per modellare le trasformazioni rigide relative [R|T],
   la matrice essenziale E e la matrice fondamentale F tra coppie di viste.
2. Calcolo della distanza epipolare simmetrica punto-linea (pixel).
3. Grafo Epipolare (EpipolarGraph) con verifica geometrica rigorosa del Loop Closure
   tra la tripletta di telecamere C1 -> C2 -> C3 -> C1.
4. Calibratore stereoscopico StereoPairCalibrator basato su cv2.stereoCalibrateExtended,
   con vincolo fisso sulle matrici intrinseche monoculari (CALIB_FIX_INTRINSIC).
5. Visualizzazione diagnostica delle linee epipolari ed esportazione artefatti per il Second Brain.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import cv2
import numpy as np

from seeurn.calibration.monocular import CameraIntrinsics

# Tavolozza di colori distinti BGR per il disegno delle corrispondenze epipolari
DISTINCT_COLORS_BGR = [
    (0, 0, 255),      # Rosso
    (0, 255, 0),      # Verde
    (255, 0, 0),      # Blu
    (0, 255, 255),    # Giallo
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Ciano
    (0, 165, 255),    # Arancione
    (203, 192, 255),  # Rosa
    (128, 0, 128),    # Viola
    (0, 128, 128),    # Oliva
    (128, 128, 0),    # Teal
    (255, 255, 255),  # Bianco
]


@dataclass(frozen=True)
class StereoExtrinsics:
    """
    Rappresenta la relazione geometrica stereoscopica estrinseca tra due telecamere.

    Trasformazione rigida:
        X_j = R @ X_i + T  (coordinate 3D espresse in mm)
    Vincolo epipolare:
        p_j^T @ F @ p_i = 0  (coordinate 2D in pixel omogenee)
        x_j^T @ E @ x_i = 0  (coordinate normalizzate)
    """

    cam_i: int
    cam_j: int
    R: np.ndarray             # Matrice di rotazione 3x3 (float64)
    T: np.ndarray             # Vettore di traslazione 3x1 in mm (float64)
    E: np.ndarray             # Matrice Essenziale 3x3 (float64)
    F: np.ndarray             # Matrice Fondamentale 3x3 (float64)
    mean_epipolar_error_px: float

    @property
    def baseline_mm(self) -> float:
        """Restituisce la distanza metrica lineare (baseline) in millimetri tra i centri ottici."""
        return float(np.linalg.norm(self.T))

    def invert(self) -> "StereoExtrinsics":
        """
        Calcola la trasformazione inversa da camera j a camera i:
            X_i = R^T @ X_j - R^T @ T
            F_{ji} = F_{ij}^T
            E_{ji} = E_{ij}^T
        """
        R_inv = self.R.T
        T_inv = -self.R.T @ self.T
        E_inv = self.E.T
        F_inv = self.F.T

        return StereoExtrinsics(
            cam_i=self.cam_j,
            cam_j=self.cam_i,
            R=np.ascontiguousarray(R_inv, dtype=np.float64),
            T=np.ascontiguousarray(T_inv, dtype=np.float64),
            E=np.ascontiguousarray(E_inv, dtype=np.float64),
            F=np.ascontiguousarray(F_inv, dtype=np.float64),
            mean_epipolar_error_px=self.mean_epipolar_error_px,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converte le informazioni estrinseche in un dizionario serializzabile JSON."""
        return {
            "cam_i": self.cam_i,
            "cam_j": self.cam_j,
            "baseline_mm": self.baseline_mm,
            "mean_epipolar_error_px": float(self.mean_epipolar_error_px),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StereoExtrinsics":
        """Ricostruisce un'istanza di StereoExtrinsics da un dizionario deserializzato."""
        return cls(
            cam_i=int(data["cam_i"]),
            cam_j=int(data["cam_j"]),
            R=np.array(data["R"], dtype=np.float64),
            T=np.array(data["T"], dtype=np.float64),
            E=np.array(data["E"], dtype=np.float64),
            F=np.array(data["F"], dtype=np.float64),
            mean_epipolar_error_px=float(data["mean_epipolar_error_px"]),
        )

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Salva i parametri estrinseci della coppia stereo in un file JSON formattato."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> "StereoExtrinsics":
        """Carica un'istanza di StereoExtrinsics da un file JSON salvato su disco."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_epipolar_distance(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """
    Calcola la distanza epipolare simmetrica (in pixel) per ogni coppia di punti corrispondenti.

    Per ciascun punto:
    1. Linea epipolare nella vista j: l_j = F @ p_i = [a_j, b_j, c_j]^T
       Distanza di p_j da l_j: d_j = |p_j^T @ l_j| / sqrt(a_j^2 + b_j^2)
    2. Linea epipolare nella vista i: l_i = F^T @ p_j = [a_i, b_i, c_i]^T
       Distanza di p_i da l_i: d_i = |p_j^T @ F @ p_i| / sqrt(a_i^2 + b_i^2)
    3. Distanza simmetrica: d_sym = (d_i + d_j) / 2.0

    Args:
        pts_i: Array (N, 2) delle coordinate pixel nella prima telecamera.
        pts_j: Array (N, 2) delle coordinate pixel nella seconda telecamera.
        F: Matrice fondamentale 3x3.

    Returns:
        Array 1D (N,) contenente la distanza epipolare simmetrica in pixel per ciascun punto.
    """
    p1 = np.asarray(pts_i, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts_j, dtype=np.float64).reshape(-1, 2)
    n_points = p1.shape[0]

    if p2.shape[0] != n_points:
        raise ValueError(f"Numero di punti disallineato: {n_points} vs {p2.shape[0]}")

    # Coordinate omogenee (N, 3)
    p1_h = np.hstack([p1, np.ones((n_points, 1), dtype=np.float64)])
    p2_h = np.hstack([p2, np.ones((n_points, 1), dtype=np.float64)])

    # Linee epipolari
    # l2 = F @ p1 (forma: (N, 3))
    l2 = (F @ p1_h.T).T
    # l1 = F^T @ p2 (forma: (N, 3))
    l1 = (F.T @ p2_h.T).T

    # Numeratore: |p2^T @ F @ p1| = |p2^T @ l2|
    numerator = np.abs(np.sum(p2_h * l2, axis=1))

    # Denominatori: sqrt(a^2 + b^2)
    denom_2 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-12
    denom_1 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-12

    d2 = numerator / denom_2
    d1 = numerator / denom_1

    return 0.5 * (d1 + d2)


class EpipolarGraph:
    """
    Rappresenta la rete di telecamere e le loro trasformazioni rigide relative.
    Consente di comporre percorsi nel grafo e verificare analiticamente la chiusura del ciclo (Loop Closure).
    """

    def __init__(self):
        self.edges: Dict[Tuple[int, int], StereoExtrinsics] = {}

    def add_edge(self, extrinsics: StereoExtrinsics) -> None:
        """Aggiunge una connessione stereo al grafo."""
        key = (extrinsics.cam_i, extrinsics.cam_j)
        self.edges[key] = extrinsics

    def get_edge(self, c_i: int, c_j: int) -> StereoExtrinsics:
        """
        Recupera la trasformazione rigida orientata tra c_i e c_j.
        Se nel grafo è presente l'arco inverso (c_j, c_i), ne calcola automaticamente l'inversa.
        """
        if (c_i, c_j) in self.edges:
            return self.edges[(c_i, c_j)]
        if (c_j, c_i) in self.edges:
            return self.edges[(c_j, c_i)].invert()
        raise ValueError(f"Nessun arco stereo trovato nel grafo tra Camera {c_i} e Camera {c_j}.")

    def check_loop_closure(self, c1: int = 1, c2: int = 2, c3: int = 3) -> Tuple[float, float]:
        """
        Calcola l'errore di rotazione (in gradi) e di traslazione (in mm)
        nella chiusura del triangolo stereoscopico c1 -> c2 -> c3 -> c1.

        Verifica:
            R_{13}^{teorica} = R_{23} @ R_{12}
            T_{13}^{teorica} = R_{23} @ T_{12} + T_{23}

        Discrepanza con la calibrazione diretta [R_{13} | T_{13}]:
            R_{diff} = R_{13}^{teorica} @ R_{13}^T
            angolo = arccos((Tr(R_{diff}) - 1) / 2)
            delta_T = || T_{13}^{teorica} - T_{13} ||_2

        Returns:
            Tuple (rotation_error_deg, translation_error_mm)
        """
        try:
            e12 = self.get_edge(c1, c2)
            e23 = self.get_edge(c2, c3)
            e13 = self.get_edge(c1, c3)
        except ValueError as err:
            raise ValueError(f"Mancano archi nel grafo per chiudere il ciclo stereo: {err}") from err

        R_composed = e23.R @ e12.R
        T_composed = e23.R @ e12.T + e23.T

        # Errore di rotazione (angolo geodetico SO(3))
        R_diff = R_composed @ e13.R.T
        trace_val = (np.trace(R_diff) - 1.0) / 2.0
        angle_rad = np.arccos(np.clip(trace_val, -1.0, 1.0))
        angle_deg = float(np.degrees(angle_rad))

        # Errore lineare di traslazione (distanza euclidea in mm)
        trans_error_mm = float(np.linalg.norm(T_composed - e13.T))

        return angle_deg, trans_error_mm


class StereoPairCalibrator:
    """
    Calibratore stereoscopico di coppie di telecamere.
    Sfrutta cv2.stereoCalibrateExtended fissando le matrici intrinseche e i coefficienti
    di distorsione precedentemente ottimizzati nella Fase 2 (CALIB_FIX_INTRINSIC).
    """

    def __init__(
        self,
        intrinsics_i: CameraIntrinsics,
        intrinsics_j: CameraIntrinsics,
    ):
        self.intrinsics_i = intrinsics_i
        self.intrinsics_j = intrinsics_j

    def calibrate(
        self,
        object_points_list: List[np.ndarray],
        image_points_i_list: List[np.ndarray],
        image_points_j_list: List[np.ndarray],
        fix_intrinsics: bool = True,
        max_iters: int = 100,
        eps: float = 1e-6,
    ) -> StereoExtrinsics:
        """
        Esegue la calibrazione stereo sui punti 3D-2D accoppiati.

        Args:
            object_points_list: Lista di array (N, 3) float32 delle coordinate scacchiera in mm.
            image_points_i_list: Lista di array (N, 2) float32 per la telecamera i.
            image_points_j_list: Lista di array (N, 2) float32 per la telecamera j.
            fix_intrinsics: Se True (raccomandato), congela K e dist_coeffs calcolati in monoculare.
            max_iters: Numero massimo di iterazioni Levenberg-Marquardt.
            eps: Criterio di convergenza.

        Returns:
            Istanza di StereoExtrinsics ottimizzata.
        """
        n_frames = len(object_points_list)
        if n_frames < 3:
            raise ValueError(f"Richiesti almeno 3 frame accoppiati per la calibrazione stereo, trovati {n_frames}")

        if len(image_points_i_list) != n_frames or len(image_points_j_list) != n_frames:
            raise ValueError("Le liste di punti oggetto e immagine devono avere la stessa lunghezza.")

        # Preparazione tipi per OpenCV
        obj_pts = [np.ascontiguousarray(p, dtype=np.float32).reshape(-1, 3) for p in object_points_list]
        img_pts_i = [np.ascontiguousarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points_i_list]
        img_pts_j = [np.ascontiguousarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points_j_list]

        K1 = self.intrinsics_i.K.copy().astype(np.float64)
        dist1 = self.intrinsics_i.dist_coeffs.copy().astype(np.float64)
        K2 = self.intrinsics_j.K.copy().astype(np.float64)
        dist2 = self.intrinsics_j.dist_coeffs.copy().astype(np.float64)

        flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iters, eps)

        # In OpenCV stereoCalibrateExtended il parametro imageSize corrisponde alle dimensioni della camera
        img_size_i = (int(self.intrinsics_i.image_size[0]), int(self.intrinsics_i.image_size[1]))

        # Chiamata estesa di calibrazione stereo
        (
            ret,
            K1_out,
            dist1_out,
            K2_out,
            dist2_out,
            R,
            T,
            E,
            F,
            rvecs,
            tvecs,
            per_view_err,
        ) = cv2.stereoCalibrateExtended(
            objectPoints=obj_pts,
            imagePoints1=img_pts_i,
            imagePoints2=img_pts_j,
            cameraMatrix1=K1,
            distCoeffs1=dist1,
            cameraMatrix2=K2,
            distCoeffs2=dist2,
            imageSize=img_size_i,
            R=None,
            T=None,
            flags=flags,
            criteria=criteria,
        )

        # Calcolo analitico dell'errore epipolare simmetrico medio su tutti i punti
        all_epi_errors = []
        for pts_i, pts_j in zip(image_points_i_list, image_points_j_list):
            errs = compute_epipolar_distance(pts_i, pts_j, F)
            all_epi_errors.append(errs)

        mean_epipolar_px = float(np.mean(np.concatenate(all_epi_errors)))

        return StereoExtrinsics(
            cam_i=self.intrinsics_i.camera_id,
            cam_j=self.intrinsics_j.camera_id,
            R=np.ascontiguousarray(R, dtype=np.float64),
            T=np.ascontiguousarray(T, dtype=np.float64),
            E=np.ascontiguousarray(E, dtype=np.float64),
            F=np.ascontiguousarray(F, dtype=np.float64),
            mean_epipolar_error_px=mean_epipolar_px,
        )

    def evaluate(
        self,
        image_points_i_list: List[np.ndarray],
        image_points_j_list: List[np.ndarray],
        F: np.ndarray,
    ) -> Tuple[float, List[float]]:
        """
        Valuta le discrepanze epipolari su un set di test out-of-sample.

        Returns:
            Tuple (overall_mean_error_px, per_frame_mean_errors_px)
        """
        per_frame_means: List[float] = []
        all_errs: List[np.ndarray] = []

        for p_i, p_j in zip(image_points_i_list, image_points_j_list):
            dist = compute_epipolar_distance(p_i, p_j, F)
            per_frame_means.append(float(np.mean(dist)))
            all_errs.append(dist)

        overall_mean = float(np.mean(np.concatenate(all_errs)))
        return overall_mean, per_frame_means

    def visualize_epipolar_lines(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        pts_i: np.ndarray,
        pts_j: np.ndarray,
        F: np.ndarray,
        num_sample_points: int = 10,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Genera una visualizzazione diagnostica side-by-side delle linee epipolari corrispondenti.

        Args:
            img_i: Immagine della telecamera i (uint8 BGR o Grayscale).
            img_j: Immagine della telecamera j (uint8 BGR o Grayscale).
            pts_i: Punti (N, 2) in telecamera i.
            pts_j: Punti (N, 2) in telecamera j.
            F: Matrice fondamentale 3x3.
            num_sample_points: Numero di punti da tracciare per non affollare l'immagine.
            seed: Seed di selezione casuale dei punti.

        Returns:
            Immagine BGR composita side-by-side con linee e punti colorati.
        """
        # Conversione in BGR per disegno a colori
        im1 = cv2.cvtColor(img_i, cv2.COLOR_GRAY2BGR) if img_i.ndim == 2 else img_i.copy()
        im2 = cv2.cvtColor(img_j, cv2.COLOR_GRAY2BGR) if img_j.ndim == 2 else img_j.copy()

        p1 = np.asarray(pts_i, dtype=np.float32).reshape(-1, 2)
        p2 = np.asarray(pts_j, dtype=np.float32).reshape(-1, 2)
        n_pts = len(p1)

        rng = np.random.default_rng(seed)
        sample_indices = (
            rng.choice(n_pts, size=min(num_sample_points, n_pts), replace=False)
            if n_pts > num_sample_points
            else np.arange(n_pts)
        )

        # Calcolo linee epipolari con OpenCV
        # computeCorrespondEpilines: points deve essere (N, 1, 2)
        p1_sub = p1[sample_indices].reshape(-1, 1, 2)
        p2_sub = p2[sample_indices].reshape(-1, 1, 2)

        lines2 = cv2.computeCorrespondEpilines(p1_sub, 1, F).reshape(-1, 3)
        lines1 = cv2.computeCorrespondEpilines(p2_sub, 2, F).reshape(-1, 3)

        # Tavolozza di colori distinti per ogni linea/punto
        colors = [
            DISTINCT_COLORS_BGR[k % len(DISTINCT_COLORS_BGR)]
            for k in range(len(sample_indices))
        ]

        # Traccia linee in Cam 1 e punti corrispondenti
        h1, w1 = im1.shape[:2]
        for r, pt1, color in zip(lines1, p1[sample_indices], colors):
            # r = [a, b, c], a*x + b*y + c = 0 -> x = 0: y = -c/b; x = w: y = -(c + a*w)/b
            x0, y0 = 0, int(-r[2] / (r[1] + 1e-9))
            x1, y1 = w1, int(-(r[2] + r[0] * w1) / (r[1] + 1e-9))
            cv2.line(im1, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            cv2.circle(im1, (int(pt1[0]), int(pt1[1])), 6, color, -1, cv2.LINE_AA)
            cv2.circle(im1, (int(pt1[0]), int(pt1[1])), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Traccia linee in Cam 2 e punti corrispondenti
        h2, w2 = im2.shape[:2]
        for r, pt2, color in zip(lines2, p2[sample_indices], colors):
            x0, y0 = 0, int(-r[2] / (r[1] + 1e-9))
            x1, y1 = w2, int(-(r[2] + r[0] * w2) / (r[1] + 1e-9))
            cv2.line(im2, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            cv2.circle(im2, (int(pt2[0]), int(pt2[1])), 6, color, -1, cv2.LINE_AA)
            cv2.circle(im2, (int(pt2[0]), int(pt2[1])), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Unione affiancata gestendo altezze differenti con canvas unificato
        max_h = max(h1, h2)
        total_w = w1 + w2
        canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)

        canvas[:h1, :w1] = im1
        canvas[:h2, w1:w1 + w2] = im2

        # Intestazioni grafiche
        cv2.putText(
            canvas,
            f"Camera {self.intrinsics_i.camera_id} (Left)",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Camera {self.intrinsics_j.camera_id} (Right)",
            (w1 + 30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def export_artifacts(
        self,
        extrinsics: StereoExtrinsics,
        test_img_i: np.ndarray,
        test_img_j: np.ndarray,
        test_pts_i: np.ndarray,
        test_pts_j: np.ndarray,
        output_dir: Union[str, Path],
        config_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Esporta in modo persistente tutti i risultati e le visualizzazioni della coppia stereo:
        1. JSON metrico in tests/output/phase3_stereo/ e in configs/calibration/
        2. Immagine PNG delle linee epipolari corrispondenti
        3. Report Markdown di sintesi metrica

        Returns:
            Dizionario con i percorsi dei file generati.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        pair_tag = f"stereo_{extrinsics.cam_i}_{extrinsics.cam_j}"

        # 1. Salvataggio JSON
        json_out = out_p / f"{pair_tag}_extrinsics.json"
        extrinsics.save_to_json(json_out)

        if config_dir is not None:
            cfg_p = Path(config_dir)
            cfg_p.mkdir(parents=True, exist_ok=True)
            extrinsics.save_to_json(cfg_p / f"{pair_tag}_extrinsics.json")

        # 2. Visualizzazione linee epipolari
        overlay = self.visualize_epipolar_lines(
            test_img_i, test_img_j, test_pts_i, test_pts_j, extrinsics.F
        )
        overlay_path = out_p / f"{pair_tag}_epipolar_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        # 3. Report Markdown
        report_path = out_p / f"{pair_tag}_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Report Calibrazione Stereo: Cam {extrinsics.cam_i} - Cam {extrinsics.cam_j}\n\n")
            f.write(f"- **Baseline Metrica:** `{extrinsics.baseline_mm:.2f} mm`\n")
            f.write(f"- **Errore Epipolare Medio:** `{extrinsics.mean_epipolar_error_px:.4f} px`\n")
            f.write("\n## Vettore di Traslazione T [mm]\n```\n")
            f.write(f"Tx = {extrinsics.T[0, 0]:.3f} mm\n")
            f.write(f"Ty = {extrinsics.T[1, 0]:.3f} mm\n")
            f.write(f"Tz = {extrinsics.T[2, 0]:.3f} mm\n```\n")
            f.write("\n## Matrice di Rotazione R\n```\n")
            f.write(f"{extrinsics.R}\n```\n")
            f.write("\n## Matrice Fondamentale F\n```\n")
            f.write(f"{extrinsics.F}\n```\n")

        return {
            "json": json_out,
            "overlay": overlay_path,
            "report": report_path,
        }
