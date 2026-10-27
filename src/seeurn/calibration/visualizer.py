"""
Visualizzatore diagnostico per la calibrazione e il controllo di qualità dei corner (Modulo 02).
Produce immagini annotate con assi di riferimento e mappe di copertura del sensore (Sensor Coverage Maps).
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np


class CornerVisualizer:
    """
    Genera feedback visivo per l'ispezione della calibrazione:
    1. Immagini con corner rilevati, assi del riferimento locale della scacchiera e punto zero evidenziato.
    2. Mappe di copertura del sensore (Sensor Coverage Map) per verificare che la scacchiera
       abbia spaziato uniformemente su tutto il campo visivo (centro e bordi).
    """

    def __init__(self):
        pass

    def draw_annotated_corners(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        pattern_size: Tuple[int, int] = (15, 9),
        frame_idx: int = 0,
        camera_id: int = 1,
        method: str = "sb_raw",
    ) -> np.ndarray:
        """
        Disegna la scacchiera sull'immagine originale con assi e indicazione dell'origine PT_0.

        Args:
            image: Immagine uint8 (1 canale o 3 canali).
            corners: Coordinate 2D degli spigoli (N, 2).
            pattern_size: Dimensioni della griglia (nx, ny).
            frame_idx: Indice progressivo del frame.
            camera_id: ID della telecamera.
            method: Metodo di rilevamento utilizzato.

        Returns:
            Immagine BGR annotata.
        """
        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        pts = corners.reshape(-1, 2).astype(np.float32)
        nx, ny = pattern_size

        # 1. Disegna i corner standard di OpenCV
        cv2.drawChessboardCorners(vis, pattern_size, pts.reshape(-1, 1, 2), True)

        # 2. Evidenzia l'origine fisica PT_0 (in alto a sinistra, vicino al bullone)
        p0 = tuple(np.round(pts[0]).astype(int))
        cv2.circle(vis, p0, 14, (255, 0, 255), 3, cv2.LINE_AA)  # Magenta
        cv2.circle(vis, p0, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            "PT_0 (Origin)",
            (p0[0] + 18, p0[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # 3. Disegna gli assi locali della scacchiera
        # Asse X: direzione lungo la larghezza del pattern (da pt 0 a pt 1)
        if len(pts) > 1:
            p_x = tuple(np.round(pts[1]).astype(int))
            # Vettore direzione normalizzato e allungato per visibilità
            vx = np.array([p_x[0] - p0[0], p_x[1] - p0[1]], dtype=float)
            norm_x = np.linalg.norm(vx)
            if norm_x > 0:
                arrow_x = tuple((np.array(p0) + (vx / norm_x) * 60).astype(int))
                cv2.arrowedLine(vis, p0, arrow_x, (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(
                    vis, "X", (arrow_x[0] + 8, arrow_x[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
                )

        # Asse Y: direzione lungo l'altezza del pattern (da pt 0 a pt nx)
        if len(pts) > nx:
            p_y = tuple(np.round(pts[nx]).astype(int))
            vy = np.array([p_y[0] - p0[0], p_y[1] - p0[1]], dtype=float)
            norm_y = np.linalg.norm(vy)
            if norm_y > 0:
                arrow_y = tuple((np.array(p0) + (vy / norm_y) * 60).astype(int))
                cv2.arrowedLine(vis, p0, arrow_y, (0, 255, 0), 3, tipLength=0.3)
                cv2.putText(
                    vis, "Y", (arrow_y[0] + 5, arrow_y[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

        # 4. Banner informativo superiore
        h, w = vis.shape[:2]
        banner_h = 42
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

        info_text = (
            f"Camera {camera_id} | Frame #{frame_idx:06d} | "
            f"Corners: {len(pts)}/{nx * ny} | Method: {method}"
        )
        cv2.putText(
            vis, info_text, (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA
        )

        return vis

    def create_sensor_coverage_map(
        self,
        sensor_size: Tuple[int, int],
        all_corners_list: Sequence[np.ndarray],
        camera_id: int = 1,
        total_frames_sampled: int = 50,
        camera_name: str = "",
    ) -> Tuple[np.ndarray, float]:
        """
        Crea una mappa grafica 2D della copertura del sensore ottico.
        Visualizza tutti i punti rilevati sui vari frame, evidenziando il poligono di copertura
        (Convex Hull) e calcolando la percentuale dell'area del sensore coperta.

        Args:
            sensor_size: (width, height) in pixel del sensore (es. (2176, 1400) o (1280, 800)).
            all_corners_list: Lista di array (135, 2) con i corner rilevati sui frame validi.
            camera_id: Identificativo della telecamera.
            total_frames_sampled: Numero totale di frame campionati per il test.
            camera_name: Nome descrittivo facoltativo.

        Returns:
            Tupla (canvas_bgr, coverage_percentage).
        """
        w, h = sensor_size

        # Risoluzione interna del grafico (scalata per una visualizzazione ottimale a schermo)
        canvas_w = 1280
        canvas_h = int(canvas_w * (h / w))
        canvas = np.full((canvas_h + 80, canvas_w, 3), (25, 20, 15), dtype=np.uint8)  # Blu navy scuro

        scale_x = canvas_w / w
        scale_y = canvas_h / h

        # 1. Griglia di riferimento del sensore (ogni 10% del campo visivo)
        for pct in range(10, 100, 10):
            gx = int(canvas_w * (pct / 100))
            gy = int(canvas_h * (pct / 100))
            cv2.line(canvas, (gx, 60), (gx, canvas_h + 60), (45, 38, 30), 1)
            cv2.line(canvas, (0, gy + 60), (canvas_w, gy + 60), (45, 38, 30), 1)

        # 2. Rettangolo del bordo del sensore
        cv2.rectangle(canvas, (0, 60), (canvas_w - 1, canvas_h + 59), (180, 140, 60), 2)

        # 3. Raccogli e mappa tutti i punti nel riferimento della canvas
        all_pts_flat: List[Tuple[float, float]] = []
        for corners in all_corners_list:
            if corners is not None and len(corners) > 0:
                pts = corners.reshape(-1, 2)
                for pt in pts:
                    all_pts_flat.append((pt[0] * scale_x, pt[1] * scale_y + 60))

        coverage_ratio = 0.0
        total_corners_count = len(all_pts_flat)

        if len(all_pts_flat) >= 3:
            pts_array = np.array(all_pts_flat, dtype=np.float32)
            hull = cv2.convexHull(pts_array).astype(np.int32)

            # Poligono semi-trasparente per l'area coperta
            poly_overlay = canvas.copy()
            cv2.fillPoly(poly_overlay, [hull], (70, 130, 40))  # Verde/azzurro trasparente
            cv2.addWeighted(poly_overlay, 0.35, canvas, 0.65, 0, canvas)
            cv2.polylines(canvas, [hull], True, (120, 230, 80), 2, cv2.LINE_AA)

            # Calcolo percentuale area: Area(Hull) / Area(Sensore)
            hull_area_canvas = cv2.contourArea(hull)
            sensor_area_canvas = canvas_w * canvas_h
            coverage_ratio = (hull_area_canvas / sensor_area_canvas) * 100.0

        # Disegna tutti i corner come punti luminosi
        for pt in all_pts_flat:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (0, 255, 200), -1, cv2.LINE_AA)

        # 4. Banner superiore con statistiche
        cv2.rectangle(canvas, (0, 0), (canvas_w, 60), (35, 28, 20), -1)
        valid_frames_count = len(all_corners_list)
        success_pct = (valid_frames_count / max(total_frames_sampled, 1)) * 100.0

        header_left = f"Camera {camera_id} ({camera_name or f'Cam {camera_id}'}) - Sensore: {w}x{h} px"
        header_right = (
            f"Frame Rilevati: {valid_frames_count}/{total_frames_sampled} ({success_pct:.0f}%) | "
            f"Corner Totali: {total_corners_count} | Copertura: {coverage_ratio:.1f}%"
        )
        cv2.putText(
            canvas, header_left, (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, header_right, (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 160), 2, cv2.LINE_AA
        )

        return canvas, coverage_ratio

