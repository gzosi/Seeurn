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

    def draw_undistortion_comparison(
        self,
        distorted: np.ndarray,
        undistorted: np.ndarray,
        camera_id: int = 1,
        grid_spacing: int = 120,
    ) -> np.ndarray:
        """
        Crea un'immagine diagnostica affiancata (Side-by-Side):
        - Sinistra: Immagine originale (distorta dal sistema ottico/rifrazione).
        - Destra: Immagine rettificata con modello a 12/14 parametri.
        - Entrambe con reticolo di riferimento ortogonale per evidenziare visivamente
          la rimozione delle curvature ai bordi e l'effetto prismatico.
        """
        h, w = distorted.shape[:2]
        vis_dist = cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR) if distorted.ndim == 2 else distorted.copy()
        vis_undist = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR) if undistorted.ndim == 2 else undistorted.copy()

        # Disegna reticolo di riferimento su entrambe
        for gx in range(grid_spacing, w, grid_spacing):
            cv2.line(vis_dist, (gx, 0), (gx, h), (0, 165, 255), 1, cv2.LINE_AA)
            cv2.line(vis_undist, (gx, 0), (gx, h), (0, 220, 120), 1, cv2.LINE_AA)
        for gy in range(grid_spacing, h, grid_spacing):
            cv2.line(vis_dist, (0, gy), (w, gy), (0, 165, 255), 1, cv2.LINE_AA)
            cv2.line(vis_undist, (0, gy), (w, gy), (0, 220, 120), 1, cv2.LINE_AA)

        # Scala per visualizzazione confortevole (larghezza totale 1600 px)
        target_w = 800
        target_h = int(target_w * (h / w))
        s_dist = cv2.resize(vis_dist, (target_w, target_h), interpolation=cv2.INTER_AREA)
        s_undist = cv2.resize(vis_undist, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Unione affiancata
        banner_h = 50
        canvas = np.zeros((target_h + banner_h, target_w * 2, 3), dtype=np.uint8)
        canvas[banner_h:, :target_w] = s_dist
        canvas[banner_h:, target_w:] = s_undist

        # Banner superiore
        cv2.rectangle(canvas, (0, 0), (target_w * 2, banner_h), (25, 20, 15), -1)
        cv2.putText(
            canvas,
            f"Camera {camera_id}: Immagine Originale (Distorta)",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Camera {camera_id}: Rettificata (Razionale + Thin-Prism)",
            (target_w + 20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 240, 140),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def draw_reprojection_overlay(
        self,
        image: np.ndarray,
        measured_corners: np.ndarray,
        projected_corners: np.ndarray,
        frame_idx: int = 0,
        camera_id: int = 1,
        error_scale: float = 10.0,
    ) -> np.ndarray:
        """
        Visualizza la sovrapposizione tra i corner misurati sull'immagine e quelli riproiettati dal modello:
        - Cerchi verdi: corner estratti con sub-pixel precision.
        - Croci rosse: corner 3D riproiettati con la matrice intrinseca stimata.
        - Segmenti gialli: vettori di residuo amplificati per renderli chiaramente visibili.
        """
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        m_pts = measured_corners.reshape(-1, 2)
        p_pts = projected_corners.reshape(-1, 2)

        residuals = np.linalg.norm(p_pts - m_pts, axis=1)
        mean_res = float(np.mean(residuals))

        for (mx, my), (px, py) in zip(m_pts, p_pts):
            imx, imy = int(round(mx)), int(round(my))
            # Corner misurato (cerchio verde)
            cv2.circle(vis, (imx, imy), 4, (0, 255, 100), 1, cv2.LINE_AA)

            # Corner proiettato (croce rossa)
            ipx, ipy = int(round(px)), int(round(py))
            cv2.drawMarker(vis, (ipx, ipy), (0, 0, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

            # Vettore residuo amplificato (segmento giallo)
            dx = (px - mx) * error_scale
            dy = (py - my) * error_scale
            cv2.line(vis, (imx, imy), (int(round(mx + dx)), int(round(my + dy))), (0, 255, 255), 1, cv2.LINE_AA)

        # Banner informativo
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 42), (20, 20, 20), -1)
        text = (
            f"Camera {camera_id} | Frame #{frame_idx:06d} | "
            f"Corner: {len(m_pts)} | Errore RMS: {mean_res:.3f} px (Vettori gialli: {error_scale:.0f}x)"
        )
        cv2.putText(vis, text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        return vis

    def draw_reprojection_dashboard(
        self,
        residuals_2d: np.ndarray,
        per_frame_errors: Sequence[float],
        camera_id: int = 1,
        train_rms: float = 0.0,
        test_rms: float = 0.0,
    ) -> np.ndarray:
        """
        Dashboard grafica della qualità di riproiezione monoculare (Modulo 02):
        - Sinistra: Scatter plot dei residui 2D (du, dv) attorno all'origine (0, 0) con cerchi da 0.5 e 1.0 px.
        - Destra: Grafico a barre degli errori RMS vista per vista con soglia limite di 0.5 px.
        """
        canvas_w, canvas_h = 1280, 560
        canvas = np.full((canvas_h, canvas_w, 3), (25, 20, 15), dtype=np.uint8)

        # 1. Banner Superiore
        cv2.rectangle(canvas, (0, 0), (canvas_w, 60), (35, 28, 20), -1)
        header = f"Camera {camera_id}: Dashboard Analitica Calibrazione Monoculare"
        metrics = f"Train RMS: {train_rms:.3f} px | Test Out-of-Sample: {test_rms:.3f} px | Viste Totali: {len(per_frame_errors)}"
        cv2.putText(canvas, header, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, metrics, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 160), 2, cv2.LINE_AA)

        # 2. Pannello Sinistro: Scatter Plot dei Residui (cx=320, cy=300)
        cx, cy = 320, 300
        radius_px_1 = 120  # 120 px canvas = 1.0 px di errore
        scale = radius_px_1 / 1.0

        # Riferimenti circolari a 0.25, 0.5, 1.0 pixel
        cv2.circle(canvas, (cx, cy), int(0.25 * scale), (45, 40, 35), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(0.5 * scale), (60, 55, 45), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(1.0 * scale), (80, 70, 50), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx - 160, cy), (cx + 160, cy), (60, 55, 45), 1)
        cv2.line(canvas, (cx, cy - 160), (cx, cy + 160), (60, 55, 45), 1)

        cv2.putText(canvas, "Dispersione Residui 2D (du, dv)", (cx - 140, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "r = 0.5 px", (cx + int(0.5 * scale) + 6, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, "r = 1.0 px", (cx + int(1.0 * scale) + 6, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 140, 200), 1, cv2.LINE_AA)

        # Plotta i punti del residuo
        pts_res = residuals_2d.reshape(-1, 2)
        for du, dv in pts_res:
            px = int(round(cx + du * scale))
            py = int(round(cy + dv * scale))
            if 0 <= px < canvas_w and 60 <= py < canvas_h:
                cv2.circle(canvas, (px, py), 2, (0, 230, 255), -1, cv2.LINE_AA)

        # 3. Pannello Destro: Grafico a Barre degli Errori per Vista
        bx_start = 660
        bx_end = 1240
        by_base = 460
        by_top = 130
        max_err_display = 1.5
        chart_h = by_base - by_top

        cv2.putText(canvas, "Errore RMS per Fotogramma (px)", (bx_start, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Linea di soglia 0.5 px (sub-pixel)
        y_thresh = int(by_base - (0.5 / max_err_display) * chart_h)
        cv2.line(canvas, (bx_start, y_thresh), (bx_end, y_thresh), (0, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Soglia sub-pixel (0.5 px)", (bx_end - 170, y_thresh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1, cv2.LINE_AA)

        # Asse base
        cv2.line(canvas, (bx_start, by_base), (bx_end, by_base), (120, 100, 80), 1)

        num_bars = len(per_frame_errors)
        if num_bars > 0:
            bar_w = max(2, int((bx_end - bx_start - 20) / num_bars) - 2)
            for idx, err in enumerate(per_frame_errors):
                bx = bx_start + 10 + idx * (bar_w + 2)
                bar_h = int(min(err / max_err_display, 1.0) * chart_h)
                color = (0, 220, 120) if err < 0.5 else (0, 160, 240) if err < 1.0 else (60, 60, 230)
                cv2.rectangle(canvas, (bx, by_base - bar_h), (bx + bar_w, by_base), color, -1)

        return canvas

