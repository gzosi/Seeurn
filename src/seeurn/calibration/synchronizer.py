"""
Sincronizzazione Spaziale delle Pose Multi-Camera (Modulo 02, Fase 4A).

Riferimento Teorico:
Andrea Fusiello, "Multi-view reconstruction" (Slide 43-59).

Fornisce:
1. Dataclass SynchronizedCameraPose per modellare la posa estrinseca assoluta [R|T]
   e il centro ottico C nel sistema di riferimento globale del mondo (fissato su Cam 1).
2. Dataclass MotionSynchronizationResult con metriche di residuo e serializzazione JSON.
3. Classe MotionSynchronizer che implementa:
   - Sincronizzazione delle rotazioni (Rotation Synchronization) tramite decomposizione
     spettrale della matrice Z e proiezione su SO(3) via SVD (Listing 17).
   - Sincronizzazione metrica delle traslazioni (Translation Synchronization) tramite
     risoluzione lineare ai minimi quadrati con matrice di incidenza B (Listing 18).
4. Esportazione degli artefatti metrici e del layout geometrico della rete di telecamere.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import cv2
import numpy as np

from seeurn.calibration.stereo import StereoExtrinsics, EpipolarGraph


@dataclass(frozen=True)
class SynchronizedCameraPose:
    """
    Rappresenta la posa estrinseca assoluta di una telecamera nel sistema di riferimento globale.

    Convenzione:
        X_cam = R @ (X_world - C) = R @ X_world + T
        T = -R @ C
        C = -R^T @ T (coordinate 3D del centro ottico nel mondo in mm)
    """

    camera_id: int
    R: np.ndarray  # Matrice di rotazione 3x3 (float64) rispetto al mondo
    T: np.ndarray  # Vettore di traslazione 3x1 in mm (float64)
    C: np.ndarray  # Posizione 3D del centro ottico nel mondo 3x1 in mm (float64)

    def to_dict(self) -> Dict[str, Any]:
        """Converte la posa in un dizionario serializzabile in JSON."""
        return {
            "camera_id": self.camera_id,
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "C": self.C.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynchronizedCameraPose":
        """Ricostruisce un'istanza di SynchronizedCameraPose da un dizionario deserializzato."""
        return cls(
            camera_id=int(data["camera_id"]),
            R=np.array(data["R"], dtype=np.float64),
            T=np.array(data["T"], dtype=np.float64),
            C=np.array(data["C"], dtype=np.float64),
        )


@dataclass(frozen=True)
class MotionSynchronizationResult:
    """Risultato complessivo della sincronizzazione dei moti rigidi del grafo epipolare."""

    camera_poses: Dict[int, SynchronizedCameraPose]
    max_rotation_discrepancy_deg: float
    translation_residual_mm: float
    reference_camera: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serializza il risultato in dizionario JSON."""
        return {
            "reference_camera": self.reference_camera,
            "max_rotation_discrepancy_deg": float(self.max_rotation_discrepancy_deg),
            "translation_residual_mm": float(self.translation_residual_mm),
            "camera_poses": {
                str(cam_id): pose.to_dict()
                for cam_id, pose in sorted(self.camera_poses.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionSynchronizationResult":
        """Deserializza il risultato da dizionario JSON."""
        poses = {
            int(cam_id): SynchronizedCameraPose.from_dict(pose_dict)
            for cam_id, pose_dict in data["camera_poses"].items()
        }
        return cls(
            camera_poses=poses,
            max_rotation_discrepancy_deg=float(data["max_rotation_discrepancy_deg"]),
            translation_residual_mm=float(data["translation_residual_mm"]),
            reference_camera=int(data.get("reference_camera", 1)),
        )

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Salva il risultato della sincronizzazione in un file JSON formattato."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> "MotionSynchronizationResult":
        """Carica il risultato della sincronizzazione da file JSON."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MotionSynchronizer:
    """
    Sincronizzatore globale dei moti rigidi delle telecamere basato sulla teoria
    spettrale e sui minimi quadrati su grafi epolari (A. Fusiello, Slide 43-59).
    """

    def synchronize_rotations(
        self,
        graph: EpipolarGraph,
        ref_cam: int = 1,
    ) -> Dict[int, np.ndarray]:
        """
        Risolve il problema globale di Rotation Synchronization (Listing 17):
            min_{R_1, ..., R_m in SO(3)} sum_{(i,j)} || R_ij - R_i @ R_j^T ||_F^2

        Costruisce la matrice simmetrica a blocchi Z (3m x 3m) e ne calcola i 3 autovettori
        dominanti della matrice pesata (D otimes I_3)^(-1) Z, proiettandoli su SO(3) con SVD.

        Args:
            graph: Istanza di EpipolarGraph contenente le stime pairwise.
            ref_cam: ID della telecamera di riferimento (fissata con R = Identità).

        Returns:
            Dizionario {camera_id: R_3x3} con le rotazioni globali coerenti.
        """
        # Estrazione nodi del grafo
        nodes = sorted(list(set(c for edge in graph.edges.keys() for c in edge)))
        m = len(nodes)
        if m < 2:
            raise ValueError(f"Richieste almeno 2 telecamere per la sincronizzazione, trovate {m}")

        node_to_idx = {node: i for i, node in enumerate(nodes)}
        if ref_cam not in node_to_idx:
            raise ValueError(f"Telecamera di riferimento {ref_cam} non presente nel grafo.")

        # Inizializzazione matrice Z (3m x 3m) e matrice di adiacenza A (m x m)
        Z = np.zeros((3 * m, 3 * m), dtype=np.float64)
        A = np.zeros((m, m), dtype=np.float64)

        # Diagonale principale di Z = I_3
        for i in range(m):
            Z[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.eye(3, dtype=np.float64)

        # Popolamento degli archi (Z simmetrica: Z_ji = Z_ij^T)
        for i, ci in enumerate(nodes):
            for j, cj in enumerate(nodes):
                if i < j:
                    try:
                        ext = graph.get_edge(ci, cj)
                        # In OpenCV ext.R mappa da ci a cj: R_ij_cv = R_j @ R_i^T
                        # Nel formalismo di Fusiello: Z_ij = R_i @ R_j^T = (R_ij_cv)^T
                        Z[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = ext.R.T
                        Z[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = ext.R
                        A[i, j] = 1.0
                        A[j, i] = 1.0
                    except ValueError:
                        pass

        # Matrice di grado D
        degrees = np.sum(A, axis=1)
        if np.any(degrees == 0):
            raise ValueError("Il grafo epipolare è sconnesso: impossibile sincronizzare le rotazioni.")

        # Matrice di transizione M = (D^(-1) otimes I_3) @ Z
        inv_degrees = 1.0 / degrees
        inv_D_kron = np.kron(np.diag(inv_degrees), np.eye(3, dtype=np.float64))
        M = inv_D_kron @ Z

        # Autovalori e autovettori dominanti
        evals, evecs = np.linalg.eig(M)
        # Ordiniamo per parte reale decrescente degli autovalori
        idx_sorted = np.argsort(np.real(evals))[::-1]
        top3_indices = idx_sorted[:3]

        # Matrice Q (3m x 3)
        Q = np.real(evecs[:, top3_indices])

        # Normalizzazione rispetto alla camera di riferimento
        ref_idx = node_to_idx[ref_cam]
        Q_ref = Q[3 * ref_idx : 3 * ref_idx + 3, :3]
        if np.linalg.matrix_rank(Q_ref) < 3:
            # Fallback se il blocco di riferimento è degenere
            Q_ref_inv = np.linalg.pinv(Q_ref)
        else:
            Q_ref_inv = np.linalg.inv(Q_ref)

        Q_normalized = Q @ Q_ref_inv

        # Proiezione di ciascun blocco 3x3 su SO(3) tramite SVD
        rotations: Dict[int, np.ndarray] = {}
        for i, cam_id in enumerate(nodes):
            if cam_id == ref_cam:
                rotations[cam_id] = np.eye(3, dtype=np.float64)
                continue

            Q_i = Q_normalized[3 * i : 3 * i + 3, :3]
            U, _, Vt = np.linalg.svd(Q_i)
            # Garantisce det = +1 (rotazione pura, nessuna riflessione)
            S_det = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
            R_proj = U @ S_det @ Vt
            rotations[cam_id] = np.ascontiguousarray(R_proj, dtype=np.float64)

        return rotations

    def synchronize_translations(
        self,
        graph: EpipolarGraph,
        rotations: Dict[int, np.ndarray],
        ref_cam: int = 1,
    ) -> Dict[int, np.ndarray]:
        """
        Risolve il problema lineare di Translation Synchronization (Listing 18):
            u_ij = -R_j^T @ T_ij = C_j - C_i
            X @ B = U  =>  (B_1^T otimes I_3) @ vec(X_1) = vec(U)

        dove B è la matrice di incidenza del grafo, C_i sono i centri ottici delle camere
        e ref_cam è vincolata a C_ref = [0, 0, 0]^T.

        Args:
            graph: Istanza di EpipolarGraph.
            rotations: Rotazioni globali sincronizzate {cam_id: R_3x3}.
            ref_cam: ID della telecamera all'origine.

        Returns:
            Dizionario {camera_id: C_3x1} con le coordinate 3D dei centri ottici in mm.
        """
        nodes = sorted(list(set(c for edge in graph.edges.keys() for c in edge)))
        m = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Raccogliamo gli archi canonici orientati con cam_i < cam_j
        unique_edges: List[StereoExtrinsics] = []
        for (ci, cj), ext in graph.edges.items():
            if ci < cj:
                unique_edges.append(ext)
            elif (cj, ci) not in graph.edges:
                unique_edges.append(ext)

        num_edges = len(unique_edges)
        if num_edges < m - 1:
            raise ValueError(f"Numero di archi insufficiente per connettere {m} nodi: {num_edges}")

        # Matrice di incidenza B (m x num_edges) e matrice U (3 x num_edges)
        B = np.zeros((m, num_edges), dtype=np.float64)
        U = np.zeros((3, num_edges), dtype=np.float64)

        for k, ext in enumerate(unique_edges):
            i_idx = node_to_idx[ext.cam_i]
            j_idx = node_to_idx[ext.cam_j]

            B[i_idx, k] = -1.0
            B[j_idx, k] = 1.0

            # Dislocamento del centro C_j rispetto a C_i nello spazio globale:
            # Nel sistema stereo X_j = R_ij @ X_i + T_ij  =>  C_j - C_i = -R_j^T @ T_ij
            R_j = rotations[ext.cam_j]
            u_ij = -R_j.T @ ext.T
            U[:, k] = u_ij.ravel()

        # Vincoliamo C_ref = 0 rimuovendo la riga corrispondente da B
        ref_idx = node_to_idx[ref_cam]
        B1 = np.delete(B, ref_idx, axis=0)  # dimensione: (m-1) x num_edges

        # Risoluzione ai minimi quadrati:
        # (B1^T otimes I_3) @ vec(X1) = vec(U)
        A_lin = np.kron(B1.T, np.eye(3, dtype=np.float64))  # (3*num_edges) x (3*(m-1))
        b_lin = U.T.reshape(-1)                             # (3*num_edges,)

        # Risoluzione least squares con pseudoinversa robusta
        x1_vec, residuals, rank, s = np.linalg.lstsq(A_lin, b_lin, rcond=None)
        X1 = x1_vec.reshape(m - 1, 3).T  # 3 x (m-1)

        # Ricostruzione matrice completa dei centri X (3 x m)
        X = np.zeros((3, m), dtype=np.float64)
        col_dest = 0
        for col_src in range(m):
            if col_src == ref_idx:
                X[:, col_src] = 0.0
            else:
                X[:, col_src] = X1[:, col_dest]
                col_dest += 1

        centers: Dict[int, np.ndarray] = {
            cam_id: np.ascontiguousarray(X[:, idx].reshape(3, 1), dtype=np.float64)
            for cam_id, idx in node_to_idx.items()
        }

        return centers

    def synchronize(
        self,
        graph: EpipolarGraph,
        ref_cam: int = 1,
    ) -> MotionSynchronizationResult:
        """
        Esegue la sincronizzazione globale completa (Rotazioni + Centri ottici + Traslazioni).

        Args:
            graph: EpipolarGraph contenente le connessioni a 3 viste.
            ref_cam: ID della telecamera fissata all'origine del mondo.

        Returns:
            Istanza di MotionSynchronizationResult pronta per inizializzare il Bundle Adjustment.
        """
        rotations = self.synchronize_rotations(graph, ref_cam=ref_cam)
        centers = self.synchronize_translations(graph, rotations, ref_cam=ref_cam)

        camera_poses: Dict[int, SynchronizedCameraPose] = {}
        for cam_id, R in rotations.items():
            C = centers[cam_id]
            T = -R @ C
            camera_poses[cam_id] = SynchronizedCameraPose(
                camera_id=cam_id,
                R=R,
                T=T,
                C=C,
            )

        # Calcolo discrepanza angolare massima su tutte le coppie disponibili
        rot_errors = []
        for (ci, cj), ext in graph.edges.items():
            R_pred = rotations[cj] @ rotations[ci].T
            R_diff = R_pred @ ext.R.T
            angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
            rot_errors.append(float(np.degrees(angle_rad)))

        max_rot_err = float(np.max(rot_errors)) if rot_errors else 0.0

        # Calcolo residuo di traslazione globale ||X @ B - U||
        trans_res_list = []
        for (ci, cj), ext in graph.edges.items():
            disp_measured = ext.baseline_mm
            disp_sync = float(np.linalg.norm(centers[cj] - centers[ci]))
            trans_res_list.append(abs(disp_sync - disp_measured))

        trans_res = float(np.max(trans_res_list)) if trans_res_list else 0.0

        return MotionSynchronizationResult(
            camera_poses=camera_poses,
            max_rotation_discrepancy_deg=max_rot_err,
            translation_residual_mm=trans_res,
            reference_camera=ref_cam,
        )

    def export_artifacts(
        self,
        result: MotionSynchronizationResult,
        output_dir: Union[str, Path],
        config_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Esporta in modo persistente tutti i risultati della sincronizzazione delle pose:
        1. JSON in tests/output/phase4a_synchronization/ e in configs/calibration/
        2. Report Markdown diagnostico con coordinate metriche dei centri e matrici R
        3. Immagine grafica top-down 2D del layout metrico dei centri ottici (in mm)

        Returns:
            Dizionario con i percorsi dei file generati.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        json_out = out_p / "synchronized_camera_poses.json"
        result.save_to_json(json_out)

        json_cfg = None
        if config_dir is not None:
            cfg_p = Path(config_dir)
            cfg_p.mkdir(parents=True, exist_ok=True)
            json_cfg = cfg_p / "synchronized_camera_poses.json"
            result.save_to_json(json_cfg)

        # Creazione diagramma top-down 2D del layout del rig (in mm) con OpenCV
        diagram_path = out_p / "camera_network_layout.png"
        self._render_camera_layout_diagram(result, diagram_path)

        # Creazione report Markdown
        report_path = out_p / "synchronization_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Report Fase 4A: Sincronizzazione Spaziale delle Pose (Motion Synchronization)\n\n")
            f.write(f"- **Telecamera di Riferimento Mondo:** `Camera {result.reference_camera}`\n")
            f.write(f"- **Discrepanza Massima di Rotazione SO(3):** `{result.max_rotation_discrepancy_deg:.4f}°`\n")
            f.write(f"- **Residuo Massimo di Traslazione:** `{result.translation_residual_mm:.4f} mm`\n\n")
            f.write("## Coordinate Metriche dei Centri Ottici nel Mondo [mm]\n\n")
            f.write("| Telecamera | X [mm] | Y [mm] | Z [mm] | Distanza da Cam 1 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")

            C1 = result.camera_poses[1].C
            for cam_id, pose in sorted(result.camera_poses.items()):
                C = pose.C.ravel()
                d_from_1 = float(np.linalg.norm(pose.C - C1))
                f.write(f"| **Camera {cam_id}** | `{C[0]:.2f}` | `{C[1]:.2f}` | `{C[2]:.2f}` | `{d_from_1:.2f} mm` |\n")

            f.write("\n## Matrici di Rotazione Assolute R_i\n```\n")
            for cam_id, pose in sorted(result.camera_poses.items()):
                f.write(f"--- Camera {cam_id} ---\n{pose.R}\n\n")
            f.write("```\n")

        readme_path = out_p / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Risultati Fase 4A: Motion Synchronization\n\n")
            f.write("Questa directory contiene i risultati della sincronizzazione globale dei moti rigidi ")
            f.write("del setup a 3 telecamere, propedeutica al Bundle Block Adjustment (Fase 4B).\n\n")
            f.write("- [`synchronized_camera_poses.json`](./synchronized_camera_poses.json): Parametri estrinseci globali\n")
            f.write("- [`camera_network_layout.png`](./camera_network_layout.png): Schema geometrico top-down dei centri ottici\n")
            f.write("- [`synchronization_report.md`](./synchronization_report.md): Dettaglio numerico completo\n")

        return {
            "json_output": json_out,
            "json_config": json_cfg or json_out,
            "report": report_path,
            "diagram": diagram_path,
        }

    def _render_camera_layout_diagram(
        self,
        result: MotionSynchronizationResult,
        out_path: Path,
    ) -> None:
        """Traccia una mappa top-down del sistema multi-camera su canvas 2D con OpenCV."""
        canvas_w, canvas_h = 1000, 800
        canvas = np.full((canvas_h, canvas_w, 3), 25, dtype=np.uint8)  # Sfondo scuro

        # Griglia di riferimento
        for x in range(0, canvas_w, 100):
            cv2.line(canvas, (x, 0), (x, canvas_h), (40, 40, 40), 1)
        for y in range(0, canvas_h, 100):
            cv2.line(canvas, (0, y), (canvas_w, y), (40, 40, 40), 1)

        # Centri in mm nel piano X-Z (o X-Y a seconda della configurazione)
        # Troviamo min/max per scalare
        xs = [pose.C[0, 0] for pose in result.camera_poses.values()]
        ys = [pose.C[1, 0] for pose in result.camera_poses.values()]
        zs = [pose.C[2, 0] for pose in result.camera_poses.values()]

        min_x, max_x = min(xs) - 150, max(xs) + 150
        min_y, max_y = min(ys) - 150, max(ys) + 150
        range_x = max(max_x - min_x, 100)
        range_y = max(max_y - min_y, 100)

        scale = min((canvas_w - 200) / range_x, (canvas_h - 200) / range_y)

        def to_px(x_mm: float, y_mm: float) -> Tuple[int, int]:
            px = int(100 + (x_mm - min_x) * scale)
            py = int(canvas_h - 100 - (y_mm - min_y) * scale)
            return px, py

        # Traccia linee del grafo (triangolo)
        centers_px = {
            cam_id: to_px(pose.C[0, 0], pose.C[1, 0])
            for cam_id, pose in result.camera_poses.items()
        }

        # Connessioni tra camere
        pairs = [(1, 2), (2, 3), (1, 3)]
        for c1, c2 in pairs:
            if c1 in centers_px and c2 in centers_px:
                p1 = centers_px[c1]
                p2 = centers_px[c2]
                cv2.line(canvas, p1, p2, (100, 100, 100), 2, cv2.LINE_AA)
                # Testo distanza in mm al centro del segmento
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                dist_mm = np.linalg.norm(result.camera_poses[c1].C - result.camera_poses[c2].C)
                cv2.putText(
                    canvas,
                    f"{dist_mm:.1f} mm",
                    (mid_x - 30, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

        # Nodi delle camere con colori distinti
        cam_colors = {1: (0, 165, 255), 2: (0, 255, 0), 3: (255, 100, 0)}  # BGR
        for cam_id, pt in centers_px.items():
            col = cam_colors.get(cam_id, (255, 255, 255))
            cv2.circle(canvas, pt, 14, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 16, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Cam {cam_id}",
                (pt[0] - 25, pt[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Intestazione grafica
        cv2.putText(
            canvas,
            "Seeurn Multi-Camera Rig Layout (Phase 4A Synchronized Poses)",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(out_path), canvas)
