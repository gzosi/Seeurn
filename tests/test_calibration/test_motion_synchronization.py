"""
Unit & Integration tests per la sincronizzazione dei moti rigidi (Modulo 02, Fase 4A).
Riferimento Teorico: A. Fusiello, "Multi-view reconstruction" (Slide 43-59).
Metodologia: Test-Driven Development (TDD).

Verifica:
1. Sincronizzazione delle rotazioni (Rotation Synchronization) tramite decomposizione
   spettrale della matrice Z e proiezione su SO(3) via SVD (Listing 17).
2. Sincronizzazione metrica delle traslazioni (Translation Synchronization) tramite
   risoluzione ai minimi quadrati con matrice di incidenza B (Listing 18).
3. Dataclass SynchronizedCameraPose e MotionSynchronizationResult con serializzazione JSON.
4. Robustezza su dati sintetici con e senza rumore.
5. Sincronizzazione globale sulle stime stereoscopiche reali del setup Seeurn (wake2 dataset4).
"""

from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from seeurn.calibration.stereo import StereoExtrinsics, EpipolarGraph
from seeurn.calibration.synchronizer import (
    SynchronizedCameraPose,
    MotionSynchronizationResult,
    MotionSynchronizer,
)


class TestMotionSynchronizationSynthetic:
    """Verifica algoritmica su geometrie multi-camera 3D virtuali a verità nota."""

    @pytest.fixture
    def ground_truth_rig(self):
        """Crea un setup virtuale a 3 telecamere con pose note a priori."""
        # Camera 1 all'origine del mondo
        R1 = np.eye(3, dtype=np.float64)
        C1 = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

        # Camera 2: ruotata di 8 gradi attorno a Y, spostata a X = 820 mm, Y = 10 mm, Z = 25 mm
        theta2 = np.radians(8.0)
        R2 = np.array([
            [np.cos(theta2), 0.0, np.sin(theta2)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta2), 0.0, np.cos(theta2)],
        ], dtype=np.float64)
        C2 = np.array([[820.0], [10.0], [25.0]], dtype=np.float64)

        # Camera 3: vista laterale, ruotata di -25 gradi attorno a Y e 10 gradi attorno a X
        th3_y = np.radians(-25.0)
        th3_x = np.radians(10.0)
        Ry = np.array([[np.cos(th3_y), 0, np.sin(th3_y)], [0, 1, 0], [-np.sin(th3_y), 0, np.cos(th3_y)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(th3_x), -np.sin(th3_x)], [0, np.sin(th3_x), np.cos(th3_x)]])
        R3 = Rx @ Ry
        C3 = np.array([[600.0], [800.0], [-150.0]], dtype=np.float64)

        rotations = {1: R1, 2: R2, 3: R3}
        centers = {1: C1, 2: C2, 3: C3}

        # Costruzione trasformazioni relative (convenzione OpenCV):
        # Rij = Rj @ Ri.T
        # tij = -Rj @ (Cj - Ci)
        edges = {}
        for (i, j) in [(1, 2), (2, 3), (1, 3)]:
            Rij = rotations[j] @ rotations[i].T
            tij = -rotations[j] @ (centers[j] - centers[i])
            dummy_mat = np.eye(3)
            edges[(i, j)] = StereoExtrinsics(
                cam_i=i,
                cam_j=j,
                R=Rij,
                T=tij,
                E=dummy_mat,
                F=dummy_mat,
                mean_epipolar_error_px=0.0,
            )

        graph = EpipolarGraph()
        for ext in edges.values():
            graph.add_edge(ext)

        return rotations, centers, graph

    def test_rotation_synchronization_exact(self, ground_truth_rig):
        gt_rot, gt_centers, graph = ground_truth_rig
        sync = MotionSynchronizer()

        recovered_rotations = sync.synchronize_rotations(graph, ref_cam=1)
        assert len(recovered_rotations) == 3

        # Camera 1 deve essere esattamente l'identità
        np.testing.assert_allclose(recovered_rotations[1], np.eye(3), atol=1e-7)

        # Verifica fedeltà rispetto a ground truth per tutte le camere
        for cam_id in (1, 2, 3):
            R_diff = recovered_rotations[cam_id] @ gt_rot[cam_id].T
            angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
            assert np.degrees(angle_rad) < 1e-4

    def test_translation_synchronization_exact(self, ground_truth_rig):
        gt_rot, gt_centers, graph = ground_truth_rig
        sync = MotionSynchronizer()

        recovered_rot = sync.synchronize_rotations(graph, ref_cam=1)
        recovered_centers = sync.synchronize_translations(graph, recovered_rot, ref_cam=1)

        assert len(recovered_centers) == 3
        np.testing.assert_allclose(recovered_centers[1], np.zeros((3, 1)), atol=1e-7)

        for cam_id in (2, 3):
            np.testing.assert_allclose(recovered_centers[cam_id], gt_centers[cam_id], atol=1e-4)

    def test_complete_synchronization_noisy(self, ground_truth_rig):
        gt_rot, gt_centers, graph = ground_truth_rig
        sync = MotionSynchronizer()

        # Aggiungiamo rumore sintetico alle rotazioni relative (0.3 gradi) e alle traslazioni (2 mm)
        noisy_graph = EpipolarGraph()
        np.random.seed(42)
        for (i, j), edge in graph.edges.items():
            rvec_noise = np.random.uniform(-0.005, 0.005, (3, 1))
            R_noise, _ = cv2.Rodrigues(rvec_noise)
            t_noise = np.random.uniform(-2.0, 2.0, (3, 1))

            noisy_edge = StereoExtrinsics(
                cam_i=i,
                cam_j=j,
                R=R_noise @ edge.R,
                T=edge.T + t_noise,
                E=edge.E,
                F=edge.F,
                mean_epipolar_error_px=edge.mean_epipolar_error_px,
            )
            noisy_graph.add_edge(noisy_edge)

        result = sync.synchronize(noisy_graph, ref_cam=1)

        assert isinstance(result, MotionSynchronizationResult)
        assert len(result.camera_poses) == 3

        # Le discrepanze globali aggregate devono rimanere contenute
        assert result.max_rotation_discrepancy_deg < 1.0
        assert result.translation_residual_mm < 5.0

    def test_json_serialization(self, tmp_path: Path):
        pose1 = SynchronizedCameraPose(1, np.eye(3), np.zeros((3, 1)), np.zeros((3, 1)))
        pose2 = SynchronizedCameraPose(2, np.eye(3), np.array([[100.0], [0.0], [0.0]]), np.array([[-100.0], [0.0], [0.0]]))
        res = MotionSynchronizationResult(
            camera_poses={1: pose1, 2: pose2},
            max_rotation_discrepancy_deg=0.12,
            translation_residual_mm=1.45,
            reference_camera=1,
        )

        json_file = tmp_path / "sync_result.json"
        res.save_to_json(json_file)
        assert json_file.exists()

        loaded = MotionSynchronizationResult.load_from_json(json_file)
        assert loaded.reference_camera == 1
        assert np.isclose(loaded.max_rotation_discrepancy_deg, 0.12)
        assert np.isclose(loaded.translation_residual_mm, 1.45)
        np.testing.assert_allclose(loaded.camera_poses[2].T, pose2.T)
        np.testing.assert_allclose(loaded.camera_poses[2].C, pose2.C)


class TestMotionSynchronizationRealData:
    """Verifica della sincronizzazione globale sulle calibrazioni stereo reali del setup Seeurn."""

    CONFIG_DIR = Path("configs/calibration")
    OUT_DIR = Path("tests/output/phase4a_synchronization")

    def test_synchronize_seeurn_real_stereos(self):
        # Carica le 3 calibrazioni stereoscopiche consolidate nella Fase 3
        ext12 = StereoExtrinsics.load_from_json(self.CONFIG_DIR / "stereo_1_2_extrinsics.json")
        ext23 = StereoExtrinsics.load_from_json(self.CONFIG_DIR / "stereo_2_3_extrinsics.json")
        ext13 = StereoExtrinsics.load_from_json(self.CONFIG_DIR / "stereo_1_3_extrinsics.json")

        graph = EpipolarGraph()
        graph.add_edge(ext12)
        graph.add_edge(ext23)
        graph.add_edge(ext13)

        sync = MotionSynchronizer()
        result = sync.synchronize(graph, ref_cam=1)

        # 1. Camera 1 posta canonicamente all'origine del mondo
        pose1 = result.camera_poses[1]
        np.testing.assert_allclose(pose1.R, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(pose1.T, np.zeros((3, 1)), atol=1e-7)
        np.testing.assert_allclose(pose1.C, np.zeros((3, 1)), atol=1e-7)

        # 2. Camera 2: distanza del centro C2 da C1 deve coincidere con la baseline 1-2 (~819 mm)
        pose2 = result.camera_poses[2]
        dist_12 = float(np.linalg.norm(pose2.C - pose1.C))
        assert np.isclose(dist_12, ext12.baseline_mm, atol=2.0), f"Distanza C1-C2 anomala: {dist_12:.2f} mm vs {ext12.baseline_mm:.2f} mm"

        # 3. Camera 3: distanze del centro C3 da C1 e C2 devono essere coerenti con le baseline misurate
        pose3 = result.camera_poses[3]
        dist_13 = float(np.linalg.norm(pose3.C - pose1.C))
        dist_23 = float(np.linalg.norm(pose3.C - pose2.C))
        assert np.isclose(dist_13, ext13.baseline_mm, atol=4.0), f"Distanza C1-C3 anomala: {dist_13:.2f} mm vs {ext13.baseline_mm:.2f} mm"
        assert np.isclose(dist_23, ext23.baseline_mm, atol=4.0), f"Distanza C2-C3 anomala: {dist_23:.2f} mm vs {ext23.baseline_mm:.2f} mm"

        # 4. Residui e discrepanze di chiusura
        assert result.max_rotation_discrepancy_deg < 0.50
        assert result.translation_residual_mm < 5.0

        # 5. Esportazione degli artefatti persistenti per la Fase 4B
        artifacts = sync.export_artifacts(
            result=result,
            output_dir=self.OUT_DIR,
            config_dir=self.CONFIG_DIR,
        )

        assert artifacts["json_output"].exists()
        assert artifacts["json_config"].exists()
        assert artifacts["report"].exists()
        assert artifacts["diagram"].exists()
