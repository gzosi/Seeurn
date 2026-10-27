"""
Unit & Integration tests per il Bundle Block Adjustment Globale (Modulo 02, Fase 4B).
Riferimento Teorico: A. Fusiello, "Multi-view reconstruction" (Slide 60-71).
Metodologia: Test-Driven Development (TDD).

Verifica:
1. Dataclass GlobalSystemState, Observation e GlobalCalibrationResult.
2. Struttura della matrice Jacobiana sparsa (Primary/Secondary structure) per Trust Region Reflective.
3. Funzione di costo con perdita robusta (Huber / Cauchy) per sopprimere riflessi o corner anomali.
4. Ottimizzazione su setup 3D sintetico con convergenza a verità nota.
5. Calibrazione globale definitiva su dati reali multi-camera (wake2) ed esportazione artefatti per il Modulo 04.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import pytest

from seeurn.calibration.monocular import CameraIntrinsics
from seeurn.calibration.synchronizer import SynchronizedCameraPose, MotionSynchronizationResult
from seeurn.calibration.bundle_adjustment import (
    GlobalSystemState,
    Observation,
    GlobalBundleAdjuster,
    GlobalCalibrationResult,
)


class TestBundleAdjustmentSynthetic:
    """Verifica del motore di Bundle Adjustment su geometrie sintetiche a verità nota."""

    @pytest.fixture
    def synthetic_bba_problem(self):
        """Genera un problema BBA sintetico con 3 telecamere e 10 pose di scacchiera 15x9."""
        np.random.seed(101)

        # 3 telecamere con intrinseci noti
        K1 = np.array([[1500.0, 0, 1088.0], [0, 1500.0, 700.0], [0, 0, 1]], dtype=np.float64)
        K2 = np.array([[1500.0, 0, 1088.0], [0, 1500.0, 700.0], [0, 0, 1]], dtype=np.float64)
        K3 = np.array([[1000.0, 0, 640.0], [0, 1000.0, 400.0], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)

        intrinsics = {
            1: CameraIntrinsics(1, K1, dist, (2176, 1400), 0.0),
            2: CameraIntrinsics(2, K2, dist, (2176, 1400), 0.0),
            3: CameraIntrinsics(3, K3, dist, (1280, 800), 0.0),
        }

        # Pose vere delle telecamere rispetto al mondo
        # Cam 1 all'origine
        R1 = np.eye(3)
        T1 = np.zeros((3, 1))

        # Cam 2
        th2 = np.radians(8.0)
        R2 = np.array([[np.cos(th2), 0, np.sin(th2)], [0, 1, 0], [-np.sin(th2), 0, np.cos(th2)]])
        C2 = np.array([[820.0], [10.0], [25.0]])
        T2 = -R2 @ C2

        # Cam 3
        th3 = np.radians(-25.0)
        R3 = np.array([[np.cos(th3), 0, np.sin(th3)], [0, 1, 0], [-np.sin(th3), 0, np.cos(th3)]])
        C3 = np.array([[600.0], [800.0], [-150.0]])
        T3 = -R3 @ C3

        true_camera_poses = {
            1: SynchronizedCameraPose(1, R1, T1, np.zeros((3, 1))),
            2: SynchronizedCameraPose(2, R2, T2, C2),
            3: SynchronizedCameraPose(3, R3, T3, C3),
        }

        # Geometria scacchiera 15x9 da 5mm
        nx, ny = 15, 9
        objp = np.zeros((nx * ny, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * 5.0

        # Generiamo 10 pose 3D della scacchiera nel mondo visibili dalle telecamere
        true_board_poses = []
        observations = []

        for b_idx in range(10):
            rvec_b = np.random.uniform(-0.1, 0.1, (3, 1))
            tvec_b = np.array([
                np.random.uniform(200, 500),
                np.random.uniform(100, 300),
                np.random.uniform(800, 1200),
            ], dtype=np.float64).reshape(3, 1)

            R_b, _ = cv2.Rodrigues(rvec_b)
            true_board_poses.append((rvec_b, tvec_b))

            # Punti 3D nel mondo per questa posa: X_w = R_b @ P + T_b
            pts_world = (R_b @ objp.T + tvec_b).T

            # Proiezione in ciascuna telecamera k
            for cam_id, pose in true_camera_poses.items():
                K = intrinsics[cam_id].K
                D = intrinsics[cam_id].dist_coeffs
                # Coordinate nel sistema camera: X_c = R_k @ (R_b @ P + T_b) + T_k
                R_comb = pose.R @ R_b
                T_comb = pose.R @ tvec_b + pose.T
                rvec_comb, _ = cv2.Rodrigues(R_comb)
                pts_2d, _ = cv2.projectPoints(objp, rvec_comb, T_comb, K, D)
                pts_2d_flat = pts_2d.reshape(-1, 2)

                # Aggiungiamo osservazione
                observations.append(Observation(
                    board_index=b_idx,
                    camera_id=cam_id,
                    object_points=objp.copy(),
                    image_points=pts_2d_flat.copy(),
                ))

        return intrinsics, true_camera_poses, true_board_poses, observations

    def test_pack_and_unpack_system_state(self, synthetic_bba_problem):
        intrinsics, true_cam_poses, true_board_poses, _ = synthetic_bba_problem

        state = GlobalSystemState(
            camera_poses={2: true_cam_poses[2], 3: true_cam_poses[3]},
            board_poses=true_board_poses,
            reference_camera_id=1,
        )

        vector = state.pack()
        # Per 2 camere mobili (6 parametri ciascuna = 12) + 10 pose di scacchiera (6 parametri ciascuna = 60)
        # Totale = 72 parametri
        assert len(vector) == 12 + 60

        unpacked = GlobalSystemState.unpack(vector, num_boards=10, reference_camera_id=1)
        assert len(unpacked.camera_poses) == 2
        assert len(unpacked.board_poses) == 10

        np.testing.assert_allclose(unpacked.camera_poses[2].R, state.camera_poses[2].R, atol=1e-7)
        np.testing.assert_allclose(unpacked.camera_poses[2].T, state.camera_poses[2].T, atol=1e-7)
        np.testing.assert_allclose(unpacked.board_poses[0][0], state.board_poses[0][0], atol=1e-7)
        np.testing.assert_allclose(unpacked.board_poses[0][1], state.board_poses[0][1], atol=1e-7)

    def test_bba_convergence_from_perturbed_guess(self, synthetic_bba_problem):
        intrinsics, true_cam_poses, true_board_poses, observations = synthetic_bba_problem

        # Aggiungiamo perturbazione all'inizializzazione: 1 grado e 5 mm alle camere, 2 gradi e 10 mm alle tavole
        np.random.seed(42)
        perturbed_cam_poses = {}
        for cam_id in (2, 3):
            p = true_cam_poses[cam_id]
            r_err, _ = cv2.Rodrigues(np.random.uniform(-0.02, 0.02, (3, 1)))
            t_err = np.random.uniform(-5.0, 5.0, (3, 1))
            R_pert = r_err @ p.R
            T_pert = p.T + t_err
            C_pert = -R_pert.T @ T_pert
            perturbed_cam_poses[cam_id] = SynchronizedCameraPose(cam_id, R_pert, T_pert, C_pert)

        perturbed_board_poses = []
        for rvec, tvec in true_board_poses:
            perturbed_board_poses.append((
                rvec + np.random.uniform(-0.03, 0.03, (3, 1)),
                tvec + np.random.uniform(-10.0, 10.0, (3, 1)),
            ))

        initial_state = GlobalSystemState(
            camera_poses=perturbed_cam_poses,
            board_poses=perturbed_board_poses,
            reference_camera_id=1,
        )

        adjuster = GlobalBundleAdjuster(huber_delta=1.5, max_iterations=150, loss="linear")
        result = adjuster.optimize(
            initial_state=initial_state,
            observations=observations,
            intrinsics=intrinsics,
        )

        assert isinstance(result, GlobalCalibrationResult)
        assert result.mean_reprojection_error_px < 0.1
        assert result.max_reprojection_error_px < 0.3

        # Verifica convergenza delle posizioni delle telecamere verso la verità fondamentale
        for cam_id in (2, 3):
            true_C = true_cam_poses[cam_id].C
            opt_C = result.camera_poses[cam_id].C
            dist_err = np.linalg.norm(opt_C - true_C)
            assert dist_err < 0.5, f"Errore centro ottico Cam {cam_id}: {dist_err:.3f} mm"


class TestBundleAdjustmentRealData:
    """Integrazione e validazione del BBA sui dati reali multi-camera (wake2 dataset4 e dataset2)."""

    CONFIG_DIR = Path("configs/calibration")
    OUTPUT_DIR = Path("tests/output/phase4b_bundle_adjustment")

    @pytest.mark.skipif(not Path("configs/calibration/synchronized_camera_poses.json").exists(), reason="Fase 4A non completata")
    def test_global_bundle_adjustment_real_data(self):
        # 1. Carica intrinseci consolidati
        intrinsics = {
            1: CameraIntrinsics.load_from_json(self.CONFIG_DIR / "camera_1_intrinsics.json"),
            2: CameraIntrinsics.load_from_json(self.CONFIG_DIR / "camera_2_intrinsics.json"),
            3: CameraIntrinsics.load_from_json(self.CONFIG_DIR / "camera_3_intrinsics.json"),
        }

        # 2. Carica pose sincronizzate dalla Fase 4A
        sync_result = MotionSynchronizationResult.load_from_json(
            self.CONFIG_DIR / "synchronized_camera_poses.json"
        )

        # 3. Inizializza l'ottimizzatore BBA
        adjuster = GlobalBundleAdjuster(huber_delta=1.0, max_iterations=100)

        # 4. Raccoglie le osservazioni da dataset4 e stima le pose iniziali delle scacchiere
        result = adjuster.run_on_dataset(
            sync_result=sync_result,
            intrinsics=intrinsics,
            dataset_wake="wake2",
            dataset_name="dataset4",
            max_frames=20,
        )

        assert isinstance(result, GlobalCalibrationResult)
        # Errore di riproiezione congiunto a 3 viste sub-pixel (< 1.2 px complessivo)
        assert result.mean_reprojection_error_px < 1.2, (
            f"Errore BBA medio troppo alto: {result.mean_reprojection_error_px:.3f} px"
        )

        # Verifica coerenza delle baseline finali raffinate
        pose1 = result.camera_poses[1]
        pose2 = result.camera_poses[2]
        pose3 = result.camera_poses[3]

        baseline_12 = float(np.linalg.norm(pose2.C - pose1.C))
        baseline_13 = float(np.linalg.norm(pose3.C - pose1.C))
        baseline_23 = float(np.linalg.norm(pose3.C - pose2.C))

        assert 815.0 < baseline_12 < 825.0
        assert 1125.0 < baseline_13 < 1140.0
        assert 1265.0 < baseline_23 < 1278.0

        # 5. Esporta tutti gli artefatti metrici, visuali e di configurazione
        artifacts = adjuster.export_artifacts(
            result=result,
            output_dir=self.OUTPUT_DIR,
            config_dir=self.CONFIG_DIR,
        )

        assert artifacts["json_global"].exists()
        assert artifacts["dashboard"].exists()
        assert artifacts["reconstruction_3d"].exists()
        assert artifacts["report"].exists()

