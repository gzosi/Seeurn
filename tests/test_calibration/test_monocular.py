"""
Unit & Integration tests per la calibrazione monoculare (Modulo 02, Fase 2).
Metodologia: Test-Driven Development (TDD).
Verifica modelli di distorsione complessi (fino a 14 parametri di OpenCV),
serializzazione/deserializzazione JSON di CameraIntrinsics, calibrazione sintetica
e validazione out-of-sample su dati reali di galleria.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from seeurn.calibration.monocular import (
    DistortionModelFlags,
    CameraIntrinsics,
    MonocularCalibrator,
)
from seeurn.calibration.target_detector import CheckerboardConfig, RobustCheckerboardDetector
from seeurn.data.catalog import CalibrationCatalog
from seeurn.data.preprocessor import ImagePreprocessor


class TestDistortionModelFlags:
    """Verifica la configurazione dei flag di distorsione OpenCV."""

    def test_default_flags_enable_extended_model(self):
        flags = DistortionModelFlags()
        assert flags.use_rational_model is True
        assert flags.use_thin_prism is True
        assert flags.use_tilted is True
        assert flags.fix_aspect_ratio is False

        cv_flags = flags.to_opencv_flags()
        assert cv_flags & cv2.CALIB_RATIONAL_MODEL
        assert cv_flags & cv2.CALIB_THIN_PRISM_MODEL
        assert cv_flags & cv2.CALIB_TILTED_MODEL

    def test_standard_pinhole_flags(self):
        flags = DistortionModelFlags(
            use_rational_model=False,
            use_thin_prism=False,
            use_tilted=False,
        )
        cv_flags = flags.to_opencv_flags()
        assert not (cv_flags & cv2.CALIB_RATIONAL_MODEL)
        assert not (cv_flags & cv2.CALIB_THIN_PRISM_MODEL)
        assert not (cv_flags & cv2.CALIB_TILTED_MODEL)


class TestCameraIntrinsics:
    """Verifica la dataclass CameraIntrinsics, proiezioni e serializzazione."""

    @pytest.fixture
    def sample_intrinsics(self) -> CameraIntrinsics:
        K = np.array([
            [1600.0, 0.0, 1088.0],
            [0.0, 1600.0, 700.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist = np.zeros((1, 14), dtype=np.float64)
        dist[0, 0] = -0.1  # k1
        dist[0, 1] = 0.02  # k2
        return CameraIntrinsics(
            camera_id=1,
            K=K,
            dist_coeffs=dist,
            image_size=(2176, 1400),
            mean_reprojection_error=0.22,
        )

    def test_project_points(self, sample_intrinsics: CameraIntrinsics):
        obj_pts = np.array([
            [0.0, 0.0, 500.0],
            [10.0, 10.0, 500.0],
            [-10.0, -10.0, 600.0],
        ], dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        proj = sample_intrinsics.project_points(obj_pts, rvec, tvec)
        assert proj.shape == (3, 2)
        # Il punto sull'asse ottico (0, 0, 500) deve proiettarsi vicino al centro ottico (1088, 700)
        np.testing.assert_allclose(proj[0], [1088.0, 700.0], atol=1.0)

    def test_json_serialization_roundtrip(self, sample_intrinsics: CameraIntrinsics, tmp_path: Path):
        json_path = tmp_path / "camera_1_intrinsics.json"
        sample_intrinsics.save_to_json(json_path)
        assert json_path.exists()

        loaded = CameraIntrinsics.load_from_json(json_path)
        assert loaded.camera_id == sample_intrinsics.camera_id
        assert loaded.image_size == sample_intrinsics.image_size
        assert np.isclose(loaded.mean_reprojection_error, sample_intrinsics.mean_reprojection_error)
        np.testing.assert_allclose(loaded.K, sample_intrinsics.K)
        np.testing.assert_allclose(loaded.dist_coeffs, sample_intrinsics.dist_coeffs)

    def test_undistort_image(self, sample_intrinsics: CameraIntrinsics):
        dummy_img = np.zeros((1400, 2176, 3), dtype=np.uint8)
        dummy_img[650:750, 1000:1100] = 255
        undistorted = sample_intrinsics.undistort(dummy_img)
        assert undistorted.shape == dummy_img.shape


class TestMonocularCalibratorSynthetic:
    """Verifica metrologica della calibrazione su viste sintetiche controllate."""

    def test_calibrate_synthetic_views(self):
        # 1. Configurazione target ideale 15x9 (5.0 mm)
        target_cfg = CheckerboardConfig(pattern_size=(15, 9), square_size_mm=5.0)
        obj_pts = target_cfg.generate_object_points()

        # 2. Parametri telecamera noti a priori
        K_true = np.array([
            [1200.0, 0.0, 640.0],
            [0.0, 1200.0, 400.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_true = np.zeros((1, 5), dtype=np.float64)
        dist_true[0, 0] = -0.15  # k1
        dist_true[0, 1] = 0.05   # k2
        img_size = (1280, 800)

        # 3. Genera 10 pose sintetiche della scacchiera nello spazio 3D
        np.random.seed(42)
        obj_points_list = []
        img_points_list = []
        for i in range(12):
            rvec = np.random.uniform(-0.3, 0.3, size=(3, 1)).astype(np.float64)
            # Traslazione a circa 400-600 mm dalla telecamera
            tvec = np.array([
                np.random.uniform(-50.0, 50.0),
                np.random.uniform(-30.0, 30.0),
                np.random.uniform(400.0, 600.0),
            ], dtype=np.float64).reshape(3, 1)

            pts_2d, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_true, dist_true)
            obj_points_list.append(obj_pts.copy())
            img_points_list.append(pts_2d.reshape(-1, 2).astype(np.float32))

        # 4. Calibra con MonocularCalibrator
        calibrator = MonocularCalibrator(
            flags=DistortionModelFlags(use_rational_model=False, use_thin_prism=False, use_tilted=False)
        )
        intrinsics = calibrator.calibrate(
            object_points_list=obj_points_list,
            image_points_list=img_points_list,
            image_size=img_size,
            camera_id=1,
        )

        # 5. Verifica che l'errore medio sia sub-pixel
        assert intrinsics.mean_reprojection_error < 0.1, (
            f"Errore di riproiezione sintetico troppo alto: {intrinsics.mean_reprojection_error:.3f} px"
        )
        # Focale stimata vicina a 1200
        np.testing.assert_allclose(intrinsics.K[0, 0], 1200.0, rtol=0.05)
        np.testing.assert_allclose(intrinsics.K[1, 1], 1200.0, rtol=0.05)


class TestRealDataMonocularCalibration:
    """
    Test di integrazione su dati reali:
    Calibra Camera 1 e Camera 2 su 25 frame di mono_train (wake2/dataset2),
    quindi valida l'errore di riproiezione su 10 frame out-of-sample di mono_test.
    """

    @pytest.fixture
    def catalog(self) -> CalibrationCatalog:
        calib_root = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration")
        if not calib_root.exists():
            pytest.skip("data/raw/calibration non disponibile")
        return CalibrationCatalog(calib_root)

    def test_monocular_calibration_with_out_of_sample_validation(self, catalog: CalibrationCatalog):
        preprocessor = ImagePreprocessor()
        detector = RobustCheckerboardDetector(CheckerboardConfig(pattern_size=(15, 9), square_size_mm=5.0))
        calibrator = MonocularCalibrator(
            flags=DistortionModelFlags(
                use_rational_model=True,
                use_thin_prism=True,
                use_tilted=False,  # Opzionale, stabilità ottimale con rational + thin_prism
            )
        )

        files = catalog.get_camera_files("wake2", "dataset2", cam_id=1)
        assert len(files) >= 500

        # Raccogliamo frame validi per train (25 frame) e test (10 frame disgiunti)
        train_obj, train_img = [], []
        test_obj, test_img = [], []

        for idx in range(0, 400, 8):
            img = preprocessor.load_image(files[idx], camera_id=1, grayscale=True)
            res = detector.detect(img, camera_id=1)
            if res.is_valid and res.image_points is not None:
                if len(train_obj) < 25:
                    train_obj.append(res.object_points)
                    train_img.append(res.image_points)
                elif len(test_obj) < 10:
                    test_obj.append(res.object_points)
                    test_img.append(res.image_points)
                else:
                    break

        assert len(train_obj) == 25, f"Trovati solo {len(train_obj)} frame per train"
        assert len(test_obj) == 10, f"Trovati solo {len(test_obj)} frame per test"

        img_size = (2176, 1400)

        # 1. Calibrazione su train set
        intrinsics = calibrator.calibrate(
            object_points_list=train_obj,
            image_points_list=train_img,
            image_size=img_size,
            camera_id=1,
        )

        # Verifica errore di riproiezione sub-pixel sul training set (< 0.5 px)
        assert intrinsics.mean_reprojection_error < 0.5, (
            f"Errore di riproiezione train troppo alto: {intrinsics.mean_reprojection_error:.3f} px"
        )
        assert intrinsics.K[0, 0] > 0
        assert intrinsics.dist_coeffs.size >= 8

        # 2. Validazione out-of-sample su test set (frame mai visti prima!)
        test_error, per_frame_errors = calibrator.evaluate_on_test_set(
            object_points_list=test_obj,
            image_points_list=test_img,
            intrinsics=intrinsics,
        )

        # L'errore medio out-of-sample monoculare con solvePnP su pose mai viste deve essere <= 1.2 px
        assert test_error < 1.2, (
            f"Errore di riproiezione out-of-sample troppo alto: {test_error:.3f} px"
        )
        assert len(per_frame_errors) == 10
        print(f"\n[Mono Cam 1 OK] Train error: {intrinsics.mean_reprojection_error:.3f} px | Test error: {test_error:.3f} px")
