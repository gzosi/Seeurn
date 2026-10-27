"""
Unit & Integration tests per seeurn.calibration.target_detector e seeurn.calibration.sampler.
Metodologia: Test-Driven Development (TDD).
"""

import pytest
from pathlib import Path
import numpy as np

from seeurn.calibration.target_detector import (
    CheckerboardConfig,
    DetectionResult,
    RobustCheckerboardDetector,
)
from seeurn.calibration.sampler import (
    CalibrationSplitGroups,
    CalibrationFrameSampler,
)


class TestCheckerboardConfig:
    """Verifica la generazione metrica dei punti 3D della scacchiera 15x9 (5mm)."""

    def test_default_15x9_5mm_geometry(self):
        config = CheckerboardConfig(pattern_size=(15, 9), square_size_mm=5.0)
        assert config.num_corners == 135

        obj_pts = config.generate_object_points()
        assert obj_pts.shape == (135, 3)
        assert obj_pts.dtype == np.float32

        # Punto 0 all'origine (0, 0, 0)
        np.testing.assert_allclose(obj_pts[0], [0.0, 0.0, 0.0])
        # Secondo punto lungo X a (5.0, 0, 0)
        np.testing.assert_allclose(obj_pts[1], [5.0, 0.0, 0.0])
        # Ultimo punto (indice 134 = (14*5, 8*5, 0) = (70.0, 40.0, 0.0))
        np.testing.assert_allclose(obj_pts[-1], [70.0, 40.0, 0.0])


class TestCanonicalOrientation:
    """
    Verifica che l'orientamento dei 135 corner sia canonico (stesso angolo fisico 3D)
    nonostante la simmetria a 180 gradi di una griglia dispari x dispari (15x9).
    """

    def test_upright_camera_canonicalization(self):
        """Per Cam 1 e Cam 2 (bullone in alto), PT_0 deve avere y0 < y_end."""
        detector = RobustCheckerboardDetector()
        # Creiamo punti fittizi ordinati al contrario (dal basso verso l'alto)
        fake_pts = np.zeros((135, 2), dtype=np.float32)
        fake_pts[0] = [500.0, 900.0]    # In basso
        fake_pts[-1] = [200.0, 100.0]   # In alto

        canonical = detector.canonicalize_corners(fake_pts, camera_id=1)
        assert canonical[0, 1] < canonical[-1, 1]
        np.testing.assert_allclose(canonical[0], [200.0, 100.0])

    def test_inverted_camera3_canonicalization(self):
        """Per Cam 3 se non pre-processata (bullone in basso), PT_0 deve avere y0 > y_end."""
        detector = RobustCheckerboardDetector(inverted_cameras=(3,))
        # Punti fittizi con PT_0 in alto (sbagliato per Cam 3 che è capovolta)
        fake_pts = np.zeros((135, 2), dtype=np.float32)
        fake_pts[0] = [200.0, 100.0]    # In alto
        fake_pts[-1] = [500.0, 900.0]   # In basso

        canonical = detector.canonicalize_corners(fake_pts, camera_id=3)
        assert canonical[0, 1] > canonical[-1, 1]
        np.testing.assert_allclose(canonical[0], [500.0, 900.0])


class TestCalibrationFrameSampler:
    """
    Verifica il campionamento casuale distanziato (no frame troppo vicini)
    e la partizione in gruppi totalmente disgiunti per ogni fase di calibrazione e test.
    """

    def test_min_frame_spacing_enforced(self):
        """Nessuna coppia di frame campionati deve essere più vicina di min_spacing."""
        available_indices = list(range(2, 1000))
        sampler = CalibrationFrameSampler(min_frame_spacing=12, seed=42)

        sampled = sampler.sample_spaced_indices(available_indices, count=40)
        assert len(sampled) == 40

        sorted_sampled = sorted(sampled)
        diffs = np.diff(sorted_sampled)
        assert np.all(diffs >= 12), f"Trovati frame troppo vicini! Distanze minime: {diffs.min()}"

    def test_disjoint_calibration_and_test_groups(self):
        """
        Verifica che i gruppi per mono_train, mono_test, stereo_train, stereo_test,
        ba_train e final_test siano 100% disgiunti (nessun data leakage).
        """
        available_indices = list(range(2, 1000))
        sampler = CalibrationFrameSampler(min_frame_spacing=10, seed=123)

        groups: CalibrationSplitGroups = sampler.create_disjoint_groups(
            available_indices,
            mono_train_count=15,
            mono_test_count=10,
            stereo_train_count=15,
            stereo_test_count=10,
            ba_train_count=15,
            final_test_count=10,
        )

        all_sets = [
            set(groups.mono_train),
            set(groups.mono_test),
            set(groups.stereo_train),
            set(groups.stereo_test),
            set(groups.ba_train),
            set(groups.final_test),
        ]

        # 1. Verifica le cardinalità richieste
        assert len(groups.mono_train) == 15
        assert len(groups.mono_test) == 10
        assert len(groups.stereo_train) == 15
        assert len(groups.stereo_test) == 10
        assert len(groups.ba_train) == 15
        assert len(groups.final_test) == 10

        # 2. Verifica che l'intersezione tra QUALUNQUE coppia di gruppi sia vuota!
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                intersection = all_sets[i] & all_sets[j]
                assert len(intersection) == 0, f"Data leakage tra gruppo {i} e {j}: {intersection}"

        # 3. Verifica che anche unendo tutti i gruppi, i frame rispettino min_frame_spacing
        union_all = sorted(list(set.union(*all_sets)))
        assert len(union_all) == 75
        assert np.all(np.diff(union_all) >= 10)


class TestRealCalibrationDetection:
    """Test di integrazione sulle immagini reali di wake2/dataset3 (15x9, 5mm)."""

    RAW_DS3 = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration/wake2")

    @pytest.mark.skipif(not RAW_DS3.exists(), reason="Dati reali non presenti")
    def test_detect_15x9_on_all_three_cameras(self):
        detector = RobustCheckerboardDetector(
            CheckerboardConfig(pattern_size=(15, 9), square_size_mm=5.0)
        )

        files = {
            1: self.RAW_DS3 / "cam1" / "dataset3" / "data.7vt5tyg1.000002.jpg",
            2: self.RAW_DS3 / "cam2" / "dataset3" / "data.7vt5tzwd.000002.jpg",
            3: self.RAW_DS3 / "cam3" / "dataset3" / "data.7vt5sy3k.000002.jpg",
        }

        for cam_id, fpath in files.items():
            res = detector.detect_from_path(fpath, camera_id=cam_id)
            assert res.is_valid is True, f"Rilevamento fallito su Cam {cam_id}"
            assert res.image_points.shape == (135, 2)
            assert res.object_points.shape == (135, 3)

            # Tutte le telecamere hanno l'origine PT_0 in alto in modo coerente nello spazio di laboratorio
            assert res.image_points[0, 1] < res.image_points[-1, 1]
