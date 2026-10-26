"""
Unit test per seeurn.data.catalog e seeurn.data.streamer.
Metodologia: Test-Driven Development (TDD).
"""

import pytest
from pathlib import Path
import numpy as np
import cv2

from seeurn.data.catalog import ExperimentCondition, DatasetCatalog
from seeurn.data.streamer import ConditionImageStreamer
from seeurn.data.synchronizer import SynchronizedFrameTriplet


class TestExperimentCondition:
    """Verifica la dataclass immutabile ExperimentCondition."""

    def test_condition_creation_and_properties(self):
        cond = ExperimentCondition(wake="wake2", kt="kt1", sigma="sigma3")
        assert cond.wake == "wake2"
        assert cond.kt == "kt1"
        assert cond.sigma == "sigma3"
        assert cond.condition_id == "wake2_kt1_sigma3"

    def test_immutability(self):
        cond = ExperimentCondition(wake="wake2", kt="kt1", sigma="sigma3")
        with pytest.raises(Exception):
            cond.wake = "wake4"  # Non deve poter essere modificato (frozen)


class TestDatasetCatalogMock:
    """Testa la scoperta delle cartelle usando una struttura temporanea creata al volo."""

    def test_discover_conditions_in_mock_dir(self, temp_dataset_dir):
        # Creiamo una gerarchia fittizia: wake2/kt1/sigma1/{cam1, cam2, cam3}
        cond_dir = temp_dataset_dir / "wake2" / "kt1" / "sigma1"
        for cam in ["cam1", "cam2", "cam3"]:
            cam_dir = cond_dir / cam
            cam_dir.mkdir(parents=True)
            # Creiamo 3 file fittizi per ciascuna camera
            for i in range(3):
                (cam_dir / f"data.test.{i:06d}.jpg").write_text("dummy")

        catalog = DatasetCatalog(root_dir=temp_dataset_dir)
        conditions = catalog.discover_conditions()

        assert len(conditions) == 1
        assert conditions[0] == ExperimentCondition(wake="wake2", kt="kt1", sigma="sigma1")

        # Verifica recupero file per camera
        cam1_files = catalog.get_camera_files(conditions[0], cam_id=1)
        assert len(cam1_files) == 3


class TestConditionImageStreamer:
    """Testa il lazy loading di immagini con array NumPy sintetici."""

    def test_streaming_lazy_loading(self, temp_dataset_dir):
        # Creiamo immagini di test sintetiche 100x100 su disco
        triplets = []
        for frame_idx in range(2, 5):
            paths = {}
            for cam_id in [1, 2, 3]:
                p = temp_dataset_dir / f"cam{cam_id}.{frame_idx:06d}.jpg"
                dummy_img = np.full((100, 100), fill_value=frame_idx * 50, dtype=np.uint8)
                cv2.imwrite(str(p), dummy_img)
                paths[cam_id] = p

            triplets.append(SynchronizedFrameTriplet(global_frame_idx=frame_idx, cam_paths=paths))

        streamer = ConditionImageStreamer(triplets)
        assert len(streamer) == 3

        # Test caricamento pigro tramite indicizzazione
        frame_dict = streamer[0]
        assert set(frame_dict.keys()) == {1, 2, 3}
        assert frame_dict[1].shape == (100, 100)
        assert frame_dict[1].dtype == np.uint8

        # Test iteratore (yield on-demand)
        collected_frames = 0
        for frames in streamer:
            assert 1 in frames and 2 in frames and 3 in frames
            assert frames[1].shape == (100, 100)
            collected_frames += 1

        assert collected_frames == 3


class TestRealDataCatalog:
    """Test di integrazione sul catalogo reale di data/raw/runs."""

    REAL_RUNS_DIR = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/runs")

    @pytest.mark.skipif(not REAL_RUNS_DIR.exists(), reason="Dati reali non presenti")
    def test_discover_real_runs(self):
        catalog = DatasetCatalog(root_dir=self.REAL_RUNS_DIR)
        conditions = catalog.discover_conditions()

        assert len(conditions) > 0
        # Verifichiamo che wake2 e wake4 siano presenti
        wakes = {c.wake for c in conditions}
        assert "wake2" in wakes or "wake4" in wakes

        # Testiamo lo streaming del primo fotogramma della prima condizione
        first_cond = conditions[0]
        triplets = catalog.get_synchronized_triplets(first_cond, max_frames=5)
        assert len(triplets) > 0

        streamer = ConditionImageStreamer(triplets)
        first_frame = streamer[0]
        assert set(first_frame.keys()) == {1, 2, 3}
        # Verifica che sia un'immagine valida (altezza > 0, larghezza > 0)
        h, w = first_frame[1].shape
        assert h > 100 and w > 100
