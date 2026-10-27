"""
Test per CalibrationCatalog (Modulo 01).
Verifica che i dataset di calibrazione vengano scoperti e indicizzati correttamente,
con allineamento temporale tramite FrameSynchronizer e integrazione con ImagePreprocessor.
"""

from pathlib import Path
import pytest

from seeurn.data.catalog import CalibrationDatasetInfo, CalibrationCatalog
from seeurn.data.streamer import ConditionImageStreamer


class TestCalibrationCatalogUnit:
    """Test unitari per CalibrationCatalog su filesystem simulato (tmp_path)."""

    def test_discover_datasets_synthetic(self, tmp_path: Path):
        # Crea struttura: tmp_path/wake2/cam1/dataset1, tmp_path/wake2/cam2/dataset1, etc.
        for wake in ("wake2", "wake4"):
            for cam in ("cam1", "cam2", "cam3"):
                d_path = tmp_path / wake / cam / "dataset2"
                d_path.mkdir(parents=True, exist_ok=True)
                for f_idx in range(5):
                    (d_path / f"data.test.{f_idx:06d}.jpg").write_text("dummy")

        catalog = CalibrationCatalog(tmp_path)
        datasets = catalog.discover_datasets()

        assert len(datasets) == 2
        wake_names = {d.wake for d in datasets}
        assert wake_names == {"wake2", "wake4"}
        for d in datasets:
            assert d.dataset_name == "dataset2"


class TestCalibrationCatalogRealData:
    """Test su dati reali di calibrazione in data/raw/calibration."""

    @pytest.fixture
    def real_catalog(self) -> CalibrationCatalog:
        calib_dir = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration")
        if not calib_dir.exists():
            pytest.skip("Directory data/raw/calibration non disponibile")
        return CalibrationCatalog(calib_dir)

    def test_discover_real_calibration_datasets(self, real_catalog: CalibrationCatalog):
        datasets = real_catalog.discover_datasets()
        assert len(datasets) > 0
        wakes = {d.wake for d in datasets}
        assert "wake2" in wakes

    def test_get_synchronized_triplets_real(self, real_catalog: CalibrationCatalog):
        # dataset3 in wake2 ha tutte e 3 le telecamere
        triplets = real_catalog.get_synchronized_triplets("wake2", "dataset3", max_frames=10)
        assert len(triplets) == 10
        for t in triplets:
            assert t.has_all_three_views
            assert set(t.cam_paths.keys()) == {1, 2, 3}

    def test_real_calibration_streamer(self, real_catalog: CalibrationCatalog):
        streamer = real_catalog.get_image_streamer("wake2", "dataset3", max_frames=2)
        assert len(streamer) == 2
        first_frame = streamer[0]
        assert set(first_frame.keys()) == {1, 2, 3}
        # Cam 1 e Cam 2: sensore alta risoluzione (1400, 2176)
        # Cam 3: sensore laterale (800, 1280)
        assert first_frame[1].shape == (1400, 2176)
        assert first_frame[2].shape == (1400, 2176)
        assert first_frame[3].shape == (800, 1280)
