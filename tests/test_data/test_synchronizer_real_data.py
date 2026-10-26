"""
Test di integrazione sui dati sperimentali reali in data/raw.
Verifica che FrameSynchronizer funzioni direttamente sui file del tunnel.
"""

import pytest
from pathlib import Path
from seeurn.data.synchronizer import FrameSynchronizer


RAW_CALIBRATION_DIR = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration/wake2")


@pytest.mark.skipif(
    not RAW_CALIBRATION_DIR.exists(),
    reason="Cartella dati reali non presente o percorso non accessibile."
)
class TestRealDataSynchronization:
    """Verifica l'allineamento sui veri file di calibrazione di wake2."""

    def test_align_wake2_dataset2(self):
        cam1_dir = RAW_CALIBRATION_DIR / "cam1" / "dataset2"
        cam2_dir = RAW_CALIBRATION_DIR / "cam2" / "dataset2"
        cam3_dir = RAW_CALIBRATION_DIR / "cam3" / "dataset2"

        assert cam1_dir.exists(), f"Cartella non trovata: {cam1_dir}"
        assert cam2_dir.exists(), f"Cartella non trovata: {cam2_dir}"
        assert cam3_dir.exists(), f"Cartella non trovata: {cam3_dir}"

        cam1_files = sorted(cam1_dir.glob("*.jpg"))
        cam2_files = sorted(cam2_dir.glob("*.jpg"))
        cam3_files = sorted(cam3_dir.glob("*.jpg"))

        assert len(cam1_files) > 0
        assert len(cam2_files) > 0
        assert len(cam3_files) > 0

        syncer = FrameSynchronizer(cam3_offset=2)

        # Verifica che Cam 3 inizi effettivamente dall'indice 2
        assert syncer.extract_index(cam3_files[0]) == 2
        # Cam 1 e 2 iniziano dall'indice 0
        assert syncer.extract_index(cam1_files[0]) == 0
        assert syncer.extract_index(cam2_files[0]) == 0

        # Esegue l'allineamento
        triplets = syncer.align(cam1_files, cam2_files, cam3_files, require_all_views=True)

        assert len(triplets) > 0
        # La prima tripletta sincronizzata deve avere indice 2
        first = triplets[0]
        assert first.global_frame_idx == 2
        assert first.has_all_three_views is True
        assert first.get_path(1).name.endswith("000002.jpg")
        assert first.get_path(2).name.endswith("000002.jpg")
        assert first.get_path(3).name.endswith("000002.jpg")

        # Verifica l'assenza di frame persi a metà sequenza su Cam 3
        missing_cam3 = syncer.find_missing_frames(cam3_files)
        assert len(missing_cam3) == 0, f"Frame persi rilevati su Cam 3: {missing_cam3}"

