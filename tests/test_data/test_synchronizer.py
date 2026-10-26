"""
Unit tests per il modulo seeurn.data.synchronizer.
Metodologia: Test-Driven Development (TDD).
"""

import pytest
from pathlib import Path
from seeurn.data.synchronizer import FrameSynchronizer, SynchronizedFrameTriplet


class TestFilenameParsing:
    """Verifica l'estrazione corretta dell'indice temporale dai nomi dei file."""

    def test_extract_index_standard_dataset(self):
        """Testa il formato reale del dataset: data.7vt5eh1g.000042.jpg -> 42."""
        syncer = FrameSynchronizer()
        path = Path("data.7vt5eh1g.000042.jpg")
        assert syncer.extract_index(path) == 42

    def test_extract_index_custom_prefix(self):
        """Testa formati con prefissi differenti, es. cam1_wake2_000105.jpg -> 105."""
        syncer = FrameSynchronizer()
        path = Path("cam1_wake2_000105.jpg")
        assert syncer.extract_index(path) == 105

    def test_extract_index_invalid_format_raises_error(self):
        """Verifica che un nome senza numero progressivo sollevi ValueError."""
        syncer = FrameSynchronizer()
        path = Path("corrupted_filename_without_number.jpg")
        with pytest.raises(ValueError, match="Impossibile estrarre"):
            syncer.extract_index(path)


class TestTemporalAlignment:
    """Verifica l'allineamento con shift hardware della Telecamera 3."""

    def test_alignment_with_cam3_offset_2(self):
        """
        Scenario reale:
        Cam 1 e Cam 2 partono da 0 (frame 0, 1, 2, 3, 4).
        Cam 3 parte da 2 (frame 2, 3, 4) a causa del ritardo hardware.
        Con offset=2:
          - Il frame 2 di Cam 3 corrisponde al frame 2 di Cam 1 e Cam 2.
          - I frame 0 e 1 di Cam 1 e 2 vengono scartati (orfani).
        """
        cam1_files = [Path(f"cam1.{i:06d}.jpg") for i in range(5)]
        cam2_files = [Path(f"cam2.{i:06d}.jpg") for i in range(5)]
        cam3_files = [Path(f"cam3.{i:06d}.jpg") for i in range(2, 5)]

        syncer = FrameSynchronizer(cam3_offset=2)
        triplets = syncer.align(cam1_files, cam2_files, cam3_files)

        assert len(triplets) == 3

        # Prima tripletta: t = 2 per tutte le camere
        t0 = triplets[0]
        assert t0.global_frame_idx == 2
        assert t0.get_path(1) == Path("cam1.000002.jpg")
        assert t0.get_path(2) == Path("cam2.000002.jpg")
        assert t0.get_path(3) == Path("cam3.000002.jpg")
        assert t0.has_all_three_views is True

    def test_dropped_frame_handling(self):
        """Verifica che se una camera perde un frame a metà sequenza, la tripletta sia scartata o parziale."""
        cam1_files = [Path(f"c1.{i:06d}.jpg") for i in [2, 3, 4]]
        cam2_files = [Path(f"c2.{i:06d}.jpg") for i in [2, 4]]      # Frame 3 perso su cam2!
        cam3_files = [Path(f"c3.{i:06d}.jpg") for i in [2, 3, 4]]

        syncer = FrameSynchronizer(cam3_offset=2)
        triplets = syncer.align(cam1_files, cam2_files, cam3_files, require_all_views=True)

        # Solo i frame 2 e 4 hanno tutte e 3 le viste
        assert len(triplets) == 2
        indices = [t.global_frame_idx for t in triplets]
        assert indices == [2, 4]


class TestMissingFramesDetection:
    """Verifica l'identificazione di buchi temporali in una sequenza."""

    def test_detect_missing_frames(self):
        syncer = FrameSynchronizer()
        files = [
            Path("img.000010.jpg"),
            Path("img.000011.jpg"),
            Path("img.000013.jpg"), # Manca il 12!
            Path("img.000015.jpg"), # Manca il 14!
        ]
        missing = syncer.find_missing_frames(files)
        assert missing == [12, 14]

