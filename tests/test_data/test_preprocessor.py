"""
Test per CameraHardwareConfig e ImagePreprocessor (Modulo 01).
Verifica che l'orientamento delle telecamere (inclusa la rotazione di 180° per Camera 3)
venga normalizzato in modo trasparente e uniforme per tutte le immagini del progetto.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from seeurn.data.hardware import CameraHardwareConfig
from seeurn.data.preprocessor import ImagePreprocessor
from seeurn.data.synchronizer import SynchronizedFrameTriplet
from seeurn.data.streamer import ConditionImageStreamer


class TestCameraHardwareConfig:
    """Test unitari per la configurazione hardware delle telecamere."""

    def test_default_config_rotations(self):
        """Verifica che di default Cam 1 e 2 abbiano 0° e Cam 3 abbia 180°."""
        config = CameraHardwareConfig()
        assert config.camera_rotations == {1: 0, 2: 0, 3: 180}
        assert config.get_rotation_flag(1) is None
        assert config.get_rotation_flag(2) is None
        assert config.get_rotation_flag(3) == cv2.ROTATE_180

    def test_custom_valid_rotations(self):
        """Verifica la corretta mappatura dei codici OpenCV per rotazioni supportate."""
        config = CameraHardwareConfig(camera_rotations={1: 90, 2: 270, 3: 0})
        assert config.get_rotation_flag(1) == cv2.ROTATE_90_CLOCKWISE
        assert config.get_rotation_flag(2) == cv2.ROTATE_90_COUNTERCLOCKWISE
        assert config.get_rotation_flag(3) is None

    def test_invalid_rotation_angle_raises_value_error(self):
        """Verifica che angoli non ortogonali (es. 45°) sollevino ValueError."""
        config = CameraHardwareConfig(camera_rotations={1: 45})
        with pytest.raises(ValueError, match="Angolo di rotazione non supportato: 45"):
            config.get_rotation_flag(1)

    def test_yaml_serialization_roundtrip(self, tmp_path: Path):
        """Verifica il salvataggio e caricamento da file YAML."""
        yaml_file = tmp_path / "test_hardware.yaml"
        original = CameraHardwareConfig(
            camera_rotations={1: 0, 2: 0, 3: 180},
            camera_names={1: "c1", 2: "c2", 3: "c3"}
        )
        original.to_yaml(yaml_file)

        loaded = CameraHardwareConfig.from_yaml(yaml_file)
        assert loaded.camera_rotations == original.camera_rotations
        assert loaded.camera_names == original.camera_names


class TestImagePreprocessor:
    """Test unitari per il preprocessore di immagini."""

    @pytest.fixture
    def preprocessor(self) -> ImagePreprocessor:
        return ImagePreprocessor(CameraHardwareConfig())

    def test_preprocess_array_rotation_cam3(self, preprocessor: ImagePreprocessor):
        """
        Verifica che una matrice asimmetrica venga ruotata di 180° solo per Cam 3.
        Mettiamo un pixel luminoso (255) nell'angolo in alto a sinistra (0, 0).
        Dopo rotazione 180°, il pixel deve trovarsi in basso a destra (-1, -1).
        """
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0, 0] = 255

        # Cam 1: non ruotata
        out_cam1 = preprocessor.preprocess_image(img, camera_id=1)
        assert out_cam1[0, 0] == 255
        assert out_cam1[-1, -1] == 0

        # Cam 2: non ruotata
        out_cam2 = preprocessor.preprocess_image(img, camera_id=2)
        assert out_cam2[0, 0] == 255
        assert out_cam2[-1, -1] == 0

        # Cam 3: ruotata di 180°
        out_cam3 = preprocessor.preprocess_image(img, camera_id=3)
        assert out_cam3[0, 0] == 0
        assert out_cam3[-1, -1] == 255

    def test_load_image_and_triplet(self, tmp_path: Path, preprocessor: ImagePreprocessor):
        """Verifica il caricamento su disco di un'immagine e di una tripletta sincronizzata."""
        cam1_file = tmp_path / "c1.png"
        cam2_file = tmp_path / "c2.png"
        cam3_file = tmp_path / "c3.png"

        # Crea 3 immagini fittizie con valore 255 solo in (0, 0)
        dummy = np.zeros((20, 20), dtype=np.uint8)
        dummy[0, 0] = 255
        cv2.imwrite(str(cam1_file), dummy)
        cv2.imwrite(str(cam2_file), dummy)
        cv2.imwrite(str(cam3_file), dummy)

        triplet = SynchronizedFrameTriplet(
            global_frame_idx=10,
            cam_paths={1: cam1_file, 2: cam2_file, 3: cam3_file}
        )

        loaded_triplet = preprocessor.load_triplet(triplet, grayscale=True)
        assert set(loaded_triplet.keys()) == {1, 2, 3}

        # Cam 1 e 2 mantengono il punto in (0, 0)
        assert loaded_triplet[1][0, 0] == 255
        assert loaded_triplet[2][0, 0] == 255

        # Cam 3 è stata ruotata di 180°: il punto è andato in (-1, -1)
        assert loaded_triplet[3][0, 0] == 0
        assert loaded_triplet[3][-1, -1] == 255

    def test_missing_file_raises_file_not_found(self, preprocessor: ImagePreprocessor, tmp_path: Path):
        """Verifica che un percorso inesistente sollevi FileNotFoundError chiaro."""
        missing = tmp_path / "non_existent.jpg"
        with pytest.raises(FileNotFoundError, match="Impossibile decodificare"):
            preprocessor.load_image(missing, camera_id=1)


class TestStreamerIntegrationWithPreprocessor:
    """Verifica che ConditionImageStreamer utilizzi il preprocessore per raddrizzare Cam 3."""

    def test_streamer_applies_preprocessor(self, tmp_path: Path):
        cam1_file = tmp_path / "c1.png"
        cam2_file = tmp_path / "c2.png"
        cam3_file = tmp_path / "c3.png"

        dummy = np.zeros((10, 10), dtype=np.uint8)
        dummy[0, 0] = 200
        cv2.imwrite(str(cam1_file), dummy)
        cv2.imwrite(str(cam2_file), dummy)
        cv2.imwrite(str(cam3_file), dummy)

        triplet = SynchronizedFrameTriplet(
            global_frame_idx=0,
            cam_paths={1: cam1_file, 2: cam2_file, 3: cam3_file}
        )

        streamer = ConditionImageStreamer([triplet])
        frame = streamer[0]

        # Cam 1 e 2: invariate
        assert frame[1][0, 0] == 200
        assert frame[2][0, 0] == 200

        # Cam 3: ruotata automaticamente di 180°
        assert frame[3][0, 0] == 0
        assert frame[3][-1, -1] == 200


class TestRealDataCam3Rotation:
    """Verifica dell'orientamento su un fotogramma reale di Camera 3."""

    @pytest.mark.skipif(
        not Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration/wake2/cam3/dataset3/data.7vt5sy3k.000100.jpg").exists(),
        reason="Dati reali non presenti sul filesystem locale"
    )
    def test_real_cam3_image_gets_rotated_180(self):
        real_path = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration/wake2/cam3/dataset3/data.7vt5sy3k.000100.jpg")
        preprocessor = ImagePreprocessor()

        # Carica senza preprocessor (grezza)
        raw_img = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE)
        # Carica con preprocessor (raddrizzata)
        proc_img = preprocessor.load_image(real_path, camera_id=3)

        assert raw_img.shape == (800, 1280)
        assert proc_img.shape == (800, 1280)

        # L'immagine pre-processata deve essere l'esatta rotazione 180° dell'originale
        expected_rot = cv2.rotate(raw_img, cv2.ROTATE_180)
        np.testing.assert_array_equal(proc_img, expected_rot)
