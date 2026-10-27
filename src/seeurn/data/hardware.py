"""
Configurazione dell'hardware delle telecamere del tunnel idrodinamico.
Incapsula le proprietà fisiche note del montaggio sperimentale, inclusi gli angoli di rotazione.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union
import cv2
import yaml


@dataclass(frozen=True)
class CameraHardwareConfig:
    """
    Rappresenta le specifiche fisiche e la disposizione spaziale delle telecamere nel tunnel.

    Attributi:
        camera_rotations: Mappa {camera_id: gradi_di_rotazione} (default: {1: 0, 2: 0, 3: 180}).
                          Compensa il fatto che la Camera 3 sia montata fisicamente capovolta.
        camera_names: Mappa {camera_id: etichetta_descrittiva}.
    """

    camera_rotations: Dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0, 3: 180}
    )
    camera_names: Dict[int, str] = field(
        default_factory=lambda: {
            1: "cam1_front",
            2: "cam2_oblique",
            3: "cam3_lateral_inverted",
        }
    )

    def get_rotation_flag(self, camera_id: int) -> Optional[int]:
        """
        Restituisce la costante cv2.ROTATE_* corrispondente all'angolo configurato per la telecamera.

        Args:
            camera_id: Identificativo della telecamera (es. 1, 2, 3).

        Returns:
            Costante intera OpenCV per cv2.rotate() oppure None se l'angolo è 0°.

        Raises:
            ValueError: Se l'angolo di rotazione non è tra i valori supportati (0, 90, 180, 270).
        """
        deg = self.camera_rotations.get(camera_id, 0)

        if deg == 0:
            return None
        elif deg == 180:
            return cv2.ROTATE_180
        elif deg == 90:
            return cv2.ROTATE_90_CLOCKWISE
        elif deg == 270:
            return cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            raise ValueError(
                f"Angolo di rotazione non supportato: {deg}. Valori ammessi: 0, 90, 180, 270."
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CameraHardwareConfig":
        """Carica la configurazione da un file YAML."""
        yaml_path = Path(path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_rotations = data.get("camera_rotations", {})
        # Assicura chiavi intere (YAML può parsare interi come chiavi o stringhe)
        rotations = {int(k): int(v) for k, v in raw_rotations.items()}

        raw_names = data.get("camera_names", {})
        names = {int(k): str(v) for k, v in raw_names.items()}

        return cls(
            camera_rotations=rotations or {1: 0, 2: 0, 3: 180},
            camera_names=names or {1: "cam1", 2: "cam2", 3: "cam3"},
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Salva la configurazione su un file YAML."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_rotations": self.camera_rotations,
            "camera_names": self.camera_names,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

