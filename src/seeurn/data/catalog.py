"""
Catalogo e indicizzazione delle condizioni sperimentali del dataset Seeurn.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import re

from seeurn.data.synchronizer import FrameSynchronizer, SynchronizedFrameTriplet
from seeurn.data.streamer import ConditionImageStreamer
from seeurn.data.preprocessor import ImagePreprocessor


@dataclass(frozen=True)
class ExperimentCondition:
    """
    Rappresenta una condizione fisica di prova nel tunnel di cavitazione.

    Attributi:
        wake: Identificativo della scia (es. "wake2").
        kt: Identificativo del coefficiente di spinta (es. "kt1").
        sigma: Identificativo del numero di cavitazione (es. "sigma1").
    """

    wake: str
    kt: str
    sigma: str

    @property
    def condition_id(self) -> str:
        """Restituisce un ID univoco standardizzato per la condizione."""
        return f"{self.wake}_{self.kt}_{self.sigma}"


class DatasetCatalog:
    """
    Scandaglia e indicizza il filesystem del dataset senza caricare immagini in memoria.

    Permette di scoprire automaticamente tutte le combinazioni di scia, carico e cavitazione,
    e di recuperare le sequenze di file allineate tramite FrameSynchronizer.
    """

    def __init__(self, root_dir: Path, synchronizer: Optional[FrameSynchronizer] = None):
        """
        Inizializza il catalogo.

        Args:
            root_dir: Cartella radice dei dati (es. 'data/raw/runs').
            synchronizer: Istanza opzionale di FrameSynchronizer (default con cam3_offset=2).
        """
        self.root_dir = Path(root_dir)
        self.synchronizer = synchronizer or FrameSynchronizer(cam3_offset=2)

    def discover_conditions(self) -> List[ExperimentCondition]:
        """
        Scandaglia l'albero delle directory (wake/kt/sigma) per trovare tutte le condizioni valide.

        Una condizione è considerata valida se contiene le cartelle cam1, cam2 e cam3.

        Returns:
            Lista ordinata di ExperimentCondition.
        """
        if not self.root_dir.exists():
            return []

        conditions: List[ExperimentCondition] = []

        # Cerchiamo cartelle al terzo livello: wake*/kt*/sigma*
        for wake_dir in sorted(self.root_dir.glob("wake*")):
            if not wake_dir.is_dir():
                continue
            for kt_dir in sorted(wake_dir.glob("kt*")):
                if not kt_dir.is_dir():
                    continue
                for sigma_dir in sorted(kt_dir.glob("sigma*")):
                    if not sigma_dir.is_dir():
                        continue

                    # Verifica che esistano almeno cam1, cam2 o cam3
                    cam_subdirs = [p.name for p in sigma_dir.iterdir() if p.is_dir()]
                    if "cam1" in cam_subdirs and "cam2" in cam_subdirs:
                        conditions.append(
                            ExperimentCondition(
                                wake=wake_dir.name,
                                kt=kt_dir.name,
                                sigma=sigma_dir.name,
                            )
                        )

        return conditions

    def get_condition_dir(self, condition: ExperimentCondition) -> Path:
        """Restituisce il percorso fisico su disco per una data condizione."""
        return self.root_dir / condition.wake / condition.kt / condition.sigma

    def get_camera_files(self, condition: ExperimentCondition, cam_id: int) -> List[Path]:
        """
        Restituisce la lista ordinata dei percorsi file per una specifica telecamera.

        Args:
            condition: La condizione sperimentale di interesse.
            cam_id: L'identificativo della telecamera (1, 2 o 3).

        Returns:
            Lista ordinata di Path.
        """
        cam_dir = self.get_condition_dir(condition) / f"cam{cam_id}"
        if not cam_dir.exists():
            return []

        # Supporta sia .jpg che .jpeg che .png
        files = [p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        return sorted(files)

    def get_synchronized_triplets(
        self,
        condition: ExperimentCondition,
        max_frames: Optional[int] = None,
        require_all_views: bool = True,
    ) -> List[SynchronizedFrameTriplet]:
        """
        Recupera e allinea temporalmente i frame delle 3 telecamere per una condizione.

        Args:
            condition: La condizione di interesse.
            max_frames: Limite massimo opzionale di frame da allineare (utile per test rapidi).
            require_all_views: Se True, include solo le triplette complete.

        Returns:
            Lista di SynchronizedFrameTriplet allineate.
        """
        cam1_files = self.get_camera_files(condition, 1)
        cam2_files = self.get_camera_files(condition, 2)
        cam3_files = self.get_camera_files(condition, 3)

        if max_frames is not None:
            # Prendiamo un piccolo margine in più per gestire l'offset iniziale di Cam 3
            margin = max_frames + self.synchronizer.cam3_offset + 5
            cam1_files = cam1_files[:margin]
            cam2_files = cam2_files[:margin]
            cam3_files = cam3_files[:margin]

        triplets = self.synchronizer.align(
            cam1_files, cam2_files, cam3_files, require_all_views=require_all_views
        )

        if max_frames is not None:
            triplets = triplets[:max_frames]

        return triplets


@dataclass(frozen=True)
class CalibrationDatasetInfo:
    """
    Rappresenta una sessione di calibrazione memorizzata su disco.

    Attributi:
        wake: Nome della scia associata (es. 'wake2', 'wake4').
        dataset_name: Nome del dataset (es. 'dataset1', 'dataset2', 'dataset3', 'dataset4').
        available_cameras: Tupla delle telecamere presenti (es. (1, 2, 3)).
    """

    wake: str
    dataset_name: str
    available_cameras: Tuple[int, ...] = (1, 2, 3)

    @property
    def identifier(self) -> str:
        """Restituisce un ID univoco per il dataset di calibrazione."""
        return f"{self.wake}_{self.dataset_name}"


class CalibrationCatalog:
    """
    Scandaglia e indicizza i dataset di calibrazione su disco (data/raw/calibration/).
    Fornisce triplette di frame sincronizzate e streamer pronti all'uso con orientamento normalizzato.
    """

    def __init__(
        self,
        calibration_root: Union[str, Path],
        synchronizer: Optional[FrameSynchronizer] = None,
    ):
        """
        Inizializza il catalogo di calibrazione.

        Args:
            calibration_root: Cartella radice dei dati di calibrazione.
            synchronizer: Sincronizzatore temporale (default con offset=2 per Cam 3).
        """
        self.root_dir = Path(calibration_root)
        self.synchronizer = synchronizer or FrameSynchronizer(cam3_offset=2)

    def discover_datasets(self) -> List[CalibrationDatasetInfo]:
        """
        Scandaglia l'albero delle cartelle per trovare tutti i dataset di calibrazione disponibili.

        Returns:
            Lista ordinata di CalibrationDatasetInfo.
        """
        if not self.root_dir.exists():
            return []

        found_map: dict[Tuple[str, str], List[int]] = {}

        for wake_dir in sorted(self.root_dir.iterdir()):
            if not wake_dir.is_dir() or not wake_dir.name.startswith("wake"):
                continue

            wake_name = wake_dir.name
            for cam_id in (1, 2, 3):
                cam_dir = wake_dir / f"cam{cam_id}"
                if not cam_dir.exists():
                    continue

                for ds_dir in sorted(cam_dir.iterdir()):
                    if ds_dir.is_dir() and ds_dir.name.startswith("dataset"):
                        key = (wake_name, ds_dir.name)
                        if key not in found_map:
                            found_map[key] = []
                        found_map[key].append(cam_id)

        results: List[CalibrationDatasetInfo] = []
        for (wake, ds_name), cams in sorted(found_map.items()):
            results.append(
                CalibrationDatasetInfo(
                    wake=wake,
                    dataset_name=ds_name,
                    available_cameras=tuple(sorted(cams)),
                )
            )

        return results

    def get_camera_files(self, wake: str, dataset_name: str, cam_id: int) -> List[Path]:
        """
        Restituisce la lista ordinata dei percorsi delle immagini per una specifica telecamera.

        Args:
            wake: Nome della scia (es. 'wake2').
            dataset_name: Nome del dataset (es. 'dataset3').
            cam_id: ID della telecamera (1, 2 o 3).

        Returns:
            Lista ordinata di Path.
        """
        cam_dir = self.root_dir / wake / f"cam{cam_id}" / dataset_name
        if not cam_dir.exists():
            return []

        files = [p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        return sorted(files)

    def get_synchronized_triplets(
        self,
        wake: str,
        dataset_name: str,
        max_frames: Optional[int] = None,
        require_all_views: bool = True,
    ) -> List[SynchronizedFrameTriplet]:
        """
        Recupera e sincronizza le triplette temporali per il dataset specificato.

        Args:
            wake: Nome della scia (es. 'wake2').
            dataset_name: Nome del dataset (es. 'dataset3').
            max_frames: Limite opzionale di frame da allineare.
            require_all_views: Se True, include solo le triplette complete con tutte e 3 le camere.

        Returns:
            Lista di SynchronizedFrameTriplet allineate.
        """
        cam1_files = self.get_camera_files(wake, dataset_name, 1)
        cam2_files = self.get_camera_files(wake, dataset_name, 2)
        cam3_files = self.get_camera_files(wake, dataset_name, 3)

        if max_frames is not None:
            margin = max_frames + self.synchronizer.cam3_offset + 5
            cam1_files = cam1_files[:margin]
            cam2_files = cam2_files[:margin]
            cam3_files = cam3_files[:margin]

        triplets = self.synchronizer.align(
            cam1_files, cam2_files, cam3_files, require_all_views=require_all_views
        )

        if max_frames is not None:
            triplets = triplets[:max_frames]

        return triplets

    def get_image_streamer(
        self,
        wake: str,
        dataset_name: str,
        max_frames: Optional[int] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        grayscale: bool = True,
    ) -> ConditionImageStreamer:
        """
        Restituisce un ConditionImageStreamer per il dataset di calibrazione specificato,
        con normalizzazione automatica dell'orientamento hardware.

        Args:
            wake: Nome della scia (es. 'wake2').
            dataset_name: Nome del dataset (es. 'dataset3').
            max_frames: Limite opzionale di frame da allineare.
            preprocessor: Preprocessore immagini opzionale (default: Cam 3 a 180°).
            grayscale: Se True carica a 1 canale (uint8), se False carica BGR.

        Returns:
            Istanza di ConditionImageStreamer pronta all'uso.
        """
        triplets = self.get_synchronized_triplets(
            wake, dataset_name, max_frames=max_frames, require_all_views=True
        )
        return ConditionImageStreamer(
            triplets=triplets,
            grayscale=grayscale,
            preprocessor=preprocessor,
        )


