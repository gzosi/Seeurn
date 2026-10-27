"""
Campionatore intelligente e disgiunto per i dataset di calibrazione.

Risolve due problemi fondamentali nei video ad alta velocità:
1. Evita di selezionare frame temporalmente troppo vicini (che avrebbero pose quasi identiche).
2. Suddivide i frame campionati in gruppi totalmente disgiunti (Zero Data Leakage)
   per:
   - Calibrazione Monoculare (mono_train) e relativo Test (mono_test)
   - Calibrazione Stereo (stereo_train) e relativo Test (stereo_test)
   - Global Bundle Adjustment (ba_train) e Validazione Finale Holdout (final_test)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class CalibrationSplitGroups:
    """
    Rappresenta la partizione in 6 gruppi mutuamente esclusivi di indici di frame.
    Nessun frame può appartenere a più di un gruppo contemporaneamente.
    """

    mono_train: List[int]
    mono_test: List[int]
    stereo_train: List[int]
    stereo_test: List[int]
    ba_train: List[int]
    final_test: List[int]

    @property
    def total_frames(self) -> int:
        return (
            len(self.mono_train)
            + len(self.mono_test)
            + len(self.stereo_train)
            + len(self.stereo_test)
            + len(self.ba_train)
            + len(self.final_test)
        )


class CalibrationFrameSampler:
    """
    Campiona frame casuali garantendo una distanza temporale minima (min_frame_spacing)
    e li distribuisce in gruppi indipendenti per le diverse fasi di calibrazione e test.
    """

    def __init__(self, min_frame_spacing: int = 10, seed: Optional[int] = 42):
        """
        Args:
            min_frame_spacing: Distanza minima (in numero di frame) tra due scatti qualsiasi.
            seed: Seme per il generatore pseudocasuale (garantisce riproducibilità scientifica).
        """
        if min_frame_spacing < 1:
            raise ValueError("min_frame_spacing deve essere >= 1")
        self.min_frame_spacing = min_frame_spacing
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample_spaced_indices(self, available_indices: Sequence[int], count: int) -> List[int]:
        """
        Estrae 'count' indici casuali da 'available_indices' garantendo che
        |idx_i - idx_j| >= min_frame_spacing per ogni coppia (i, j).

        Utilizza la biiezione combinatoria di Kaplansky su posizioni ordinate,
        seguita da verifica sul valore effettivo degli indici.
        """
        sorted_avail = sorted(set(available_indices))
        n_avail = len(sorted_avail)

        if count <= 0:
            return []

        d = self.min_frame_spacing
        max_reduced_pool = n_avail - (count - 1) * (d - 1)
        if max_reduced_pool < count:
            raise ValueError(
                f"Impossibile estrarre {count} frame con distanza minima {d} "
                f"da un pool di {n_avail} frame disponibili."
            )

        # Estrazione uniforme senza reinserimento nello spazio compresso
        compressed_picks = np.sort(
            self._rng.choice(max_reduced_pool, size=count, replace=False)
        )
        # Espansione con distanza minima garantita d sugli indici di posizione
        spaced_positions = compressed_picks + np.arange(count) * (d - 1)
        selected_indices = [sorted_avail[pos] for pos in spaced_positions]

        # Rimescoliamo l'ordine di uscita così che una successiva divisione in fette
        # produca gruppi distribuiti uniformemente lungo tutta la durata del video
        shuffled = list(self._rng.permutation(selected_indices))
        return [int(x) for x in shuffled]

    def create_disjoint_groups(
        self,
        available_indices: Sequence[int],
        mono_train_count: int = 20,
        mono_test_count: int = 10,
        stereo_train_count: int = 20,
        stereo_test_count: int = 10,
        ba_train_count: int = 20,
        final_test_count: int = 10,
    ) -> CalibrationSplitGroups:
        """
        Campiona e partiziona i frame in 6 gruppi completamente disgiunti
        e distribuiti uniformemente su tutta la sequenza temporale.
        """
        total_needed = (
            mono_train_count
            + mono_test_count
            + stereo_train_count
            + stereo_test_count
            + ba_train_count
            + final_test_count
        )

        pool = self.sample_spaced_indices(available_indices, count=total_needed)

        # Tagliamo il pool rimescolato nei 6 gruppi disgiunti e li ordiniamo internamente
        c1 = mono_train_count
        c2 = c1 + mono_test_count
        c3 = c2 + stereo_train_count
        c4 = c3 + stereo_test_count
        c5 = c4 + ba_train_count

        return CalibrationSplitGroups(
            mono_train=sorted(pool[0:c1]),
            mono_test=sorted(pool[c1:c2]),
            stereo_train=sorted(pool[c2:c3]),
            stereo_test=sorted(pool[c3:c4]),
            ba_train=sorted(pool[c4:c5]),
            final_test=sorted(pool[c5:total_needed]),
        )
