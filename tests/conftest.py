"""
Configurazione globale e fixture per la suite di test pytest di Seeurn.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dataset_dir():
    """Crea una cartella temporanea con una gerarchia fittizia di immagini."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
