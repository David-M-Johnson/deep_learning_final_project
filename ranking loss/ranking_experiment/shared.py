"""Reuse existing dense MASTER, data processing, and evaluation unchanged."""
import importlib.util
import importlib
from pathlib import Path
import sys
SOURCE = Path(__file__).resolve().parents[2] / "top k" / "src"
NAME = "_master_shared"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
shared = importlib.import_module(NAME + ".train")
MASTER = shared.MASTER
DailyData = shared.DailyData
training_labels = shared.training_labels
evaluate = shared.evaluate
checksum = shared.checksum
