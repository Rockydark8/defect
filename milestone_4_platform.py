import importlib.util
from pathlib import Path

_module_path = Path(__file__).with_name("bug_platform.py")
_spec = importlib.util.spec_from_file_location("_local_bug_platform", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load local module from {_module_path}")

_local_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_module)

InMemoryVectorStore = _local_module.InMemoryVectorStore
IntelligentBugPlatform = _local_module.IntelligentBugPlatform
HistoricalDefect = _local_module.HistoricalDefect
SubmittedBug = _local_module.SubmittedBug

__all__ = [
    "InMemoryVectorStore",
    "IntelligentBugPlatform",
    "HistoricalDefect",
    "SubmittedBug",
]
