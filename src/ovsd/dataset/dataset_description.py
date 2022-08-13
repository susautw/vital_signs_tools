from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any


@dataclass
class DatasetDescription:
    version: Sequence[int]
    root: Path
    name: str
    desc: str
    attrs: dict[str, Any]
