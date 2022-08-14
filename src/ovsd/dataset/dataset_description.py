import yaml
from fancy import config as cfg
from pathlib import Path
from typing import Any, Optional


def process_2_tuple(x):
    if len(x) != 2:
        raise ValueError(f"length is not 2: {x}")
    return tuple(x)


class DatasetDescription(cfg.BaseConfig):
    version = cfg.Option(required=True, type=process_2_tuple)
    root: Path = cfg.Option(required=True, type=Path)
    name: str = cfg.Option(required=True, type=str)
    desc: Optional[str] = cfg.Option(nullable=True, type=str)
    attrs: dict[str, Any] = cfg.Option(default={}, type=dict)

    @classmethod
    def get_desc_from_data_path(cls, path: Path, limit: int = 5) -> "DatasetDescription":
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            path = path.parent
        found = False
        root: Optional[list[Path]] = None
        for i in range(limit):
            root = list(path.glob('ds_root.yaml'))
            if root:
                found = True
                break
            path = path.parent
        if not found:
            raise FileNotFoundError('Can not find dataset root file')
        desc = yaml.safe_load(root[0].open())
        desc['root'] = root[0].parent  # roo_file's directory
        return DatasetDescription(cfg.DictConfigLoader(desc))
