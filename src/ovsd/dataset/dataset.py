import ctypes
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Type

import h5py
import numpy as np

import tlv
from . import DatasetDescription
from .. import structures, MMWInfo
from ..plot import Zone
from ..structures import VitalSignsVectorTLV


class IDataset(ABC):
    @abstractmethod
    def get_source_iter(self, path: Path) -> Iterator[MMWInfo]: ...

    @abstractmethod
    def get_description(self) -> DatasetDescription: ...


class CachedDataset(IDataset, ABC):
    _cached = True

    @abstractmethod
    def _get_cached_source_iter(self, path: Path) -> Iterator[MMWInfo]:
        ...

    @abstractmethod
    def _get_non_cached_source_iter(self, path: Path) -> Iterator[MMWInfo]:
        ...

    @abstractmethod
    def make_cache(self, path) -> Path:
        ...

    def no_cache(self) -> None:
        self._cached = False

    def enable_cache(self) -> None:
        self._cached = True

    def get_source_iter(self, path: Path) -> Iterator[MMWInfo]:
        if path.is_absolute():
            path = path.relative_to(self.get_description().root.absolute())
        if self._cached:
            return self._get_cached_source_iter(self.make_cache(path))
        else:
            return self._get_non_cached_source_iter(path)


class V3Dataset(CachedDataset):
    compound_type: Type[ctypes.Structure] = None

    def __init__(self, desc: DatasetDescription):
        self.desc = desc
        self.data_dir = desc.root / (desc.attrs['data_dir'] if 'data_dir' in desc.attrs else 'data')
        self.cached_dir = desc.root / 'cache'

    def get_description(self) -> DatasetDescription:
        return self.desc

    def make_cache(self, path: Path) -> Path:
        file_path = path.relative_to(self.data_dir)
        cached_file_path = self.cached_dir / file_path.with_suffix('.h5')
        cached_file_path.parent.mkdir(parents=True, exist_ok=True)
        if cached_file_path.is_file():
            return cached_file_path

        self._initialize_compound_type()

        try:
            with (path.open("rb") as fp,
                  h5py.File(cached_file_path, "w") as h5fp
                  ):
                self._create_h5(fp, h5fp)
        except Exception:
            # delete the cached file when error is encountered.
            cached_file_path.unlink(missing_ok=True)
            raise

        return cached_file_path

    def _initialize_compound_type(self):
        if self.compound_type is None:
            class _CompoundType(ctypes.Structure):
                _fields_ = [
                    ("header", tlv.FrameHeader),
                    ("heatmap", structures.heatmap_type),
                    ("decision", structures.decision_type),
                    ('vitalsigns', structures.vital_signs_type)
                ]

            self.compound_type = _CompoundType

    def _create_h5(self, fp, h5fp):
        compound_data_list = []
        for frame in tlv.from_stream(fp, structures.get_parser()):
            compound_data = {'header': frame.frame_header}
            for t in frame:
                if isinstance(t, structures.heatmap_type):
                    compound_data['heatmap'] = t
                if isinstance(t, structures.decision_type):
                    compound_data['decision'] = t
                if isinstance(t, structures.vital_signs_type):
                    compound_data['vitalsigns'] = t
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # noinspection PyTypeChecker
                compound_data_list.append(np.array(self.compound_type(**compound_data)))

        h5fp.create_dataset('mmw_infos', data=np.asarray(compound_data_list))
        metadata = h5fp.create_dataset('metadata', shape=())
        metadata.attrs['zones'] = [
            list(z.to_dict().values())
            for z in structures.config.zone_def.zones
        ]

    def _get_cached_source_iter(self, path: Path) -> Iterator[MMWInfo]:
        with h5py.File(path) as fp:
            zones = [
                Zone(*z_def)
                for z_def in fp['metadata'].attrs['zones']
            ]
            len_zone = len(zones)
            vs_type = VitalSignsVectorTLV * len_zone
            mmw_infos_source = np.array(fp['mmw_infos'])
            for mmw_info_source in mmw_infos_source:
                yield MMWInfo(
                    zone_infos=dict(zip(zones, vs_type.from_buffer(mmw_info_source['vitalsigns']))),
                    zone_decisions=dict(zip(zones, mmw_info_source['decision'])),
                    hmap=mmw_info_source['heatmap']
                )

    def _get_non_cached_source_iter(self, path: Path) -> Iterator[MMWInfo]:
        raise RuntimeError('No cache is currently disabled.')
