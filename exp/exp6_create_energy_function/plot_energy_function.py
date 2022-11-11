from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Iterator, Protocol, Iterable, Optional, cast, Callable

import numpy as np
from matplotlib import pyplot as plt

from ovsd import MMWInfo
from ovsd.dataset import V3Dataset, DatasetDescription
from ovsd.plot import AbstractZone, Zone

BASE_DIR = Path(__file__).parent
DS_DIR = BASE_DIR.parent.parent / "datasets"


class Context:
    start: int
    stop: int
    source: Path
    out: Path
    zone: AbstractZone
    infos: list[MMWInfo]
    base_info: MMWInfo


def main():
    base_frame_number = 848
    ctx = Context()
    ctx.start = 845
    ctx.stop = 867
    ctx.source = DS_DIR / "ds5/bins/test.20220830T142514.bin"
    ctx.out = BASE_DIR / "out"
    ctx.out.mkdir(parents=True, exist_ok=True)
    ctx.zone = Zone(15, 7, 20, 7)
    ctx.infos = list(get_mmw_infos(ctx.source, ctx.start, ctx.stop))
    try:
        ctx.base_info = next(info for info in ctx.infos if info.get_header().frame_number == base_frame_number)
    except StopIteration:
        print("base info does not exist.")
        exit(1)

    profiles = ["sof"]
    profiles.extend([
        f"autocorr_{calc_name}"
        for calc_name in calcs
    ])

    for profile in profiles:
        print(profile)
        process_profile(profile, ctx)
    print("Done.")


def process_profile(method_name: str, ctx: Context):
    fig: plt.Figure = plt.figure(figsize=(4.475 * (ctx.stop - ctx.start + 1), 4), frameon=False, tight_layout=dict(pad=0))
    fig.suptitle(method_name)
    ax: plt.Axes = plt.subplot(111)
    ax.set_axis_off()
    method: MMWInfoEnergyTransform
    if method_name == "sof":
        method = sum_of_frames
    elif method_name.startswith("autocorr_"):
        calc_name = method_name[9:]
        if calc_name not in calcs:
            raise RuntimeError(f"calculator does not exist: {calc_name}")
        method = cast(MMWInfoEnergyTransform, partial(auto_correlation, base=ctx.base_info, calc=calcs[calc_name]))
    else:
        raise RuntimeError(f"method doesn't exist: {method_name}")

    vals = method(ctx.infos, ctx.zone)
    ax.plot(vals, linewidth=30)
    filename = f'{ctx.source.stem}_{method_name}'
    fig.savefig(ctx.out / f'{filename}.png')
    with open(ctx.out / f'{filename}.csv', "w") as fp:
        vals.tofile(fp, sep="\n")


def get_mmw_infos(source: Path, start: int, stop: int) -> Iterator[MMWInfo]:
    source_it = V3Dataset(DatasetDescription.get_desc_from_data_path(source)).get_source_iter(source)
    for info in source_it:
        fn = info.get_header().frame_number
        if fn >= start:
            yield info
        if fn >= stop:
            break


class MMWInfoEnergyTransform(Protocol):
    def __call__(self, infos: Iterable[MMWInfo], zone: Optional[AbstractZone]) -> np.ndarray: ...


def sum_of_frames(infos: Iterable[MMWInfo], zone: Optional[AbstractZone]) -> np.ndarray:
    return np.array([info.get_hmap(zone).sum() for info in infos])


def auto_correlation(
        infos: Iterable[MMWInfo],
        zone: Optional[AbstractZone],
        calc: "ACCalculator",
        base: MMWInfo = None,
) -> np.ndarray:
    vals = []
    if base is not None:
        calc.init_base(base.get_hmap(zone))

    for info in infos:
        hmap = info.get_hmap(zone)
        if not calc.initialized:
            calc.init_base(hmap)
        vals.append(calc.calc_ar(hmap))
    return np.array(vals)


class ACCalculator(ABC):
    _base_initialized = False

    def init_base(self, hmap: np.ndarray) -> None:
        self._base_initialized = True

    @abstractmethod
    def calc_ar(self, hmap: np.ndarray) -> float: ...

    @property
    def initialized(self) -> bool:
        return self._base_initialized


def identical(x):
    return x


class LambdaNormalizedCalculator(ACCalculator):
    base_hmap: np.ndarray

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray] = None):
        if fn is None:
            fn = identical
        self.fn = fn

    def init_base(self, hmap: np.ndarray) -> None:
        super().init_base(hmap)
        self.base_hmap = self.fn(hmap)

    def calc_ar(self, hmap: np.ndarray) -> float:
        return (self.fn(hmap) * self.base_hmap).sum()


class StdNormalizedCalculator(ACCalculator):
    base_hmap: np.ndarray
    base_std: np.ndarray

    def init_base(self, hmap: np.ndarray) -> None:
        super().init_base(hmap)
        self.base_hmap = hmap - hmap.mean()
        self.base_std = hmap.std()

    def calc_ar(self, hmap: np.ndarray) -> float:
        return ((hmap - hmap.mean()) * self.base_hmap).sum() / (hmap.std() * self.base_std)


calcs = {
    # "ori": LambdaNormalizedCalculator(),
    "std": StdNormalizedCalculator(),
    # "minmax": LambdaNormalizedCalculator(lambda x: x / x.max()),
    "32000": LambdaNormalizedCalculator(lambda x: x / 32000)
}

if __name__ == '__main__':
    main()
