import pprint
from pathlib import Path

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection import Config


def main():
    cp = Path("configs/vod_vs_68xx_10fps.cfg")
    with cp.open() as fp:
        c = Config(MMWaveConfigLoader(fp.readlines()))
    pprint.pprint(c.to_dict(), indent=4, width=120, sort_dicts=False)


if __name__ == '__main__':
    main()
