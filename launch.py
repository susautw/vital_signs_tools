import argparse
import importlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "src"))

cmds = {
    "mmwshow": importlib.import_module('occupancy_and_vital_signs_detection.mmwshow'),
    "mmwsave": importlib.import_module('occupancy_and_vital_signs_detection.mmwsave'),
}


def main():
    args, sub_args = get_arg_parser().parse_known_args()
    cmds[args.cmd].main(sub_args)


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("cmd", type=str, choices=cmds.keys())
    return parser


if __name__ == '__main__':
    main()
