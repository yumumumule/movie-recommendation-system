"""Download the official MovieLens Latest Small dataset."""

import argparse
from pathlib import Path

from src.data_loader import DEFAULT_DATA, download_data


def main() -> None:
    """Parse download arguments and print the local dataset location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(download_data(args.data_dir))


if __name__ == "__main__":
    main()
