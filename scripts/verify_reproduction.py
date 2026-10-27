"""Verify exact input files and platform-independent seeded split fingerprints."""

import argparse
import hashlib
import json
from pathlib import Path

from src.data_loader import DEFAULT_DATA, load_data, sha256_file
from src.preprocessing import split_ratings


def main() -> None:
    """Compare local data with the published experiment provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--provenance", type=Path, default=Path("results/provenance.json"))
    args = parser.parse_args()
    expected = json.loads(args.provenance.read_text(encoding="utf-8"))
    for filename, digest in expected["input_sha256"].items():
        if sha256_file(args.data_dir / filename) != digest:
            parser.exit(1, f"Input snapshot mismatch: {filename}\n")
    ratings, _ = load_data(args.data_dir)
    splits = split_ratings(ratings, seed=expected["seed"])
    for name, frame in zip(("train", "validation", "test"), splits):
        canonical = (
            frame[["userId", "movieId", "rating", "timestamp"]]
            .to_csv(index=False, lineterminator="\n")
            .encode()
        )
        if hashlib.sha256(canonical).hexdigest() != expected["split_sha256"][name]:
            parser.exit(1, f"Split fingerprint mismatch: {name}\n")
    print("Verified both input checksums and all three seeded split fingerprints.")


if __name__ == "__main__":
    main()
