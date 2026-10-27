# Local data

Run `python -m scripts.download_data` from the repository root. It downloads
MovieLens Latest Small from GroupLens and places `ratings.csv`, `movies.csv`,
the official `README.txt`, and a download manifest under `data/raw/ml-latest-small/`.
This directory is ignored by Git. No course materials or private test labels are used.

The recorded snapshot has 100,836 ratings, 610 users, and 9,742 catalogue movies.
`checksums.json` records the official archive URL and SHA-256 digests of the archive
and the two input files. Since Latest Small can change, run
`python -m scripts.verify_reproduction` before comparing with the recorded results.
It verifies both input files and the exact seeded split membership/order.

Source: https://grouplens.org/datasets/movielens/latest/

The loader rejects malformed rows, duplicate user/movie pairs, missing values,
out-of-range ratings and mismatched catalogue IDs. It does not silently remove or
impute interactions. Encoders are fitted on training IDs only; unknown holdout IDs
remain unknown and use documented prediction fallbacks.

The dataset remains subject to [GroupLens conditions](../results/DATA_LICENSE.md).
Source code's MIT license does not replace those terms. Raw data is not distributed
in this repository.
