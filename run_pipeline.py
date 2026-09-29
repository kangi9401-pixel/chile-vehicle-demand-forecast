#!/usr/bin/env python
"""CLI entry point for the Chile vehicle demand forecasting pipeline.

Usage:
    python data/generate_synthetic_data.py   # once, to create the sample dataset
    python run_pipeline.py
"""

import argparse
import logging

from chile_forecast.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-deepar",
        action="store_true",
        help="Fast smoke-test mode; the documented full run includes DeepAR.",
    )
    parser.add_argument("--deepar-epochs", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(include_deepar=not args.skip_deepar, deepar_epochs=args.deepar_epochs)
    print(result.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
