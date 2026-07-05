#!/usr/bin/env python
"""CLI entry point for the Chile vehicle demand forecasting pipeline.

Usage:
    python data/generate_synthetic_data.py   # once, to create the sample dataset
    python run_pipeline.py
"""

import logging

from chile_forecast.pipeline import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
