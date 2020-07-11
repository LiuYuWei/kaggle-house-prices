"""This file is main file."""
# import relation package.
import argparse
import sys

# import project package.
from src.app.data_etl_app import DataEtlApp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is main file.')
    parser.add_argument('--mode', type=str, help='Mode', required=True)
    args = parser.parse_args()
    if args.mode == "etl":
        data_etl_app = DataEtlApp()
        data_etl_app.start()
    elif args.mode == "train":
        pass
    elif args.mode == "eval":
        pass
    else:
        sys.exit(1)
