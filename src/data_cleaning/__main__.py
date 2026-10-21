"""CLI: python -m src.data_cleaning [--data PATH] [--out DIR]"""
import argparse

from .cleaning import DEFAULT_DATA_PATH, build_features, save_features


def main() -> None:
    p = argparse.ArgumentParser(description="Clean and engineer features for the attrition data.")
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the Excel file.")
    p.add_argument("--out", default="data", help="Directory for train_features.csv and test_features.csv.")
    args = p.parse_args()

    fs = build_features(args.data)
    train_path, test_path = save_features(fs, args.out)
    print(f"[save] {train_path}\n[save] {test_path}")


if __name__ == "__main__":
    main()
