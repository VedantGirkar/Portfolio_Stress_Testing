"""Command-line entry point."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stress-portfolio",
        description="Stress-tested equity portfolio research platform.",
    )
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
