# -*- coding: utf-8 -*-
"""
Perform autocalibration procedure.
"""
import argparse

from qibolab import Platform

parser = argparse.ArgumentParser()
parser.add_argument(
    "--platform",
    default="tiiq",
    type=str,
    help="Platform name for calibration. Default: tiiq.",
)


def main():
    # read command line arguments
    args = vars(parser.parse_args())

    # extract platform name
    platform_name = args.get("platform")

    # load platform and connect
    platform = Platform(platform_name)
    platform.connect()
    platform.setup()

    # run auto calibration
    platform.run_calibration()
