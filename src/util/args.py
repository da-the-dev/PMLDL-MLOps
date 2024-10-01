import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        help="Number of epochs. Default is 10",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-lr",
        help="Learning rate. Default is 0.001",
        default=0.001,
        type=float,
    )

    args = parser.parse_args()
    return args
