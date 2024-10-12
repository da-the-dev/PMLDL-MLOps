import argparse
import os


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        dest="epochs",
        help="Number of epochs. Default is 10",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-lr",
        dest="lr",
        help="Learning rate. Default is 0.001",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        help="Path to train and test data. Default is $(pwd)/data",
        default=os.path.join(os.getcwd(), "data"),
        type=str,
    )
    parser.add_argument(
        "--course-labels",
        dest="course_labels",
        help="Whether to use course labels or not. True by default",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-b",
        dest="batch_size",
        help="Batch size. Default is 32",
        default=32,
        type=int,
    )

    args = parser.parse_args()
    return args
