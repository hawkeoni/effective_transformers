"""
Adapted from
https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py
"""
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np


MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25


def generate_tree(depth: int, max_depth: int, max_args: int):
    """Generate tree-like equations.
    Args:
        depth: current depth of the node, int.
        max_depth: maximum depth of the tree, int.
        max_args: maximum number of arguments per operator, int.
    Returns:
        The root node of a tree structure.
    """
    if depth < max_depth:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value, 1
    else:
        length = 2
        num_values = random.randint(2, max_args)
        values = []
        for _ in range(num_values):
            sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
            values.append(sub_t)
            length += sub_l

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t, length


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return "( " + to_string(t[0]) + " " + to_string(t[1]) + " )"


def to_value(t):
    """Compute the output of equation t.
    Args:
        t: a tree structure that represents equation t, list.
    Returns:
        The result of equation t, int.
    """
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return np.sum(l[1]) % 10
    elif isinstance(l, tuple):
        # We"ve hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])


def write_to_file(data: List[Tuple["Tree", int]], filename: Path):
    """Write to file output."""
    df_dict = {"Source": [], "Target": []}
    for tree, value in data:
        df_dict["Source"].append(tree)
        df_dict["Target"].append(value)
    df = pd.DataFrame(df_dict)
    df.to_csv(filename, index=False, sep="\t")


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--num_test_samples", type=int, default=2000)
    parser.add_argument("--num_valid_samples", type=int, default=2000)
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="maximum tree depth of training sequences.",
    )
    parser.add_argument(
        "--max_args",
        type=int,
        default=10,
        help="maximum number of arguments per operator in training sequences.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="maximum number of arguments per operator in training sequences.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=500,
        help="maximum number of arguments per operator in training sequences.",
    )
    parser.add_argument("--task", type=str, default="basic")
    parser.add_argument("--output_dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    data = set()
    num_samples = (
        args.num_train_samples + args.num_test_samples + args.num_valid_samples
    )
    print("Started generating")
    while len(data) < num_samples:
        tree, length = generate_tree(1, args.max_depth, args.max_args)
        length = to_string(tree).count(" ") + 1
        if length > args.min_length and length < args.max_length:
            data.add(tree)
            if len(data) % 1000 == 0:
                print("Processed {}".format(len(data)))
    train = []
    for example in data:
        train.append([to_string(example), to_value(example)])

    val = train[args.num_train_samples :]
    test = val[args.num_valid_samples :]
    val = val[: args.num_valid_samples]
    train = train[: args.num_train_samples]

    args.output_dir.mkdir(exist_ok=True)
    write_to_file(train, args.output_dir / "train.tsv")
    write_to_file(val, args.output_dir / "val.tsv")
    write_to_file(test, args.output_dir / "test.tsv")


if __name__ == "__main__":
    main()
