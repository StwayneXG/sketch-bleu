# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import argparse
from pathlib import Path
from typing import List, Tuple

from codebleu import calc_codebleu, calc_repobleu

PACKAGE_DIR = Path(__file__).parent


def main(
    ref_repo: str,
    hyp_repo: str,
    lang: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> None:
    repo_bleu_score = calc_repobleu(
        Path(ref_repo),
        Path(hyp_repo),
        lang,
    )
    print(repo_bleu_score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ref", type=str, required=True, help="reference files")
    parser.add_argument("--hyp", type=str, required=True, help="hypothesis file")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["java", "js", "c_sharp", "php", "go", "python", "ruby", "rust"],
    )
    parser.add_argument("--params", type=str, default="0.25,0.25,0.25,0.25", help="alpha, beta and gamma")

    args = parser.parse_args()

    lang = args.lang
    alpha, beta, gamma, theta = [float(x) for x in args.params.split(",")]

    main(
        args.ref,
        args.hyp,
        args.lang,
        weights=(alpha, beta, gamma, theta),
    )
