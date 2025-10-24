# sketch-BLEU

sketch-BLEU is a repository-level code similarity metric, derived from
[k4black/codebleu](https://github.com/k4black/codebleu) at commit
[`b0edb62`](https://github.com/k4black/codebleu/commit/b0edb622f6a52fe9d1edc407be5061d3e1462a7f)
— the upstream `main` tip, 16 commits after release `0.7.1`.

CodeBLEU scores a pair of code *files*. sketch-BLEU scores a pair of code
*repositories*, for evaluating generated code where the unit of output is a
whole project rather than a single translated function. Reaching that meant
changing the metric itself, not just wrapping it:

- **`calc_repobleu`** — a new entry point that walks two repository trees,
  stacks their sources, and combines four component scores under CodeBLEU's
  usual alpha/beta/gamma/theta weighting.
- **Repository-aware structure match** — replaces per-file AST comparison with
  a single tree spanning directory layout *and* per-file tree-sitter ASTs, so
  where a file sits counts toward the score. Directory entries are sorted, so
  the result does not depend on filesystem order.
- **Cross-function dataflow match** — functions are extracted from both repos
  and matched one-to-one by data-flow-graph similarity using the Hungarian
  algorithm, rather than assuming a pre-existing alignment.
- **Precision instead of recall** in the weighted n-gram component, measuring
  how much of the generated code is justified by the reference rather than how
  much of the reference is covered.
- **Unigram-only n-gram weights** `(1, 0, 0, 0)`, because higher-order n-grams
  over a stacked repository largely measure incidental file ordering.
- **A half-length brevity penalty** — `1 / (1 + log(r / 2h))`, flat once the
  hypothesis reaches half the reference length. The standard `exp(1 - r/h)`
  collapses towards zero for generated repositories and swamps every other
  component.

`calc_codebleu` is unchanged and still scores file pairs exactly as upstream
does. Note that upstream's test suite asserts the original CodeBLEU values, so
the tests covering the changed components fail here by design.

## Usage

```python
from codebleu import calc_repobleu
from pathlib import Path

calc_repobleu(Path("reference_repo"), Path("generated_repo"), "python")
# {'repo_bleu': 0.74, 'ngram_match_score': 0.64, 'weighted_ngram_match_score': 0.82,
#  'structure_match_score': 0.88, 'dataflow_match_score': 0.63}
```

or from the command line:

```bash
python -m codebleu --ref reference_repo --hyp generated_repo --lang python
```

**Status: research code.** It is not published to PyPI and has no CI of its own.
The commit history was reorganised from the original development history into
self-contained commits; the original is preserved under the `legacy-history` tag.

Everything below the line is upstream's documentation of CodeBLEU, kept as-is
and describing `calc_codebleu`.

---


This repository contains an unofficial `CodeBLEU` implementation that supports `Linux`, `MacOS` (incl. M-series) and `Windows`. It is available through `PyPI` and the `evaluate` library.

Available for: `Python`, `C`, `C#`, `C++`, `Java`, `JavaScript`, `PHP`, `Go`, `Ruby`, `Rust`.

---

The code is based on the original [CodeXGLUE/CodeBLEU](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU) and updated version by [XLCoST/CodeBLEU](https://github.com/reddy-lab-code-research/XLCoST/tree/main/code/translation/evaluator/CodeBLEU).  It has been refactored, tested, built for macOS and Windows, and multiple improvements have been made to enhance usability.

## Metric Description

> An ideal evaluation metric should consider the grammatical correctness and the logic correctness.
> We propose weighted n-gram match and syntactic AST match to measure grammatical correctness, and introduce semantic data-flow match to calculate logic correctness.
> ![CodeBLEU](CodeBLEU.jpg)  
[from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU) repo]

In a nutshell, `CodeBLEU` is a weighted combination of `n-gram match (BLEU)`, `weighted n-gram match (BLEU-weighted)`, `AST match` and `data-flow match` scores.

The metric has shown higher correlation with human evaluation than `BLEU` and `accuracy` metrics.


## Installation

This library requires `so` file compilation with tree-sitter, so it is platform dependent.  
Currently available for `Linux` (manylinux), `MacOS` and `Windows` with Python 3.8+.

The metrics is available as [pip package](https://pypi.org/project/codebleu/) and can be installed as indicated above:
```bash
pip install codebleu
```
or directly from git repo (require internet connection to download tree-sitter):
```bash
pip install git+https://github.com/k4black/codebleu.git
```

Also you have to install tree-sitter language you need (e.g. python, rust, etc):
```bash
pip install tree-sitter-python
```
Or you can install all languages:
```bash
pip install codebleu[all]
```

Note: At the moment (May 2024) precompiled languages are NOT available for arm64 (M1) MacOS, so you have to install and build tree-sitter languages manually, for example:
```bash
pip install pip install git+https://github.com/tree-sitter/tree-sitter-python.git
```


## Usage 

```python
from codebleu import calc_codebleu

prediction = "def add ( a , b ) :\n return a + b"
reference = "def sum ( first , second ) :\n return second + first"

result = calc_codebleu([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)
# {
#   'codebleu': 0.5537, 
#   'ngram_match_score': 0.1041, 
#   'weighted_ngram_match_score': 0.1109, 
#   'syntax_match_score': 1.0, 
#   'dataflow_match_score': 1.0
# }
```
where `calc_codebleu` takes the following arguments:
- `refarences` (`list[str]` or `list[list[str]]`): reference code
- `predictions` (`list[str]`) predicted code
- `lang` (`str`): code language, see `codebleu.AVAILABLE_LANGS` for available languages (python, c_sharp c, cpp, javascript, java, php, go and ruby at the moment)
- `weights` (`tuple[float,float,float,float]`): weights of the `ngram_match`, `weighted_ngram_match`, `syntax_match`, and `dataflow_match` respectively, defaults to `(0.25, 0.25, 0.25, 0.25)`
- `tokenizer` (`callable`): to split code string to tokens, defaults to `s.split()`

and outputs the `dict[str, float]` with following fields:
- `codebleu`: the final `CodeBLEU` score
- `ngram_match_score`: `ngram_match` score (BLEU)
- `weighted_ngram_match_score`: `weighted_ngram_match` score (BLEU-weighted)
- `syntax_match_score`: `syntax_match` score (AST match)
- `dataflow_match_score`: `dataflow_match` score

Alternatively, you can use `k4black/codebleu` from HuggingFace Spaces (`codebleu` package required):
```python
import evaluate
metric = evaluate.load("dvitel/codebleu")

prediction = "def add ( a , b ) :\n return a + b"
reference = "def sum ( first , second ) :\n return second + first"

result = metric.compute([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25))
```

Feel free to check the HF Space with online example: [k4black/codebleu](https://huggingface.co/spaces/k4black/codebleu) 


## Contributing

Contributions are welcome!  
If you have any questions, suggestions, or bug reports, please open an issue on GitHub.

Make your own fork and clone it:
```bash
git clone https://github.com/k4black/codebleu
```

For development, you need to install library with `all` precompiled languages and `test` extra:  
(require internet connection to download tree-sitter)
```bash
python -m pip install -e .[all,test]
python -m pip install -e .\[all,test\]  # for macos
```

For testing just run pytest:
```bash
python -m pytest
```

To perform a style check, run:
```bash
python -m isort codebleu --check
python -m black codebleu --check
python -m ruff codebleu
python -m mypy codebleu
```


## License

This project is licensed under the terms of the MIT license.


## Citation

Official [CodeBLEU paper](https://arxiv.org/abs/2009.10297) can be cited as follows:
```bibtex
@misc{ren2020codebleu,
      title={CodeBLEU: a Method for Automatic Evaluation of Code Synthesis}, 
      author={Shuo Ren and Daya Guo and Shuai Lu and Long Zhou and Shujie Liu and Duyu Tang and Neel Sundaresan and Ming Zhou and Ambrosio Blanco and Shuai Ma},
      year={2020},
      eprint={2009.10297},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```