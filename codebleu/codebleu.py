# Copyright (c) Microsoft Corporation.
# Copyright (c) 2023 Konstantin Chernyshev.
# Licensed under the MIT license.
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from . import bleu, dataflow_match, syntax_match, weighted_ngram_match
from .utils import AVAILABLE_LANGS, get_tree_sitter_language
import psutil
import os

PACKAGE_DIR = Path(__file__).parent

import logging
import time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def calc_codebleu(
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    lang: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    tokenizer: Optional[Callable] = None,
    keywords_dir: Path = PACKAGE_DIR / "keywords",
) -> Dict[str, float]:
    """Calculate CodeBLEU score

    Args:
        predictions: list of predictions
        references: list of lists with references
        lang: input language, one of AVAILABLE_LANGS
        weights: weights of the ngram_match, weighted_ngram_match, syntax_match, and dataflow_match respectively
        tokenizer: tokenizer function, Defaults to lambda s: s.split()
        keywords_dir: path to the directory with keywords files
        lang_so_file: path to the .so file with the parser for the language

    Return:
        Scores dict
    """
    assert len(references) == len(predictions), "Number of references and predictions should be the same"
    assert lang in AVAILABLE_LANGS, f"Language {lang} is not supported (yet). Available languages: {AVAILABLE_LANGS}"
    assert len(weights) == 4, "weights should be a tuple of 4 floats (alpha, beta, gamma, theta)"
    assert keywords_dir.exists(), f"keywords_dir {keywords_dir} does not exist"

    # get the tree-sitter language for a given language
    tree_sitter_language = get_tree_sitter_language(lang)

    # preprocess inputs
    references = [[x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()] for ref in references]
    hypothesis = [x.strip() for x in predictions]

    # calculate ngram match (BLEU)
    if tokenizer is None:

        def tokenizer(s):
            return s.split()

    tokenized_hyps = [tokenizer(x) for x in hypothesis]
    tokenized_refs = [[tokenizer(x) for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
        keywords = [x.strip() for x in f.readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [
        [[reference_tokens, make_weights(reference_tokens, keywords)] for reference_tokens in reference]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )

    alpha, beta, gamma, theta = weights
    code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * (dataflow_match_score or 1)
    )

    return {
        "codebleu": code_bleu_score,
        "ngram_match_score": ngram_match_score,
        "weighted_ngram_match_score": weighted_ngram_match_score,
        "syntax_match_score": syntax_match_score,
        "dataflow_match_score": dataflow_match_score,
    }

import os
import ast
import math
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
import ast

def stack_source_code(file_list: List[Path]) -> str:
    source_code = ""
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            source_code += f.read().strip() + "\n"

    return source_code

def get_file_list(dir: Path, ext: str) -> List[Path]:
    SPECIAL_FILEPATHS = ["augment_comments.py", "mutate_methodnames.py", "reorder_methods.py"]
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file in SPECIAL_FILEPATHS:
                continue
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list

def extract_functions(source):
    lines = source.splitlines()

    # Try Python 3 first
    try:
        tree = ast.parse(source)
        node_type = ast.FunctionDef
    except SyntaxError:
        # Fallback to Python 2
        try:
            import typed_ast.ast27 as ast27
            tree = ast27.parse(source)
            node_type = ast27.FunctionDef
        except:
            return []
    
    functions = [node for node in ast.walk(tree) if isinstance(node, node_type)]
    
    result = []
    for func in functions:
        # Try ast.unparse first (Python 3.9+)
        if hasattr(ast, 'unparse'):
            try:
                result.append(ast.unparse(func))
                continue
            except:
                pass
        
        # Fallback: extract by line numbers
        start = func.lineno - 1
        base_indent = len(lines[start]) - len(lines[start].lstrip())
        
        end = start + 1
        while end < len(lines):
            line = lines[end]
            # Skip empty lines - they don't determine function boundaries
            if line.strip() == "":
                end += 1
                continue
            # Only stop if we find a non-empty line with less/equal indentation
            if len(line) - len(line.lstrip()) <= base_indent:
                break
            end += 1
        
        result.append('\n'.join(lines[start:end]))
    
    return result

def get_file_content(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        file_content = f.read().strip()
    return file_content

def get_repo_files(reference_repo: Path, prediction_repo: Path, ext: str = ".py") -> Tuple[List[Path], List[Path]]:
    start_time = time.time()
    
    reference_files = get_file_list(reference_repo, ext)
    prediction_files = get_file_list(prediction_repo, ext)

    logging.debug(f"Number of reference files: {len(reference_files)}")
    logging.debug(f"Number of prediction files: {len(prediction_files)}")
    logging.debug(f"Time taken to get file lists: {(time.time() - start_time):.2f} seconds")
    return reference_files, prediction_files

def get_repo_source_codes(reference_files: List[Path], prediction_files: List[Path]) -> Tuple[List[str], List[str]]:
    start_time = time.time()
    
    reference_sources = []
    for file in reference_files:
        reference_sources.append(get_file_content(file))
    prediction_sources = []
    for file in prediction_files:
        prediction_sources.append(get_file_content(file))
    
    logging.debug(f"Time taken to get file contents: {(time.time() - start_time):.2f} seconds")
    return reference_sources, prediction_sources

def stack_repo_source_codes(reference_sources: List[str], prediction_sources: List[str]) -> Tuple[str, str]:
    start_time = time.time()

    reference_source = "\n".join(reference_sources)
    prediction_source = "\n".join(prediction_sources)

    logging.debug(f"Time taken to stack file contents: {(time.time() - start_time):.2f} seconds")
    return reference_source, prediction_source

def tokenize_repo_source_codes(reference_source: str, prediction_source: str, tokenizer: Callable) -> Tuple[List[str], List[str]]:
    start_time = time.time()

    if tokenizer is None:
        def tokenizer(s):
            return s.split()

    tokenized_refs = tokenizer(reference_source)
    tokenized_hyps = tokenizer(prediction_source)

    logging.debug(f"Time taken to tokenize source codes: {(time.time() - start_time):.2f} seconds")
    return tokenized_refs, tokenized_hyps

def calc_ngram_match(tokenized_refs: List[str], tokenized_hyps: List[str]) -> float:
    start_time = time.time()
    ngram_match_score = bleu.corpus_bleu([[tokenized_refs]], [tokenized_hyps])
    logging.debug(f"Time taken to calculate ngram match: {(time.time() - start_time):.2f} seconds")
    return ngram_match_score

def calc_weighted_ngram_match(tokenized_refs: List[str], tokenized_hyps: List[str], keywords_dir: Path, lang: str) -> float:
    start_time = time.time()
    with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
        keywords = [x.strip() for x in f.readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [tokenized_refs, make_weights(tokenized_refs, keywords)]
    tokenized_hyps_with_weights = [tokenized_hyps, make_weights(tokenized_hyps, keywords)]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu([[tokenized_refs_with_weights]], [tokenized_hyps_with_weights])
    
    logging.debug(f"Time taken to calculate weighted ngram match: {(time.time() - start_time):.2f} seconds")
    return weighted_ngram_match_score

def calc_structure_match(reference_repo: Path, prediction_repo: Path, lang: str, tree_sitter_language) -> float:
    start_time = time.time()
    structure_match_score = syntax_match.repo_structure_match(
        reference_repo, prediction_repo, lang, tree_sitter_language=tree_sitter_language
    )
    logging.debug(f"Time taken to calculate structure match: {(time.time() - start_time):.2f} seconds")
    return structure_match_score

