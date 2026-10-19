"""
Lab 2.

BPE and machine translation evaluation
"""

# pylint:disable=unused-argument
import json
from math import exp,log
from typing import Sequence


def prepare_word(
    raw_word: str, start_of_word: str | None, end_of_word: str | None
) -> tuple[str, ...] | None:
    """
    Tokenize a word into characters and attach optional boundary tokens.

    Args:
        raw_word (str): Original word
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str | None): A token that signifies the end of word

    Returns:
        tuple[str, ...] | None: Preprocessed word

    In case of corrupt input arguments, None is returned
    """
    if not all([isinstance(raw_word, str),
                 isinstance(start_of_word, (str, type(None))),
                 isinstance(end_of_word, (str, type(None)))]):
        return None

    prepared_words = []

    if start_of_word is not None:
        prepared_words += [start_of_word]

    prepared_words.extend(list(raw_word))

    if end_of_word is not None:
        prepared_words += [end_of_word]

    return tuple(prepared_words)


def collect_frequencies(
    text: str, start_of_word: str | None, end_of_word: str
) -> dict[tuple[str, ...], int] | None:
    """
    Count number of occurrences of each word.

    Args:
        text (str): Original text with no preprocessing
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str): A token that signifies the end of word

    Returns:
        dict[tuple[str, ...], int] | None: Dictionary where key - preprocessed word,
            value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not all([isinstance(text, str),
                 isinstance(start_of_word, (str, type(None))),
                 isinstance(end_of_word, str)]):
        return None

    word_frequencies = {}

    for line in text.split('\n'):
        for word in line.split():
            prepared_word = prepare_word(word, start_of_word, end_of_word)
            if prepared_word is None:
                return None

            if prepared_word not in word_frequencies:
                word_frequencies[prepared_word] = 0
            word_frequencies[prepared_word] += 1

    return word_frequencies


def count_tokens_pairs(
    word_frequencies: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int] | None:
    """
    Count number of occurrences of each pair of subsequent tokens.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences

    Returns:
        dict[tuple[str, str], int] | None: A dictionary where
            key - token pair, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(word_frequencies, dict)
        or not all(isinstance(k, tuple) and isinstance(v, int) for k, v in word_frequencies.items())
        or not all(isinstance(token, str) for word in word_frequencies.keys() for token in word)
    ):
        return None

    pair_frequencies = {}

    for word, freq in word_frequencies.items():
        for i in range(len(word) - 1):
            token_pair = (word[i], word[i + 1])
            if token_pair not in pair_frequencies:
                pair_frequencies[token_pair] = 0
            pair_frequencies[token_pair] += freq

    return pair_frequencies


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Update word frequency dictionary by replacing a pair of tokens with a merged one.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        pair (tuple[str, str]): A pair of tokens to be merged

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(word_frequencies, dict)
        or not all(isinstance(word, tuple)
        and isinstance(freq, int) for word, freq in word_frequencies.items())
        or not isinstance(pair, tuple) or len(pair) != 2
        or not all(isinstance(token, str) for token in pair)
    ):
        return None

    new_tokens_frequencies = {}

    new_token = pair[0] + pair[1]

    for word, freq in word_frequencies.items():
        new_token_in_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_token_in_word.append(new_token)
                i += 2
            else:
                new_token_in_word.append(word[i])
                i += 1

        new_word = tuple(new_token_in_word)
        new_tokens_frequencies[new_word] = new_tokens_frequencies.get(new_word, 0) + freq

    return new_tokens_frequencies


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Create required number of new tokens by merging existing ones.

    Args:
        word_frequencies (dict[tuple[str, ...], int] | None): A dictionary where
            key - preprocessed word, value - number of occurrences
        num_merges (int): Required number of new tokens

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if(
        not isinstance(word_frequencies, dict)
        or not isinstance(num_merges, int)
        or not all(isinstance(word, tuple)
        and isinstance(freq, int) for word, freq in word_frequencies.items())
        or not all(isinstance(token, str) for word in word_frequencies.keys() for token in word)
    ):
        return None


    for _ in range(num_merges):
        freq_pairs = count_tokens_pairs(word_frequencies)

        if freq_pairs is None:
            return None

        if not freq_pairs:
            break

        sorted_freq_pairs = sorted(
            freq_pairs.items(),
            key=lambda x: (
                -x[1],
                -len(x[0][0] + x[0][1]),
                x[0][0] + x[0][1]
            )
        )

        sorted_word_frequencies = merge_tokens(word_frequencies, sorted_freq_pairs[0][0])

        if sorted_word_frequencies is None:
            return None

        word_frequencies = sorted_word_frequencies

    return word_frequencies


def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establish correspondence between tokens and its integer identifier.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        unknown_token (str): A token to signify an unknown token

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if(
        not isinstance(word_frequencies, dict)
        or not isinstance(unknown_token, str)
        or not all(isinstance(word, tuple)
        and isinstance(freq, int) for word, freq in word_frequencies.items())
        or not all(isinstance(token, str) for word in word_frequencies.keys() for token in word)
    ):
        return None

    new_tokens = set()
    id_tokens = {}

    for word in word_frequencies.keys():
        for token in word:
            new_tokens.add(token)
            for letter in token:
                new_tokens.add(letter)

    new_tokens.add(unknown_token)

    sorted_tokens = sorted(new_tokens, key=lambda x: (-len(x), x))

    for i, token in enumerate(sorted_tokens):
        id_tokens[token] = i

    return id_tokens


def decode(
    encoded_text: Sequence[int] | None,
    vocabulary: dict[str, int] | None,
    end_of_word_token: str | None,
) -> str | None:
    """
    Translate encoded sequence into decoded one.

    Args:
        encoded_text (Sequence[int] | None): A sequence of token identifiers
        vocabulary (dict[str, int] | None): A dictionary where
            key - token, value - identifier
        end_of_word_token (str | None): An end-of-word token

    Returns:
        str | None: Decoded sequence

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(encoded_text, list)
        or not all(isinstance(token, int) for token in encoded_text)
        or not isinstance(vocabulary, dict)
    ):
        return None

    if (
        encoded_text is None
        or vocabulary is None
        or not all(isinstance(token, str)
        and isinstance(id_vocabulary, int) for token, id_vocabulary in vocabulary.items())
        or not isinstance(end_of_word_token, (str, type(None)))
    ):
        return None




    id_to_token = {id_vocabulary: token for token, id_vocabulary in vocabulary.items()}
    prepared_string = ''
    symbols = []
    for token_id in encoded_text:
        token = id_to_token[token_id]
        if end_of_word_token is not None and end_of_word_token in token:
            token = token.replace(end_of_word_token, " ")

        symbols.append(token)

    return prepared_string.join(symbols)


def tokenize_word(
    word: tuple[str, ...], vocabulary: dict[str, int], end_of_word: str | None, unknown_token: str
) -> list[int] | None:
    """
    Split word into tokens.

    Args:
        word (tuple[str, ...]): Preprocessed word
        vocabulary (dict[str, int]): A dictionary where key - token, value - identifier
        end_of_word (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(word, tuple)
        or not all(isinstance(token, str) for token in word)
        or not isinstance(vocabulary, dict)
        or not all(isinstance(token, str)
        and isinstance(id_vocabulary, int) for token, id_vocabulary in vocabulary.items())
    ):
        return None

    if (
        not isinstance(end_of_word, (str, type(None)))
        or not isinstance(unknown_token, str)
    ):
        return None


    text = ''.join(word)
    if end_of_word is not None and not text.endswith(end_of_word):
        text = text + end_of_word

    sorted_by_length_tokens = sorted(vocabulary.keys(), key=lambda x: (-len(x), x))

    private_tokens = [False] * len(text)

    started_tokens = [""] * len(text)

    for token in sorted_by_length_tokens:
        if token == unknown_token:
            continue


        for i in range(len(text) - len(token) + 1):
            not_private_tokens = not any(
                private_tokens[i + pos_token] for pos_token in range(len(token))
            )
            if not not_private_tokens:
                continue

            if text[i:i + len(token)] != token:
                continue

            for j in range(len(token)):
                private_tokens[i + j] = True
                started_tokens[i] = token

    pos = 0
    tokenized_word = []

    while pos < len(text):
        current_token = started_tokens[pos]
        if current_token:
            tokenized_word.append(vocabulary[current_token])
            pos += len(current_token)
        else:
            tokenized_word.append(vocabulary[unknown_token])
            pos += 1

    return tokenized_word


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Read and retrieve dictionary of type <token: identifier>.

    Args:
        vocab_path (str): A path to the saved vocabulary

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(vocab_path, str):
        return None

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = file.read()

    content = json.loads(vocab)
    if not isinstance(content, dict):
        return None

    return content


def encode(
    original_text: str,
    vocabulary: dict[str, int] | None,
    start_of_word_token: str | None,
    end_of_word_token: str | None,
    unknown_token: str,
) -> list[int] | None:
    """
    Translate original text into a sequence of token identifiers.

    Args:
        original_text (str): Original text
        vocabulary (dict[str, int] | None): A dictionary where key - token, value - identifier
        start_of_word_token (str | None): A start-of-word token
        end_of_word_token (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if (
        not isinstance(original_text, str)
        or not isinstance(vocabulary, dict)
        or not all(isinstance(token, str)
        and isinstance(id_vocabulary, int) for token, id_vocabulary in vocabulary.items())
    ):
        return None

    if (
        not isinstance(start_of_word_token, (str, type(None)))
        or not isinstance(end_of_word_token, (str, type(None)))
        or not isinstance(unknown_token, str)
    ):
        return None

    encoded_sequence = []

    for word in original_text.split():
        prepared_word = prepare_word(word, start_of_word_token, end_of_word_token)
        if prepared_word is None:
            return None

        tokenized_word = tokenize_word(prepared_word, vocabulary, end_of_word_token, unknown_token)
        if tokenized_word is None:
            return None

        encoded_sequence.extend(tokenized_word)

    return encoded_sequence


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extract n-grams from the given sequence.

    Args:
        text (str): Original text
        order (int): Required number of elements in a single n-gram

    Returns:
        list[tuple[str, ...]] | None: A sequence of n-grams

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(text,str)
        or not isinstance(order,int)
        or order <= 0
    ):
        return None

    prepared_ngrams = []
    for i in range(len(text)- order +1):
        ngram = tuple(text[i:i+order])
        prepared_ngrams.append(ngram)

    return prepared_ngrams


def calculate_precision(
    actual: Sequence[tuple[str, ...]], reference: Sequence[tuple[str, ...]]
) -> float | None:
    """
    Compare two sequences by virtue of Precision metric.

    Args:
        actual (Sequence[tuple[str, ...]]): Predicted sequence of n-grams
        reference (Sequence[tuple[str, ...]]): Expected sequence of n-grams

    Returns:
        float | None: Value of Precision metric

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(actual, Sequence) or not isinstance(reference, Sequence):
        return None

    if not all(
        isinstance(ngram, tuple)
        and all(isinstance(letter, str ) for letter in ngram)
        for ngram in actual
    ):
        return None

    if not all(
        isinstance(ngram, tuple)
        and all(isinstance(letter, str) for letter in ngram)
        for ngram in reference
    ):
        return None


    if len(actual) == 0:
        return 0.0

    if len(reference) == 0:
        return 0.0

    count = 0
    for ngram in actual:
        if ngram in reference:
            count += 1
    count = len(set(actual) & (set(reference)))

    return count/len(set(actual))


def calculate_geo_mean(precisions: Sequence[float], max_order: int) -> float | None:
    """
    Compute geometric mean of sequence of values.

    Args:
        precisions (Sequence[float]): A sequence of Precision values
        max_order (int): Maximum length of n-gram considered

    Returns:
        float | None: A value of geometric mean of Precision metric

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(precisions, Sequence)
        or len(precisions) == 0.0
        or not all(isinstance(p, (int,float)) for p in precisions)
        or not isinstance(max_order, int)
    ):
        return None

    sum_log = 0.0
    if any(p <= 0 for p in precisions):
        return 0.0

    for precision in precisions:
        sum_log += log(precision)

    return exp(sum_log / max_order)

def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compare two sequences by virtue of BLEU metric.

    Args:
        actual (str | None): Predicted sequence
        reference (str): Expected sequence
        max_order (int, optional): Max length of n-gram to consider for comparison. Defaults to 3

    Returns:
        float | None: A value of BLEU metric

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if (
        not isinstance(actual,str)
        or not isinstance(reference,str)
        or not isinstance(max_order, int)
    ):
        return None

    prec = []

    for order in range(1,max_order +1):
        actual_ngram = collect_ngrams(actual,order)
        reference_ngram = collect_ngrams(reference,order)


        if actual_ngram is None or reference_ngram is None:
            return None

        precision = calculate_precision(
            actual_ngram,
            reference_ngram
        )
        if precision is None:
            return None

        prec.append(precision)

    geo_mean_calculated = calculate_geo_mean(prec,max_order)

    if geo_mean_calculated is None:
        return None

    return geo_mean_calculated*100
