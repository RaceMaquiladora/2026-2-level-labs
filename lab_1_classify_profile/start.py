"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
from main import calculate_frequencies, get_top_n_words, remove_stop_words, tokenize

def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    result = None
    assert result, "Detection result is None"

    print(tokenize(de_text))
    print(remove_stop_words(tokenize(de_text), stopwords))
    print(calculate_frequencies(remove_stop_words(tokenize(de_text), stopwords)))
    print(get_top_n_words(calculate_frequencies(remove_stop_words(tokenize(de_text), stopwords)), 10))

if __name__ == "__main__":
    main()



