"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code


def main() -> None:
    """
    Launches an implementation.
    """
    from main import (
        calculate_frequencies,
        get_top_n_words,
        remove_stop_words,
        tokenize,
        create_language_profile,
        detect_language_by_top_n,

    )

    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()

    # Практическое задание mark 4
    tokens = tokenize(de_text)
    tokens_without_stopwords = remove_stop_words(tokens, stopwords)
    freq_dict = calculate_frequencies(tokens_without_stopwords)
    result = get_top_n_words(freq_dict, 7)
    print(result)

    # Практическое задание mark 6
    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    unknown_profile = create_language_profile("unknown", unknown_text, stopwords)
    result = detect_language_by_top_n(unknown_profile, de_profile, en_profile, 15)
    print(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()