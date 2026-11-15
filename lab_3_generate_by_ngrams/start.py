"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable
from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor, BeamSearchTextGenerator


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()


    #mark 4
    text_processor = TextProcessor("_")

    encoded_corpus = text_processor.encode(text)
    print (f"Encoded text (100 numbers):\n {encoded_corpus[:100]}")
    assert encoded_corpus, "Text can not be encoded"

    decoded_corpus = text_processor.decode(encoded_corpus)
    print (f"Decoded text (150 chars):\n {decoded_corpus[:150]}")
    assert decoded_corpus, "Text can not be decoded"

    #mark 6
    language_model = NGramLanguageModel(encoded_corpus, n_gram_size=7)

    build_model = language_model.build()
    assert build_model, "Model can not be built"

    generator = GreedyTextGenerator(language_model, text_processor)
    result = generator.run(prompt="Vernon", seq_len=51)
    print (f"Generated text:\n {result}")
    assert result, "Text can not be generated"

    #mark 8
    beam_generator = BeamSearchTextGenerator(language_model, text_processor, beam_width=7)
    result = beam_generator.run(prompt="Vernon", seq_len=56)
    print (f"Generated text with beam search:\n {result}")
    assert result, "Text can not be generated with beam search"

if __name__ == "__main__":
    main()
