"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable
from lab_2_tokenize_by_bpe.main import (
    collect_frequencies,
    decode,
    get_vocabulary,
    train,
)


def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8") as text_file:
        text_reference_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8") as text_file:
        translation_encoded_raw = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        secret_1 = text_file.read()
    result = None


    #mark 4
    result = collect_frequencies(text, start_of_word = None, end_of_word = "</s>")
    assert result, "Frequencies not collected"
    print(result)

    #mark 6
    machine_train = train(result, 100)
    assert machine_train, "Training not working"
    vocabulary = get_vocabulary(machine_train,'<unk>')
    assert vocabulary, "Vocabulary not working"
    secret_1 =[int(number) for number in secret_1.split()]
    secret_1_encoded = decode(secret_1, vocabulary, '</s>')
    assert secret_1_encoded, "Decoding not working"
    print(secret_1_encoded)



    assert result, "Translation not working"


if __name__ == "__main__":
    main()
