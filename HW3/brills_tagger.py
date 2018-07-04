import numpy as np
from nltk.tag import brill_trainer, brill, str2tuple, UnigramTagger


def read_txt(filename):
    """Reading input file"""
    with open(filename, encoding="iso8859_2") as f:
        text = f.read()
        tokens = text.split()
        print("Tokens count", len(tokens))
        return tokens


def split_sentences(tokens):
    """Splitting data into sentences"""
    sentences = [[str2tuple(token) for token in sent.split("\n") if token != ""]
                 for sent in "\n".join(tokens).split("###/###")]
    return [sentence for sentence in sentences if sentence != []]


def train_evaluate_brills(train_data, test_data):
    """Training and evaluating of Brill`s tagger"""
    # Define templates for rules, provided by nltk
    brill.Template._cleartemplates()
    templates = brill.fntbl37()
    # Define initial tagger, tagging by the most common tag
    initial_tagger = UnigramTagger(train_data)
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger=initial_tagger,   # better unk words handling
                                               templates=templates, trace=3,
                                               deterministic=True)
    tagger = trainer.train(train_data, max_rules=100)   # max number of rules to learn 100
    print("Accuracy:", tagger.evaluate(test_data))
    return tagger.evaluate(test_data)


def run_different_splits(text):
    """Running algorithm for different splits of the data into test, heldout and train"""
    print("=" * 30)
    print(text)
    print("=" * 30)

    tokens = read_txt(text)
    accuracies = []

    # T = rest, S = last 40, H = pre-last 20
    T = split_sentences(tokens[:-60000])
    S = split_sentences(tokens[-40000:])
    accuracies.append(train_evaluate_brills(T, S))

    # T = rest, S = first 40, H = after-first 20
    T = split_sentences(tokens[60000:])
    S = split_sentences(tokens[:40000])
    accuracies.append(train_evaluate_brills(T, S))

    # T = rest, S = first 40, H = last 20
    T = split_sentences(tokens[40000:-20000])
    S = split_sentences(tokens[:40000])
    accuracies.append(train_evaluate_brills(T, S))

    # T = rest, S = last 40, H = first 20
    T = split_sentences(tokens[20000:-40000])
    S = split_sentences(tokens[-40000:])
    accuracies.append(train_evaluate_brills(T, S))

    # T = rest, S = after-first 40, H = first 20
    T = split_sentences(tokens[60000:])
    S = split_sentences(tokens[20000:60000])
    accuracies.append(train_evaluate_brills(T, S))

    print("Mean accuracy:", np.mean(accuracies))
    print("Standard deviation of accuracy:", np.std(accuracies))


if __name__ == "__main__":
    cz_text = "textcz2.ptg.txt"
    en_text = "texten2.ptg.txt"

    run_different_splits(cz_text)
    run_different_splits(en_text)
