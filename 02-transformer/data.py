import sys
import collections
import numpy as np
import pickle


# _, bert = nlp.model.bert_12_768_12(dataset_name='wiki_multilingual_uncased', pretrained=False, root='./data')
# tokenizer = nlp.data.BERTTokenizer(vocab=bert)


def tokenize(text):
    return text.strip().split()
    # return tokenizer(text)


def build_data(vocab, lines, length):
    text = []
    for line in lines:
        tokens = tokenize(line)
        text.append(tokens)
    
    return text_to_data(vocab, text, length)


def build_text(file):
    text = []
    length = 0
    with open(file) as f:
        for line in f:
            tokens = tokenize(line)
            text.append(tokens)
            length = max(length, len(tokens))
    
    return text, length


def build_vocab(texts):
    tokens = []
    for text in texts:
        for line in text:
            tokens.extend(line)

    counter = collections.Counter(tokens)
    vocab = { "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3 }
    index = 4
    for key, _ in counter.items():
        vocab[key] = index
        index += 1
    return vocab


def text_to_data(vocab, text, length):
    data = []
    for line in text:
        line_data = []
        line_data.append(vocab["<bos>"])
        for token in line:
            if token in vocab:
                line_data.append(vocab[token])
            else:
                line_data.append(vocab["<unk>"])
        line_data.append(vocab["<eos>"])
        line_data.extend([vocab["<pad>"]] * (length - len(line_data)))
        data.append(line_data)

    return np.array(data)


def dump_data(train, valid, test, save_vocab, save_data):
    train_text, train_len = build_text(train)
    valid_text, valid_len = build_text(valid)
    test_text, test_len = build_text(test)
    length = max(train_len, valid_len, test_len)

    vocab = build_vocab([train_text, valid_text, test_text])

    train = np.array(text_to_data(vocab, train_text, length + 2))
    valid = np.array(text_to_data(vocab, valid_text, length + 2))
    test = np.array(text_to_data(vocab, test_text, length + 2))

    with open(save_vocab, 'wb') as f:
        pickle.dump((vocab), f)
        print("save vocab to %s" % save_vocab)

    with open(save_data, 'wb') as f:
        pickle.dump((train, valid, test), f)
        print("save data to %s" % save_data)


def load_vocab(file):
    with open(file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_data(file):
    with open(file, 'rb') as f:
        train, valid, test = pickle.load(f)
    return train, valid, test


if __name__ == "__main__":
    dump_data("data/train.en.atok", "data/val.en.atok", "data/test.en.atok", "data/pickle.vocab.en", "data/pickle.data.en")
    dump_data("data/train.de.atok", "data/val.de.atok", "data/test.de.atok", "data/pickle.vocab.de", "data/pickle.data.de")

