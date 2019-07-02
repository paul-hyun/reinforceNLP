import sys
import argparse
import fileinput

import config
import data
import transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def exec_model(config, vocab_enc, vocab_dec, vocab_dec_bw, model, line):
    enc_inputs = data.build_data(vocab_enc, [line], config.n_enc_seq)
    dec_inputs = data.build_data(vocab_enc, [""], config.n_dec_seq)
    labels = dec_inputs[:, 1:]
    enc_inputs = enc_inputs[:, :-1]
    dec_inputs = dec_inputs[:, :-1]

    enc_inputs = torch.tensor(enc_inputs, dtype=torch.long).to(config.device)
    dec_inputs = torch.tensor(dec_inputs, dtype=torch.long).to(config.device)
    for i in range(config.n_dec_seq):
        dec_logits, _, _, _ = model(enc_inputs, dec_inputs)
        _, index = dec_logits.max(dim=2)
        dec_inputs[0][i + 1] = index[0][i]
        if index[0][i] == vocab_dec["<eos>"]:
            break
    output = []
    for line in dec_inputs:
        out_line = []
        for i in line:
            i = i.item()
            if i == vocab_dec["<bos>"]:
                pass
            elif i == vocab_dec["<eos>"]:
                break
            else:
                out_line.append(vocab_de_bw[i])
        output.append(out_line)
    print(output)


def run_model(config, vocab_enc, vocab_dec, vocab_de_bw, file):
    model = transformer.Transformer(config)
    model.to(config.device)
    model.load_state_dict(torch.load(file))

    sys.stdout.write("enter: ")
    sys.stdout.flush()
    for line in fileinput.input():
        line = line.strip()
        if 0 < len(line):
            exec_model(config, vocab_enc, vocab_dec, vocab_de_bw, model, line)

        sys.stdout.write("enter: ")
        sys.stdout.flush()


if __name__ == "__main__":
    config = config.Config.load("config.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("lang", choices=['en', 'de'], default='en', const='en', nargs='?')
    args = parser.parse_args()

    vocab_en = data.load_vocab("data/pickle.vocab.en")
    train_en, valid_en, test_en = data.load_data("data/pickle.data.en")
    vocab_de = data.load_vocab("data/pickle.vocab.de")
    train_de, valid_de, test_de = data.load_data("data/pickle.data.de")

    # device and pad set
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.i_pad = vocab_en["<pad>"]

    if args.lang == "en":
        vocab_de_bw = {}
        for key, value in vocab_de.items():
            vocab_de_bw[value] = key
    
        config.n_enc_vocab = len(vocab_en)
        config.n_dec_vocab = len(vocab_de)
        config.n_enc_seq = len(train_en[0])
        config.n_dec_seq = len(train_de[0])

        run_model(config, vocab_en, vocab_de, vocab_de_bw, "data/ckpoint.en-de")
    else:
        vocab_en_bw = {}
        for key, value in vocab_en.items():
            vocab_en_bw[value] = key

        config.n_enc_vocab = len(vocab_de)
        config.n_dec_vocab = len(vocab_en)
        config.n_enc_seq = len(train_de[0])
        config.n_dec_seq = len(train_en[0])

        run_model(config, vocab_de, vocab_en, vocab_en_bw, "data/ckpoint.de-en")

