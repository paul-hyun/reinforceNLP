import config
import data
import transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# 참고: https://github.com/jadore801120/attention-is-all-you-need-pytorch


def build_loader(enc_inputs, dec_inputs, device, batch_size):
    enc_inputs = torch.LongTensor(enc_inputs).to(device)
    dec_inputs = torch.LongTensor(dec_inputs).to(device)
    dataset = torch.utils.data.TensorDataset(enc_inputs, dec_inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def eval_model(config, model, loader):
    n_word_total = 0
    n_word_correct = 0

    model.eval()
    with torch.no_grad():
        for i, value in enumerate(loader, 0):
            batch_enc_inputs, batch_dec_inputs = value
            batch_labels = batch_dec_inputs[:, 1:]
            batch_enc_inputs = batch_enc_inputs[:, :-1]
            batch_dec_inputs = batch_dec_inputs[:, :-1]

            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = model(batch_enc_inputs, batch_dec_inputs)

            non_pad_mask = batch_labels.ne(config.i_pad)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            _, index = dec_logits.max(dim=2)
            n_correct = index.eq(batch_labels)
            n_correct = n_correct.masked_select(non_pad_mask).sum().item()
            n_word_correct += n_correct

    return n_word_total, n_word_correct


def train_model(config, vocab_enc, vocab_dec, train_loader, valid_loader, test_loader, save):
    model = transformer.Transformer(config)
    model.to(config.device)

    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='sum')
    # 기본 Adam Optimizer로는 학습이 안됨
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = transformer.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        config.d_embed, 4000)

    epochs = []
    valid_score = []
    test_score = []

    min_loss = 999999
    max_dev = 0
    max_test = 0
    for epoch in range(config.n_epoch):
        epochs.append(epoch + 1)

        train_loss = 0
        n_word_total = 0
        n_word_correct = 0

        model.train()
        for i, value in enumerate(train_loader, 0):
            batch_enc_inputs, batch_dec_inputs = value
            batch_labels = batch_dec_inputs[:, 1:]
            batch_enc_inputs = batch_enc_inputs[:, :-1]
            batch_dec_inputs = batch_dec_inputs[:, :-1]

            optimizer.zero_grad()

            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = model(batch_enc_inputs, batch_dec_inputs)
            # print("{} : {}".format(batch_labels.size(), dec_logits.size()))
            # torch.Size([64, 235]) : torch.Size([64, 235, 21186])
            loss = loss_fn(dec_logits.view(-1, dec_logits.size(2)), batch_labels.contiguous().view(-1))
            loss.backward()
            # optimizer.step()
            optimizer.step_and_update_lr()
        
            train_loss += loss.item()

            non_pad_mask = batch_labels.ne(config.i_pad)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            _, index = dec_logits.max(dim=2)
            n_correct = index.eq(batch_labels)
            n_correct = n_correct.masked_select(non_pad_mask).sum().item()
            n_word_correct += n_correct
        train_loss = train_loss / len(train_loader)
        print("Training [%2d], loss: %.3f, acc: %.3f" % (epoch + 1, train_loss, n_word_correct * 100 / n_word_total))

        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), save)
            print("model saved %s" % (save))

        n_word_total = 0
        n_word_correct = 0

        n_word_total, n_word_correct = eval_model(config, model, valid_loader)
        print("Validation [%2d], acc: %.3f" % (epoch + 1, n_word_correct * 100 / n_word_total))
        n_word_total, n_word_correct = eval_model(config, model, test_loader)
        print("Test [%2d], acc: %.3f" % (epoch + 1, n_word_correct * 100 / n_word_total))
    del model


if __name__ == "__main__":
    config = config.Config.load("config.json")

    vocab_en = data.load_vocab("data/pickle.vocab.en")
    train_en, valid_en, test_en = data.load_data("data/pickle.data.en")
    vocab_de = data.load_vocab("data/pickle.vocab.de")
    train_de, valid_de, test_de = data.load_data("data/pickle.data.de")

    # device and pad set
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.i_pad = vocab_en["<pad>"]

    # change config for en to de
    config.n_enc_vocab = len(vocab_en)
    config.n_dec_vocab = len(vocab_de)
    config.n_enc_seq = len(train_en[0])
    config.n_dec_seq = len(train_de[0])

    print(config)
    train_loader = build_loader(train_en, train_de, config.device, config.n_batch)
    valid_loader = build_loader(valid_en, valid_de, config.device, config.n_batch)
    test_loader = build_loader(test_en, test_de, config.device, config.n_batch)
    train_model(config, vocab_en, vocab_de, train_loader, valid_loader, test_loader, "data/ckpoint.en-de")

    # change config for de to en
    config.n_enc_vocab = len(vocab_de)
    config.n_dec_vocab = len(vocab_en)
    config.n_enc_seq = len(train_de[0])
    config.n_dec_seq = len(train_en[0])

    print(config)
    train_loader = build_loader(train_de, train_en, config.device, config.n_batch)
    valid_loader = build_loader(valid_de, valid_en, config.device, config.n_batch)
    test_loader = build_loader(test_de, test_en, config.device, config.n_batch)
    train_model(config, vocab_en, vocab_de, train_loader, valid_loader, test_loader, "data/ckpoint.de-en")



    

