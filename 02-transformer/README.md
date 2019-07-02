# 02-transformer
Transformer 네트워크

## prepare data
- 참고: https://github.com/jadore801120/attention-is-all-you-need-pytorch#usage
- data file data/[train.en.atok, val.en.atok, test.en.atok, train.de.atok, val.de.atok, test.de.atok]


## make data (pickle)
- python data.py
- result: in 02-transformer/data dir [pickle.vocab.en, pickle.data.en, pickle.vocab.de, pickle.data.de]


## train
- python train.py
- result: in 02-transformer/data dir [ckpoint.en-de, ckpoint.de-en]


## run
- python run.py en|de

