# 02-transformer
Transformer 네트워크

## prepare data
- 참고: https://github.com/jadore801120/attention-is-all-you-need-pytorch#usage
- mkdir data
- 


## make vocab (en)
- python data.py
- result: in 02-transformer/data dir [pickle.vocab.ko, pickle.data.ko, pickle.vocab.en, pickle.data.en]


## train
- python train.py
- result: in 02-transformer/data dir [ckpoint.ko-en, ckpoint.en-ko]


## run
- python run.py en|ko

