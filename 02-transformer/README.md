# 02-transformer
Transformer 네트워크

## prepare data
- mkdir data
- wget https://raw.githubusercontent.com/haven-jeon/ko_en_neural_machine_translation/master/korean_parallel_corpora/korean-english-v1/korean-english-park.train.ko
- wget https://raw.githubusercontent.com/haven-jeon/ko_en_neural_machine_translation/master/korean_parallel_corpora/korean-english-v1/korean-english-park.train.en
- wget https://raw.githubusercontent.com/haven-jeon/ko_en_neural_machine_translation/master/korean_parallel_corpora/korean-english-v1/korean-english-park.dev.ko
- wget https://raw.githubusercontent.com/haven-jeon/ko_en_neural_machine_translation/master/korean_parallel_corpora/korean-english-v1/korean-english-park.dev.en
- wget https://raw.githubusercontent.com/haven-jeon/ko_en_neural_machine_translation/master/korean_parallel_corpora/korean-english-v1/korean-english-park.test.ko
- wget https://raw.githubusercontent.com/haven-jeon/ko_en_neural_machine_translation/master/korean_parallel_corpora/korean-english-v1/korean-english-park.test.en


## make vocab (en)
- python data.py
- result: in 02-transformer/data dir [pickle.vocab.ko, pickle.data.ko, pickle.vocab.en, pickle.data.en]


## train
- python train.py
- result: in 02-transformer/data dir [ckpoint.ko-en, ckpoint.en-ko]


## run
- python run.py en|ko

