modelscope download --dataset ddzhu123/seq-monkey --local_dir data
tar -xvf data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 -C data
rm -rf data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2
wget https://hf-mirror.com/datasets/BelleGroup/train_3.5M_CN/resolve/main/train_3.5M_CN.json -O data/train_3.5M_CN.json