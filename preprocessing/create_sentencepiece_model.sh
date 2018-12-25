rm -rf data
mkdir data

for method in unigram bpe; do
  for vocab_size in 08000 16000 32000 64000; do
    time spm_train --input=./data/jawiki-20181201-pages-articles.txt \
      --control_symbols=[PAD], [CLS], [SEP], [MASK]  --vocab_size $vocab_size  \
      --model_prefix=./model/sentencepiece/spm.20181201.$vocab_size.$method \
      --model_type $method
  done
done
