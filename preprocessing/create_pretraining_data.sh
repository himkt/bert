
split -l 2000000 --additional-suffix .split data/jawiki-20181201-pages-articles.bert.txt data/

for method in unigram bpe; do
  for vocab_size in 08000 16000 32000 64000; do
    python3 create_pretraining_data.py \
      --model_file ./model/sentencepiece/jawiki-pages-articles.20181201.$vocab_size.$method.model \
      --vocab_file ./model/sentencepiece/jawiki-pages-articles.20181201.$vocab_size.$method.vocab \
      --input_file $(python3 ./preprocessing/get_files.py data) \
      --output_file $(python3 ./preprocessing/get_files.py --output data)
  done
done
