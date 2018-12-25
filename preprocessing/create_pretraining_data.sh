rm -rf data
mkdir data

for method in unigram bpe; do
  for vocab_size in 008k 016k 032k 064k; do
    echo python create_pretraining_data.py --model_file ../tiny_tokenizer/data/sentencepiece/spm.20181201.$vocab_size.$method.model --vocab_file \
      ../tiny_tokenizer/data/sentencepiece/spm.20181201.$vocab_size.$method.vocab \
      --input_file ../tiny_tokenizer/data/jawiki-20181201-pages-articles.bert.txt \
      --output_file data/pretraining.jawiki-20181201.$vocab_size.$method.bert.txt
  done
  # wait
done
