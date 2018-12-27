from tiny_tokenizer.word_tokenizer import WordTokenizer
from tiny_tokenizer.sentence_tokenizer import SentenceTokenizer

import argparse
import sys
import json
import tqdm
import pathlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', action='store_true')
    args = parser.parse_args()

    wikipextractor_dir = pathlib.Path('./data/jawiki-20181201')
    sentence_tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()

    if args.bert:
        output_fpath = './data/jawiki-20181201-pages-articles.bert.txt'
    else:
        output_fpath = './data/jawiki-20181201-pages-articles.spm.txt'

    output_file = open(output_fpath, 'w')
    json_fpath_list = list(wikipextractor_dir.glob('*/wiki_*'))

    num_iters = 0
    for json_fpath in tqdm.tqdm(json_fpath_list):
        for json_str in open(json_fpath.as_posix()):

            # input of bert requires breakline between documents
            if args.bert and num_iters > 0:
                print(file=output_file)

            try:
                num_iters += 1
                json_object = json.loads(json_str)
                document = json_object['text']
                sentences = sentence_tokenizer.tokenize(document)
                for t, sentence in enumerate(sentences):
                    # first element of sentences is title of a document
                    if t == 0:
                        continue

                    print(word_tokenizer.tokenize(sentence), file=output_file)

            except Exception as e:
                print(e, file=sys.stderr)

    output_file.close()
