import sys
import json
import tqdm
import pathlib


if __name__ == '__main__':
    sys.path.append('./')
    from tiny_tokenizer.word_tokenizer import WordTokenizer
    from tiny_tokenizer.sentence_tokenizer import SentenceTokenizer
    wikipextractor_dir = pathlib.Path('./data/jawiki-20181201')
    sentence_tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()
    wakati_fpath = './data/jawiki-20181201-pages-articles.bert.txt'

    wakati_file = open(wakati_fpath, 'w')
    json_fpath_list = list(wikipextractor_dir.glob('*/wiki_*'))

    num_iters = 0
    for json_fpath in tqdm.tqdm(json_fpath_list):
        for json_str in open(json_fpath):
            if num_iters > 0:
                print(file=wakati_file)

            try:
                num_iters += 1
                json_object = json.loads(json_str)
                document = json_object['text']
                sentences = sentence_tokenizer.tokenize(document)
                for t, sentence in enumerate(sentences):
                    if t == 0:
                        continue  # t=0 represents title of a text

                    print(word_tokenizer.tokenize(sentence), file=wakati_file)
            except Exception as e:
                print(e, file=sys.stderr)

    wakati_file.close()
