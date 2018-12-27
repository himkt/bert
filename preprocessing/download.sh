wget https://dumps.wikimedia.org/jawiki/20181201/jawiki-20181201-pages-articles.xml.bz2 -O ./data/jawiki-20181201-pages-articles.xml.bz2
WikiExtractor.py ./data/jawiki-20181201-pages-articles.xml.bz2 -o data/jawiki-20181201 --json
python3 preprocessing/tokenize_document.py --bert
