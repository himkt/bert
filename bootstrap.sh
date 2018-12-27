bash ./preprocessing/download.sh
python3 ./preprocessing/tokenize_document.py --bert
bash ./preprocessing/create_pretraining_data.sh
