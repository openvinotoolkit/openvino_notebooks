#!/usr/bin/python3

import os
import urllib.request

MRPC_DEV_IDS_URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc'
MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'


data_dir = './'
print("Processing MRPC...")
mrpc_dir = os.path.join(data_dir, "MRPC")
if not os.path.isdir(mrpc_dir):
    os.mkdir(mrpc_dir)
print("Local MRPC data not specified, downloading data from %s" % MRPC_TRAIN)
mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
urllib.request.urlretrieve(MRPC_DEV_IDS_URL, os.path.join(mrpc_dir, "dev_ids.tsv"))

dev_ids = []
with open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
    for row in ids_fh:
        dev_ids.append(row.strip().split('\t'))

with open(mrpc_train_file, encoding="utf8") as data_fh, \
     open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding="utf8") as dev_fh:
    header = data_fh.readline()
    dev_fh.write(header)
    for row in data_fh:
        label, id1, id2, s1, s2 = row.strip().split('\t')
        if [id1, id2] in dev_ids:
            dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

print("\tCompleted!")

