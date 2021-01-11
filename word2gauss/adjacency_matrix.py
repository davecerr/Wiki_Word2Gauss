import numpy as np
import gzip
import time
import tqdm
import json
import os

def _open_file(filename):
    max_len = 0
    with gzip.open(filename) as infile:
        for _, line in enumerate(infile):
            #curr_len = len(list(line.split(",")))
            #if curr_len > max_len:
                #max_len = curr_len
                #max_list = list(line.split(","))
            yield json.loads(line)

# load files
files = []
for _, _, fs in os.walk("data_title/", topdown=False):
  files += [f for f in fs if f.endswith(".gz")]

files = [os.path.join("data_title/", f) for f in files]
print("files found")

# load data from files
with gzip.open(files[0], 'r') as fin:        # 4. gzip
    json_bytes = fin.read()                  # 3. bytes (i.e. UTF-8)

json_str = json_bytes.decode('utf-8')       # 2. string (i.e. JSON)
data = json.loads(json_str)                      # 1. data

print(data)
