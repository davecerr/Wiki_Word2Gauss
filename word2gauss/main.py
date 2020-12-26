import numpy as np
import os
import pickle as pkl
import time
import gzip
import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gzip import GzipFile
from collections import defaultdict, Counter
from tqdm import tqdm
from embeddings import GaussianEmbedding #, iter_pairs
from words import Vocabulary, iter_pairs
from utils import cosine

######################### SETTINGS ############################################

save_on = False

# War & Peace (MWE = 1) vs Wikipedia (MWE = 0)
MWE = 1

# embedding properties
dimension = 50
cov_type = 'diagonal'
E_type = 'KL'

# Gaussian initialisation (random noise is added to these)
mu0 = 0.1
sigma_mean0 = 0.5
sigma_std0 = 1.0

# Gaussian bounds (avoid e.g. massive sigmas to get good overlap)
mu_max = 2.0
sigma_min = 0.7
sigma_max = 1.5

# training properties
num_workers = 6
num_epochs = 10
iteration_verbose_flag = False

eta = 0.1 # learning rate : pass float for global learning rate (no min) or dict with keys mu,sigma,mu_min,sigma_min (local learning rate for each)
Closs = 0.1 # regularization parameter in max-margin loss

if MWE == 1:
    report_schedule = 1000
else:
    report_schedule = 1000000
    max_list_length = 6203

###############################################################################








def _open_file(filename):
    max_len = 0
    with gzip.open(filename) as infile:
        for _, line in enumerate(infile):
            #curr_len = len(list(line.split(",")))
            #if curr_len > max_len:
                #max_len = curr_len
                #max_list = list(line.split(","))
            yield json.loads(line)
    #print("Maximum list length = {}".format(max_len))
    #print("Maximum list = {}".format(max_list))
def tokenizer(s):
    '''
    Whitespace tokenizer
    '''
    if MWE:
        return s.lower().replace(".", "").replace(",", "").replace(":", "").replace("--"," ").replace("-"," ").replace(";", "").replace("'s","").replace("'","").replace("!","").replace('"','').replace("?","").replace("(","").replace(")","").strip().split()
    else:
        return s.lower().replace(".", "").replace(",", "").replace(":", "").replace(";", "").strip().split()

def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))





######################### LOAD DATA ###########################################

if MWE == 1:
    filename = 'war_and_peace.txt'
    with open(filename, 'r') as file:
        data = tokenizer(file.read().replace('\n', ' '))
else:
    print("\n\n----------- LOADING DATA ----------")
    if os.path.exists("data_list.pkl"):
    #     start = time.time()
    #     print("loading from existing pickle")
    #     pickle_in = open("data_list.pkl","rb")
    #     data_list = pkl.load(pickle_in)
    #     end = time.time()
    #     print("loaded in {} secs".format(round(end - start,2)))
    # else:
        print("loading from gzip files")
        files = []
        for _, _, fs in os.walk("data/", topdown=False):
            files += [f for f in fs if f.endswith(".gz")]

        files = [os.path.join("data/page_dist_training_data/", f) for f in files]
        data_list = []
        for i, file in tqdm(enumerate(files)):
            if MWE == 1:
                sentences = list(_open_file(file))
                data_list += sentences
            else:
                sentence
        # pickle_out = open("data_list.pkl","wb")
        # pkl.dump(data_list, pickle_out)
        # pickle_out.close()

    print("LOADING TO STRING")
    lst = []
    # data_list = data_list[:10]
    for item in tqdm(data_list):
      # print(item)
      lst.append(listToString(item))
    corpus = listToString(lst)
    data = tokenizer(corpus)


# print(data)


################################################################################









entity_2_idx = defaultdict(lambda: len(entity_2_idx))
counter = Counter()
dataset = []

for entity in tqdm(data):
    entity_2_idx[entity]
    counter[entity_2_idx[entity]] += 1
    dataset.append(entity_2_idx[entity])

# print(entity_2_idx)
num_tokens = len(entity_2_idx)
print("num_tokens = {}".format(num_tokens))


# print(entity_2_idx)
# print("\n\n")
# print(counter)
# print("\n\n")
# print(dataset)
dataset_length = len(dataset)
print("Dataset length = {}".format(dataset_length))

#### OLD ####

# load the vocabulary
vocab = Vocabulary(entity_2_idx,tokenizer)
# print(vocab)
# create the embedding to train
# use 100 dimensional spherical Gaussian with KL-divergence as energy function

# embed = GaussianEmbedding(num_tokens, dimension,
#     covariance_type=cov_type, energy_type=E_type)

embed = GaussianEmbedding(N=num_tokens, size=dimension,
          covariance_type=cov_type, energy_type=E_type,
          mu_max=mu_max, sigma_min=sigma_min, sigma_max=sigma_max,
          init_params={'mu0': mu0,
              'sigma_mean0': sigma_mean0,
              'sigma_std0': sigma_std0},
          eta=eta, Closs=Closs,
          iteration_verbose_flag=iteration_verbose_flag)


# open the corpus and train with 8 threads
# the corpus is just an iterator of documents, here a new line separated
# gzip file for example

epoch_losses = []
for e in range(num_epochs):
    print("---------- EPOCH {} ----------".format(e+1))
    if MWE == 1:
        with open(filename, 'r') as corpus:
            epoch_losses.append(embed.train(iter_pairs(corpus, vocab), n_workers=num_workers, report_interval=report_schedule))
    else:
        epoch_losses.append(embed.train(iter_pairs(corpus, vocab), n_workers=num_workers, report_interval=report_schedule))

print("EPOCH LOSSES : {}".format(epoch_losses))


print("---------- INITIAL EMBEDDING MEANS ----------")
print(embed.mu)
print("---------- INITIAL EMBEDDING COVS ----------")
print(embed.sigma)

print("---------- FINAL EMBEDDING MEANS ----------")
print(embed.mu)
print("---------- FINAL EMBEDDING COVS ----------")
print(embed.sigma)

sigma_norms = np.linalg.norm(embed.sigma, axis=1)
most_general_indices = np.split(sigma_norms,2)[0].argsort()[-10:][::-1]
most_specific_indices = np.split(sigma_norms,2)[0].argsort()[:10]

idx_2_entity = {v: k for k, v in entity_2_idx.items()}

print("MOST GENERAL ENTITIES")
for idx in most_general_indices:
    print(idx_2_entity[idx])

print("MOST SPECIFIC ENTITIES")
for idx in most_specific_indices:
    print(idx_2_entity[idx])

# save the model for later
if save_on:
    print("SAVING MODEL")
    embed.save('model_file_location_{}'.format(dimension), vocab=vocab.id2word, full=True)

# print("LOADING MODEL")
# test = GaussianEmbedding(N=num_tokens, size=dimension,
#           covariance_type=cov_type, energy_type=E_type,
#           mu_max=mu_max, sigma_min=sigma_min, sigma_max=sigma_max,
#           init_params={'mu0': mu0,
#               'sigma_mean0': sigma_mean0,
#               'sigma_std0': sigma_std0},
#           eta=eta, Closs=Closs)
#
# test.load('model_file_location')


print("FINDING NEAREST NEIGHBOURS")

target = "war"
metric = cosine
num = 10

target_idx = entity_2_idx[target]
neighbours = embed.nearest_neighbors(target=target_idx, metric=metric, num=num+1, vocab=vocab,
                  sort_order='similarity')

print("\n\n")
print("Target = {}".format(target))
print("Similarity metric = {}".format(metric))
for i in range(1,num+1):
    print("{}: {}".format(i,neighbours[i]))
    # print("rank {}: word = {}, sigma = {}, id = {}, similarity = {}".format(i,neighbours[i][word],neighbours[i][sigma],neighbours[i][id],neighbours[i][similarity]))
