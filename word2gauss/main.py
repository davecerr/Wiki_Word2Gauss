import argparse
import numpy as np
import os
import pickle as pkl
import time
import gzip
import json
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gzip import GzipFile
from collections import defaultdict, Counter
from tqdm import tqdm
from embeddings import GaussianEmbedding #, iter_pairs
from words import Vocabulary, iter_pairs
from utils import cosine, cosine_between_vecs, KL_Multivariate_Gaussians, get_predictions
from scipy.stats import pearsonr, spearmanr
#from lib.wikipedia import standardise_title

######################### SETTINGS ############################################
sys.settrace
save_on = True

# War & Peace (MWE = 1) vs Wikipedia single file (MWE = 2) vs full Wikipedia (MWE = 0)

# embedding properties
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
verbose_pairs=0

# additional parameters if dynamic_window_size = False
window=5
batch_size=10

eta = 0.1 # learning rate : pass float for global learning rate (no min) or dict with keys mu,sigma,mu_min,sigma_min (local learning rate for each)
Closs = 0.1 # regularization parameter in max-margin loss

# validation dataset path
validation_path = "data/WiRe.csv"

# calculation/printing controls
print_init_embeddings = False
print_final_embeddings = False
calc_general_and_specific = False
calc_similarity_example = True
calc_nearest_neighbours = True


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
def tokenizer_MWE1(s):
    '''
    Whitespace tokenizer
    '''
    return s.lower().replace(".", "").replace(",", "").replace(":", "").replace("--"," ").replace("-"," ").replace(";", "").replace("'s","").replace("'","").replace("!","").replace('"','').replace("?","").replace("(","").replace(")","").replace("\n","").strip().split()

def tokenizer_MWE0(s):
    '''
    Whitespace tokenizer
    '''
    return s.lower().replace(".", "").replace(",", "").replace(":", "").replace(";", "").strip().split()

def listToString(s,MWE):
    # initialize an empty string
    str1 = " "
    # return string
    if MWE == 1:
        return str1.join(s)
    else:
        return str1.join(s).encode('ascii', 'ignore')







def parse_args():

    parser = argparse.ArgumentParser(description='Gaussian embedding')

    parser.add_argument('--MWE', type=int, required=True,
                        help='MWE=0: full Wikipedia, MWE=1: War & Peace, MWE=2: single Wikipedia file')
    parser.add_argument('--num_threads', type=int, required=True,
                        help='Number of training threads (integer >= 1)')
    parser.add_argument('--dynamic_window_size', type=bool, required=True,
                        help='Should window adjust to list length (True) or retain a fixed size (False)')
    parser.add_argument('--report_schedule', type=int, required=True,
                        help='Frequency of logging (integer >= 1)')
    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs (integer >= 1)')
    parser.add_argument('--dim', type=int, required=True,
                        help='Dimension of embedding space (integer >= 1)')
    parser.add_argument('--neg_samples', type=int, required=True,
                        help='Number of negative samples for each positive examples (integer >= 1)')
    parser.add_argument('--iteration_verbose_flag', type=bool, default=False,
                        help='Verbose losses')
    args = parser.parse_args()
    return args

def main_script():
    args = parse_args()

    if args.MWE not in [0,1,2]:
        raise self.error('MWE must be 0,1 or 2')


    # set report schedule based on MWE case
    #if args.MWE == 1:
    #    report_schedule = 1000
    #else:
    #    report_schedule = 100
    #    max_list_length = 6203

    ######################### LOAD DATA ###########################################
    if args.MWE == 1:
        filename = 'war_and_peace.txt'
        #with open(filename, 'r') as file:
            #data = tokenizer_MWE1(file.read().replace('\n', ' '))
            #print(data)
        lst = []
        f = open(filename, "r")
        for line in f:
            for word in line.split(" "):
                if word == "\n":
                    continue
                elif "\n" in word:
                    lst.append(word[:-1])
                else:
                    lst.append(word)
        data_string = listToString(lst, args.MWE)
        print("STRING CREATED")
        text_file = open("w_and_p.txt", "w")
        text_file.write(data_string)
        text_file.close()
        print("STRING WRITTEN TO TEXT FILE")
        data = tokenizer_MWE1(data_string)
        print("STRING TOKENIZED")
        #print(data)

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
                if args.MWE == 2:
                    files += [f for f in fs if f.endswith("00000.gz")]
                else:
                    files += [f for f in fs if f.endswith(".gz")]

            files = [os.path.join("data/page_dist_training_data/", f) for f in files]
            data_list = []
            for i, file in tqdm(enumerate(files)):
                sentences = list(_open_file(file))
                data_list += sentences
            # pickle_out = open("data_list.pkl","wb")
            # pkl.dump(data_list, pickle_out)
            # pickle_out.close()

        #if args.MWE == 2:
            #data_list = data_list[:2]


        print("WRITING DATA")
        lst = []
        for entities in tqdm(data_list):
            lst.append(listToString(entities, args.MWE))
            lst.append("\n")
        data_string = listToString(lst, args.MWE)
        print("STRING CREATED")
        text_file = open("wikipedia.txt", "w")
        text_file.write(data_string)
        text_file.close()
        print("STRING WRITTEN TO TEXT FILE")
        data = tokenizer_MWE0(data_string)
        print("STRING TOKENIZED")
        #print(data)

    #print(corpus)
    #print(data)


    ################################################################################

    entity_2_idx = defaultdict(lambda: len(entity_2_idx))
    counter = Counter()
    dataset = []

    print("WRITING ENTITY2IDX DICT")
    for entity in tqdm(data):
        entity_2_idx[entity]
        counter[entity_2_idx[entity]] += 1
        dataset.append(entity_2_idx[entity])

    # print(entity_2_idx)
    num_tokens = len(entity_2_idx)
    print("num_tokens = {}".format(num_tokens))


    #print(entity_2_idx)
    #print("\n\n")
    #print(counter)
    #print("\n\n")
    #print(dataset)
    dataset_length = len(dataset)
    print("Dataset length = {}".format(dataset_length))


    # load the vocabulary
    if args.MWE == 1:
        vocab = Vocabulary(entity_2_idx,tokenizer_MWE1)
    else:
        vocab = Vocabulary(entity_2_idx,tokenizer_MWE0)

    ############################################################################

    # create the embedding to train
    # use 100 dimensional spherical Gaussian with KL-divergence as energy function

    # embed = GaussianEmbedding(num_tokens, dimension,
    #     covariance_type=cov_type, energy_type=E_type)

    embed = GaussianEmbedding(N=num_tokens, size=args.dim,
              covariance_type=cov_type, energy_type=E_type,
              mu_max=mu_max, sigma_min=sigma_min, sigma_max=sigma_max,
              init_params={'mu0': mu0,
                  'sigma_mean0': sigma_mean0,
                  'sigma_std0': sigma_std0},
              eta=eta, Closs=Closs,
              iteration_verbose_flag=args.iteration_verbose_flag)



    ###########################################################################


    # open the corpus and train with 8 threads
    # the corpus is just an iterator of documents, here a new line separated
    # gzip file for example

    if print_init_embeddings:
        print("---------- INITIAL EMBEDDING MEANS ----------")
        print(embed.mu)
        print("---------- INITIAL EMBEDDING COVS ----------")
        print(embed.sigma)



    epoch_losses = []
    for e in range(args.num_epochs):
        print("---------- EPOCH {} ----------".format(e+1))
        if args.MWE == 1:
            with open('w_and_p.txt', 'r') as corpus:
                total_num_examples = len(open('w_and_p.txt').readlines(  ))
                if args.dynamic_window_size:
                    epoch_losses.append(embed.train_dynamic(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, total_num_examples=total_num_examples, verbose_pairs=verbose_pairs, report_interval=args.report_schedule))
                else:
                    epoch_losses.append(embed.train(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, verbose_pairs=verbose_pairs, report_interval=args.report_schedule))
        else:
            with open('wikipedia.txt', 'r') as corpus:
                total_num_examples = len(open('wikipedia.txt').readlines(  ))
                if args.dynamic_window_size:
                    epoch_losses.append(embed.train_dynamic(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, total_num_examples=total_num_examples, verbose_pairs=verbose_pairs, report_interval=args.report_schedule))
                else:
                    epoch_losses.append(embed.train(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, verbose_pairs=verbose_pairs, report_interval=args.report_schedule))

    print("EPOCH LOSSES : {}".format(epoch_losses))



    if print_final_embeddings:
        print("---------- FINAL EMBEDDING MEANS ----------")
        print(embed.mu)
        print("---------- FINAL EMBEDDING COVS ----------")
        print(embed.sigma)



    ############################################################################

    if calc_general_and_specific:
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

    ###########################################################################

    # save the model for later
    if save_on:
        print("SAVING MODEL")
        embed.save('model_MWE={}_d={}_e={}_neg={}'.format(args.MWE,args.dim,args.num_epochs,args.neg_samples), vocab=vocab.id2word, full=True)


    ###########################################################################

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


    ###########################################################################

    if calc_similarity_example:
        print("TESTING KL SIMILARITY")
        entity1 = 'Copenhagen'
        entity2 = 'Denmark'
        idx1 = vocab.word2id(entity1)
        idx2 = vocab.word2id(entity2)
        mu1 = embed.mu[idx1]
        Sigma1 = np.diag(embed.sigma[idx1])
        mu2 = embed.mu[idx2]
        Sigma2 = np.diag(embed.sigma[idx2])
        print("ENTITY 1 : {}".format(entity1))
        #print("mu1 = {}".format(mu1))
        #print("Sigma1 = {}".format(Sigma1))
        print("ENTITY 2 : {}".format(entity2))
        #print("mu2 = {}".format(mu2))
        #print("Sigma2 = {}".format(Sigma2))
        forward_KL_similarity = KL_Multivariate_Gaussians(mu1, Sigma1, mu2, Sigma2)
        reverse_KL_similarity = KL_Multivariate_Gaussians(mu2, Sigma2, mu1, Sigma1)
        print("KL[entity1 || entity2] similarity = {}".format(round(forward_KL_similarity,4)))
        print("KL[entity2 || entity1] similarity = {}".format(round(reverse_KL_similarity,4)))
        print("cosine similarity = {}".format(round(cosine_between_vecs(mu1,mu2),4)))


    ############################################################################

    if calc_nearest_neighbours:
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


    ###########################################################################

    print("MEASURING EMBEDDING PERFORMANCE ON VALIDATION DATA")
    actual, pred_KL_fwd, pred_KL_rev, pred_cos = get_predictions(validation_path, embed, vocab, is_round=False)

    ### forward KL predictions ###
    pear_r_fwd, _ = pearsonr(actual, pred_KL_fwd)
    spear_r_fwd, _ = spearmanr(actual, pred_KL_fwd)
    print("------ FORWARD KL SIMILARITY KL[src||dst] ------")
    print("Pearson R: {},  Spearman R: {}".format(pear_r_fwd, spear_r_fwd))

    ### reverse KL predictions ###
    pear_r_rev, _ = pearsonr(actual, pred_KL_rev)
    spear_r_rev, _ = spearmanr(actual, pred_KL_rev)
    print("------ REVERSE KL SIMILARITY KL[dst||src] ------")
    print("Pearson R: {},  Spearman R: {}".format(pear_r_rev, spear_r_rev))

    ### cosine predictions ###
    pear_r_cos, _ = pearsonr(actual, pred_cos)
    spear_r_cos, _ = spearmanr(actual, pred_cos)
    print("------ COSINE SIMILARITY OF MEANS ------")
    print("Pearson R: {},  Spearman R: {}".format(pear_r_cos, spear_r_cos))




if __name__ == '__main__':
    main_script()

### NOTES ###

# batch size is used in words.py - we can turn on print statements to see it controls how many entity lists are processed at once
# this has no effect on the output since negative samples can be drawn from anywhere in vocab rather than just that batch
# therefore the report_schedule is what the output calls "batch" since we report at the end of each report_schedule/batch
# for this reason batch_size is a default argparser and report_schedule is a required argparser.

# if we make verbose_pairs=1 then we can see the actual pairs being generated. observe that there are window * (window-1) * neg_samples
# examples for each entity. This is because each of the N elements in a list can be paired with each of the N-1 other elements, and then we
# can repeat this for each of the neg_samples. Note that for a list [1,2,3,4,5] these pairings would naturally be
# 12,13,14,15,21,23,24,25,31,32,34,35,41,42,43,45,51,52,53,54 giving a total of 20.
# However, if we print pairs we see that it generates them in the order
# 12,21,13,31,14,41,15,51,23,32,24,42,25,52,34,43,35,53,45,54 giving a total of 20. The benefit of doing it this was is that we only need to make
# a single forward pass through the list. In actual fact, they appear as 12, 12 etc but with a 0/1 flag (5th element of the pair list) to indicate
# which is the focus word i.e. one corresponds to 12 and the other to 21
