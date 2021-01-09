import argparse
import numpy as np
import pandas as pd
import os
import pickle as pkl
import time
import gzip
import json
import sys
import csv
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
np.random.seed(0)

# War & Peace (MWE = 1) vs Wikipedia single file (MWE = 2) vs full Wikipedia (MWE = 0)

# embedding properties
cov_type = 'diagonal'
E_type = 'KL'

# Gaussian initialisation (random noise is added to these)
mu0 = 0.1
sigma_mean0 = 0.5
sigma_std0 = 1.0

# Gaussian bounds (avoid e.g. massive sigmas to get good overlap)
mu_max = 10.0
sigma_min = 0.2
sigma_max = 5

# additional parameters if dynamic_window_size = False
window=5
batch_size=10

#eta = 0.1 # learning rate : pass float for global learning rate (no min) or dict with keys mu,sigma,mu_min,sigma_min (local learning rate for each)
#Closs = 0.1 # regularization parameter in max-margin loss

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


def unpickler(input):
    partial = []
    for line in input:
        partial.append(line)
        if line == '\n':
            obj = ''.join(partial)
            partial = []
            yield pkl.loads(obj)

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



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def parse_args():

    parser = argparse.ArgumentParser(description='Gaussian embedding')

    # minimum working example or full Wikipedia
    parser.add_argument('--MWE', type=int, required=True,
                        help='MWE=0: full Wikipedia, MWE=1: War & Peace, MWE=2: single Wikipedia file, MWE=3: WiRe items')

    # training details
    parser.add_argument('--num_threads', type=int, required=True,
                        help='Number of training threads (integer >= 1)')
    parser.add_argument('--dynamic_window_size', type=str2bool, required=True,
                        help='Should window adjust to list length (True) or retain a fixed size (False)')
    parser.add_argument('--report_schedule', type=int, required=True,
                        help='Frequency of logging (integer >= 1)')
    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs (integer >= 1)')

    # hyperparameters
    parser.add_argument('--dim', type=int, required=True,
                        help='Dimension of embedding space (integer >= 1)')
    parser.add_argument('--neg_samples', type=int, required=True,
                        help='Number of negative samples for each positive examples (integer >= 1)')
    parser.add_argument('--eta', type=float, required=True,
                        help='Learning rate')
    parser.add_argument('--Closs', type=float, required=True,
                        help='Margin size in Hinge Loss')

    # saving model/results
    parser.add_argument('--save', type=str2bool, required=True,
                        help='Save the model (True) or not (False)')
    parser.add_argument('--csv', type=str2bool, required=True,
                        help='Record training results in csv file')

    # debugging
    parser.add_argument('--verbose_loss', type=str2bool, default=False,
                        help='Print losses')
    parser.add_argument('--verbose_gradients', type=str2bool, default=False,
                        help='Print gradients')
    parser.add_argument('--verbose_pairs', type=str2bool, default=False,
                        help='Print positive and negative pairs for each update')

    args = parser.parse_args()
    return args

def main_script():
    args = parse_args()

    print("save = {}".format(args.save))
    print("csv = {}".format(args.csv))

    if args.MWE not in [0,1,2,3]:
        sys.exit('MWE must be 0,1,2 or 3')
    if args.num_threads <= 0:
        sys.exit('num_threads must be a positive integer')
    if args.num_epochs <= 0:
        sys.exit('num_epochs must be a positive integer')
    if args.dim <= 0:
        sys.exit('dim must be a positive integer')
    if args.neg_samples <= 0:
        sys.exit('neg_samples must be a positive integer')
    if args.report_schedule <= 0:
        sys.exit('report_schedule must be a positive integer')


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

    elif args.MWE == 3:
        if os.path.exists("out.gz"):
            print("loading from gzip files")
            file = "out.gz"
            data_list = list(_open_file(file))[0]
            #print(data_list)
        else:
            wire_vocab = set()
            df_wire = pd.read_csv(validation_path)
            for _, record in df_wire.iterrows():
                wire_vocab.add(record["srcWikiTitle"])
                wire_vocab.add(record["dstWikiTitle"])
            wire_vocab = list(wire_vocab)
            print("WiRe vocab loaded successfully")

            files = []
            for _, _, fs in os.walk("data/", topdown=False):
                files += [f for f in fs if f.endswith("00000.gz")]

            files = [os.path.join("data/page_dist_training_data/", f) for f in files]
            data_list = []
            for i, file in tqdm(enumerate(files)):
                    sentences = list(_open_file(file))
                    data_list += sentences

            original_data_length = len(data_list)

            new_list = []
            for i, page in enumerate(data_list):
                if i % 10000 == 0:
                    print("{}/{}".format(i,original_data_length))
                c = sum(item in page for item in wire_vocab)
                # only include Wikipedia pages that mention at least 2 WiRe elements
                if c>=2:
                    decoded_page = [x.encode('ascii','ignore') for x in page]
                    new_list.append(decoded_page)

            data_list = new_list

            print("Original data length = {}".format(original_data_length))
            print("Reduced data length = {}".format(len(new_list)))


            with gzip.open("out.gz", "w") as tfz:
                tfz.write(json.dumps(new_list))
            tfz.close()
            #    for page in new_list:
            #        ascii_page = listToString(page,args.MWE)
            #        tfz.write(" ".join([str(entity) for entity in ascii_page]))
            #tfz.close()

            #with gzip.open('wirezip.gz', 'a') as zip:
            #    for page in new_list:
            #        ascii_page = listToString(page,args.MWE)
            #        for entity in ascii_page:
            #            zip.write(entity)
            #        zip.write("\n")
            #zip.close()

        print("WRITING DATA")
        lst = []
        for entities in tqdm(data_list):
            lst.append(listToString(entities, args.MWE))
            lst.append("\n")
        data_string = listToString(lst, args.MWE)
        print(data_string)
        print("STRING CREATED")
        text_file = open("wire.txt", "w")
        text_file.write(data_string)
        text_file.close()
        print("STRING WRITTEN TO TEXT FILE")
        data = tokenizer_MWE0(data_string)
        print("STRING TOKENIZED")

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
            with gzip.open(files[0],'rt') as f:
                for line in f:
                    print(line)
            for i, file in tqdm(enumerate(files)):
                    sentences = list(_open_file(file))
                    data_list += sentences
            # pickle_out = open("data_list.pkl","wb")
            # pkl.dump(data_list, pickle_out)
            # pickle_out.close()

        #if args.MWE == 2:
            #data_list = data_list[20]

        print("WRITING DATA")
        lst = []
        for entities in tqdm(data_list):
            lst.append(listToString(entities, args.MWE))
            lst.append("\n")
        data_string = listToString(lst, args.MWE)
        print(data_string)
        print("STRING CREATED")
        text_file = open("wikipedia.txt", "w")
        text_file.write(data_string)
        text_file.close()
        print("STRING WRITTEN TO TEXT FILE")
        data = tokenizer_MWE0(data_string)
        print("STRING TOKENIZED")
        #print(data[:2])

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
    #print(dataset[:2])
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
              eta=args.eta, Closs=args.Closs,
              verbose_loss=args.verbose_loss,
              verbose_gradients=args.verbose_gradients)



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
    epoch_fwd_KL_pears = []
    epoch_fwd_KL_spears = []
    epoch_rev_KL_pears = []
    epoch_rev_KL_spears = []
    epoch_fisher_pears = []
    epoch_fisher_spears = []
    epoch_cos_pears = []
    epoch_cos_spears = []
    epoch_times = []

    train_time_start = time.time()

    for e in range(args.num_epochs):
        epoch_start = time.time()
        print("---------- EPOCH {} ----------".format(e+1))
        if args.MWE == 1:
            with open('w_and_p.txt', 'r') as corpus:
                total_num_examples = len(open('w_and_p.txt').readlines(  ))
                if args.dynamic_window_size:
                    epoch_losses.append(embed.train_dynamic(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, total_num_examples=total_num_examples, verbose_pairs=args.verbose_pairs, report_interval=args.report_schedule))
                else:
                    epoch_losses.append(embed.train(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, verbose_pairs=args.verbose_pairs, report_interval=args.report_schedule))

        elif args.MWE == 3:
            with open('wire.txt', 'r') as corpus:
                total_num_examples = len(open('wikipedia.txt').readlines(  ))
                if args.dynamic_window_size:
                    epoch_losses.append(embed.train_dynamic(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, total_num_examples=total_num_examples, verbose_pairs=args.verbose_pairs, report_interval=args.report_schedule))
                else:
                    epoch_losses.append(embed.train(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, verbose_pairs=args.verbose_pairs, report_interval=args.report_schedule))

        else:
            with open('wikipedia.txt', 'r') as corpus:
                total_num_examples = len(open('wikipedia.txt').readlines(  ))
                if args.dynamic_window_size:
                    epoch_losses.append(embed.train_dynamic(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, total_num_examples=total_num_examples, verbose_pairs=args.verbose_pairs, report_interval=args.report_schedule))
                else:
                    epoch_losses.append(embed.train(iter_pairs(corpus, vocab, dynamic_window_size=args.dynamic_window_size, batch_size=batch_size, nsamples=args.neg_samples, window=window),
                                    n_workers=args.num_threads, verbose_pairs=args.verbose_pairs, report_interval=args.report_schedule))

        epoch_end = time.time()
        epoch_times.append(round(epoch_end - epoch_start,2))

        if args.save==True:
            print("Epoch {} complete. Saving model.".format(e+1))
            os.chdir("Models/")
            embed.save('model_MWE={}_d={}_e={}_neg={}_eta={}_C={}_epoch={}'.format(args.MWE,args.dim,args.num_epochs,args.neg_samples,args.eta,args.Closs,e+1), vocab=vocab.id2word, full=True)
            os.chdir('..')


        print("MEASURING EMBEDDING PERFORMANCE ON VALIDATION DATA")
        actual, pred_KL_fwd, pred_KL_rev, pred_fisher, pred_cos = get_predictions(validation_path, e, embed, vocab, is_round=False)

        ### forward KL predictions ###
        pear_r_fwd, _ = pearsonr(actual, pred_KL_fwd)
        spear_r_fwd, _ = spearmanr(actual, pred_KL_fwd)
        print("------ Epoch: {} FORWARD KL SIMILARITY KL[src||dst] ------".format(e+1))
        print("Pearson R: {},  Spearman R: {}".format(pear_r_fwd, spear_r_fwd))

        ### reverse KL predictions ###
        pear_r_rev, _ = pearsonr(actual, pred_KL_rev)
        spear_r_rev, _ = spearmanr(actual, pred_KL_rev)
        print("------ Epoch: {} REVERSE KL SIMILARITY KL[dst||src] ------".format(e+1))
        print("Pearson R: {},  Spearman R: {}".format(pear_r_rev, spear_r_rev))

        ### fisher predictions ###
        pear_fisher, _ = pearsonr(actual, pred_fisher)
        spear_fisher, _ = spearmanr(actual, pred_fisher)
        print("------ Epoch: {} FISHER DISTANCE ------".format(e+1))
        print("Pearson R: {},  Spearman R: {}".format(pear_fisher, spear_fisher))

        ### cosine predictions ###
        pear_r_cos, _ = pearsonr(actual, pred_cos)
        spear_r_cos, _ = spearmanr(actual, pred_cos)
        print("------ Epoch: {} COSINE SIMILARITY OF MEANS ------".format(e+1))
        print("Pearson R: {},  Spearman R: {}".format(pear_r_cos, spear_r_cos))

        epoch_fwd_KL_pears.append(pear_r_fwd)
        epoch_fwd_KL_spears.append(spear_r_fwd)
        epoch_rev_KL_pears.append(pear_r_rev)
        epoch_rev_KL_spears.append(spear_r_rev)
        epoch_fisher_pears.append(pear_fisher)
        epoch_fisher_spears.append(spear_fisher)
        epoch_cos_pears.append(pear_r_cos)
        epoch_cos_spears.append(spear_r_cos)

    train_time_end = time.time()
    training_time = round(train_time_end - train_time_start,2)

    print("\n\n\nEPOCH LOSSES : {}".format(epoch_losses))
    print("EPOCH fwd KL Pearson R : {}".format(epoch_fwd_KL_pears))
    print("EPOCH fwd KL Spearman R : {}".format(epoch_fwd_KL_spears))
    print("EPOCH rev KL Pearson R : {}".format(epoch_rev_KL_pears))
    print("EPOCH rev KL Spearman R : {}".format(epoch_rev_KL_spears))
    print("EPOCH Fisher Pearson R : {}".format(epoch_fisher_pears))
    print("EPOCH Fisher Spearman R : {}".format(epoch_fisher_spears))
    print("EPOCH cosine Pearson R : {}".format(epoch_cos_pears))
    print("EPOCH cosine Spearman R : {}".format(epoch_cos_spears))
    print("TOTAL TRAININT TIME = {} secs".format(training_time))
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
        print("\n\n")
        print("FINDING NEAREST NEIGHBOURS")

        target = "war"
        metric = cosine
        num = 10

        target_idx = entity_2_idx[target]
        neighbours = embed.nearest_neighbors(target=target_idx, metric=metric, num=num+1, vocab=vocab,
                          sort_order='similarity')

        print("Target = {}".format(target))
        print("Similarity metric = {}".format(metric))
        for i in range(1,num+1):
            print("{}: {}".format(i,neighbours[i]))
            # print("rank {}: word = {}, sigma = {}, id = {}, similarity = {}".format(i,neighbours[i][word],neighbours[i][sigma],neighbours[i][id],neighbours[i][similarity]))



    ############################################################################

    if args.csv:
        f_results = 'CSVs/grid_search_results_epochs={}.csv'.format(args.num_epochs)

        hyperparameter_list = ["Threads", "Dimension", "Neg samples", "Eta", "Closs"]
        epoch_list = ['Epoch {} Loss'.format(i+1) for i in range(args.num_epochs)]
        pear_r_fwd_list = ['Epoch {} fwd KL Pearson R'.format(i+1) for i in range(args.num_epochs)]
        spear_r_fwd_list = ['Epoch {} fwd KL Spearman R'.format(i+1) for i in range(args.num_epochs)]
        pear_r_rev_list = ['Epoch {} rev KL Pearson R'.format(i+1) for i in range(args.num_epochs)]
        spear_r_rev_list = ['Epoch {} rev KL Spearman R'.format(i+1) for i in range(args.num_epochs)]
        pear_r_cos_list = ['Epoch {} cosine Pearson R'.format(i+1) for i in range(args.num_epochs)]
        spear_r_cos_list = ['Epoch {} cosine Spearman R'.format(i+1) for i in range(args.num_epochs)]
        time_list = ['Epoch {} Time'.format(i+1) for i in range(args.num_epochs)]

        header_list = hyperparameter_list + epoch_list + pear_r_fwd_list + spear_r_fwd_list + pear_r_rev_list + spear_r_rev_list + pear_r_cos_list + spear_r_cos_list + time_list

        if os.path.exists(f_results):
            append_write = 'a' # append if already exists
        else:
            # write header
            with open(f_results, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(header_list)
            append_write = 'a' # make a new file if not

        with open(f_results, append_write) as file:
            writer = csv.writer(file)
            hyperparameter_values = [args.num_threads, args.dim, args.neg_samples, args.eta, args.Closs]
            values_list = hyperparameter_values + epoch_losses + epoch_fwd_KL_pears + epoch_fwd_KL_spears + epoch_rev_KL_pears + epoch_rev_KL_spears + epoch_cos_pears + epoch_cos_spears + epoch_times
            writer.writerow(values_list)

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
