import numpy as np
import pandas as pd
import os
import csv


def cosine(a, b, normalize=True):
    '''
    Compute the cosine measure between a and b.

    a = numpy array with each row one observation.
    b = vector or 1d array, len(b) == a.shape[1]

    Returns length a.shape[0] array with the cosine similarities
    '''
    if not normalize:
        return np.dot(a, b.reshape(-1, 1)).flatten()

    else:
        # normalize b
        norm_b = np.sqrt(np.sum(b ** 2))
        b_normalized = b / norm_b

        # get norms of a
        norm_a = np.sqrt(np.sum(a ** 2, axis=1))

        # compute cosine measure and normalize
        return np.dot(a, b_normalized.reshape(-1, 1)).flatten() / norm_a

def cosine_between_vecs(a, b, normalize=True):
    '''
    Compute the cosine measure between a and b.

    a = vector
    b = vector

    Returns scalar with the cosine similarities
    '''
    if not normalize:
        return np.dot(a, b)

    else:
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))

        a_normalized = a / norm_a
        b_normalized = b / norm_b

        # compute cosine measure
        return np.dot(a_normalized, b_normalized)



def KL_Multivariate_Gaussians(mu1, Sigma1, mu2, Sigma2):
    '''
    Implement KL[P1||P2]using the formula from p13 of
    http://stanford.edu/~jduchi/projects/general_notes.pdf

    mu1 = vector
    Sigma1 = vector (of diagonal elements)
    mu2 = vector
    Sigma2 = vector (of diagonal elements)

    Returns scalar with the KL divergence between the two Gaussian distributions
    '''

    n = len(mu1)

    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)

    Sigma2_inv = np.linalg.pinv(Sigma2)

    return 0.5 * ( np.log(det_Sigma2/det_Sigma1) - n + np.trace( np.matmul(Sigma2_inv,Sigma1) ) + np.matmul((mu2-mu1).T,np.matmul(Sigma2_inv,(mu2-mu1))) )




def fisher_dist(mu1, Sigma1, mu2, Sigma2):
    '''
    see eqn 20 of https://reader.elsevier.com/reader/sd/pii/S0166218X14004211?token=F35D5419800D2CC5C0E38D81D39228C93E23043D197A45EA1E160B1E11FCF7F448FF088B7E1FB3A406674C942DBFDB89
    '''
    n = len(mu1)

    summation = 0
    for i in range(n):
        term1 = np.linalg.norm(np.array([ (mu1[i]-mu2[i])/np.sqrt(2), Sigma1[i]+Sigma2[i] ]))
        term2 = np.linalg.norm(np.array([ (mu1[i]-mu2[i])/np.sqrt(2), Sigma1[i]-Sigma2[i] ]))
        summation += ( np.log( (term1+term2)/(term1-term2) ) )**2

    return np.sqrt(2 * summation)



def get_predictions(val_path, epoch, model, vocab, is_round=False):
    actual = []
    pred_KL_fwd = []
    pred_KL_rev = []
    pred_fisher = []
    pred_cos = []

    # iterate over full dataset
    df_val = pd.read_csv(val_path)
    print("Validation data loaded successfully")

    total_records = len(df_val.index)
    missing_records_count = 0

    for _, record in df_val.iterrows():
        src = record["srcWikiTitle"]
        dst = record["dstWikiTitle"]
        act_sim = float(record["relatedness"])

        src_idx = vocab.word2id(src)
        dst_idx = vocab.word2id(dst)

        mu_src = model.mu[src_idx]
        Sigma_src = np.diag(model.sigma[src_idx])
        sigma_src = model.sigma[src_idx]
        mu_dst = model.mu[dst_idx]
        Sigma_dst = np.diag(model.sigma[dst_idx])
        sigma_dst = model.sigma[dst_idx]

        # predict similarity
        try:
            pred_fwd_KL_sim = float(KL_Multivariate_Gaussians(mu_src, Sigma_src, mu_dst, Sigma_dst))
            pred_rev_KL_sim = float(KL_Multivariate_Gaussians(mu_dst, Sigma_dst, mu_src, Sigma_src))
            pred_fisher_sim = float(fisher_dist(mu_src, sigma_src, mu_dst, sigma_dst))
            pred_cos_sim = float(cosine_between_vecs(mu_src,mu_dst))

            if is_round:
                pred_fwd_KL_sim = np.round(pred_fwd_KL_sim)
                pred_rev_KL_sim = np.round(pred_rev_KL_sim)
                pred_fisher_sim = np.round(pred_fisher_sim)
                pred_cos_sim = np.round(pred_cos_sim)
        except KeyError:
            missing_records_count += 1
            continue

        # add records
        actual.append(act_sim)
        pred_KL_fwd.append(pred_fwd_KL_sim)
        pred_KL_rev.append(pred_rev_KL_sim)
        pred_fisher.append(pred_fisher_sim)
        pred_cos.append(pred_cos_sim)


        f_results = 'CSVs/preds_epoch={}.csv'.format(epoch)

        header_list = ['srcWikiTitle','dstWikiTitle','relatedness','fwd KL','rev KL','fisher','cosine']

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
            for i in range(len(actual)):
                writer.writerow([src,dst,act_sim,pred_fwd_KL_sim,pred_rev_KL_sim,pred_fisher_sim,pred_cos_sim])

    assert missing_records_count == 0
    #print("Missing records = {}/{}".format(missing_records_count,total_records))

    return np.array(actual), np.array(pred_KL_fwd), np.array(pred_KL_rev), np.array(pred_fisher), np.array(pred_cos)
