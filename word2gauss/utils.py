import numpy as np
import pandas as pd

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



def get_predictions(validation_path, model, vocab, is_round=False):
    actual = []
    pred_KL_fwd = []
    pred_KL_rev = []
    pred_cos = []

    # iterate over full dataset
    validation_data = pd.read_csv(validation_path)
    print("Validation data loaded successfully")

    for _, record in validation_data.iterrows():
        src = standardise_title(record["srcWikiTitle"])
        dst = standardise_title(record["dstWikiTitle"])
        act_sim = float(record["relatedness"])

        src_idx = vocab.word2id(src)
        dst_idx = vocab.word2id(dst)

        mu_src = model.mu[src_idx]
        Sigma_src = np.diag(model.sigma[src_idx])
        mu_dst = model.mu[dst_idx]
        Sigma_dst = np.diag(model.sigma[dst_idx])

        # predict similarity
        try:
            pred_fwd_KL_sim = float(KL_Multivariate_Gaussians(mu_src, Sigma_src, mu_dst, Sigma_dst))
            pred_rev_KL_sim = float(KL_Multivariate_Gaussians(mu_dst, Sigma_dst, mu_src, Sigma_src))
            pred_cos_sim = float(cosine_between_vecs(mu_src,mu_dst))

            if is_round:
                pred_fwd_KL_sim = np.round(pred_fwd_KL_sim)
                pred_rev_KL_sim = np.round(pred_rev_KL_sim)
                pred_cos_sim = np.round(pred_cos_sim)
        except KeyError:
            continue

        # add records
        actual.append(act_sim)
        pred_KL_fwd.append(pred_fwd_KL_sim)
        pred_KL_rev.append(pred_rev_KL_sim)
        pred_cos.append(pred_cos_sim)

    return np.array(actual), np.array(pred_KL_fwd), np.array(pred_KL_rev), np.array(pred_cos)
