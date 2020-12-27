import numpy as np

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



def KL_Multivariate_Gaussians(mu1, Sigma1, mu_2, Sigma2):
    '''
    Implement KL[P1||P2]using the formula from p13 of
    http://stanford.edu/~jduchi/projects/general_notes.pdf
    '''

    n = mu1.shape[0]

    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)

    Sigma2_inv = np.linalg.pinv(Sigma2)

    return 0.5 * ( np.log(det_Sigma2/det_Sigma1) - n + np.trace(Sigma2_inv @ Sigma1) + (mu_2-mu_1).T @ Sigma2_inv @ (mu_2-mu_1) )

    
