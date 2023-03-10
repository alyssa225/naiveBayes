import numpy as np

from free_response.data import build_dataset
from src.naive_bayes_em import NaiveBayesEM


def q2a():
    """
    First, fit a NaiveBayesEM model to the dataset.  Then, using your fit
    model's `beta` parameters, define:
      `f(j) = p(w_j | y = 1, beta) - p(w_j | y = 0, beta)`

    as the difference in "party affiliation" of the `j`th word, which takes
    positive values for words that are more likely in speeches by Republican
    presidents and takes negative values for words more likely in speeches by
    Democratic presidents.

    Compute `f(j)` for each word `w_j` in the vocabulary. Please use
    probabilities, not log probabilities.

    Hint: `f(j)` should be between -1 and 1 for all `j` (and quite close to 0)

    You will use these `f(j)` values to answer FRQ 2a.
    """
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    nbem = NaiveBayesEM(max_iter=10)
    nbem.fit(data, labels)

    f = np.exp(nbem.beta[:,1])-np.exp(nbem.beta[:,0])
    print('f(j): ', f)
    fmin = np.sort(f)[0:5]
    fmax = np.sort(f)[-5:]
    print('fmaxs: ',fmax)
    print('fmins: ',fmin)
    fmins_ind = f.argsort()[0:5]
    fmaxs_ind = f.argsort()[-5:]
    # print('fmaxs_ind: ',fmaxs_ind)
    # print('fmins_ind: ',fmins_ind)
    pos_words = vocab[fmaxs_ind]
    neg_words = vocab[fmins_ind]
    print('high value words: ', pos_words)
    print('low value words: ', neg_words)
    return f


def q2b():
    """
    Helper code for the Free Response Q2b
    You shouldn't need to edit this function
    """
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    isfinite = np.isfinite(labels)
    nbem = NaiveBayesEM(max_iter=10)
    nbem.fit(data, labels)

    # Use predict_proba to see output probabilities
    probs = nbem.predict_proba(data)[isfinite]
    preds = nbem.predict(data)
    correct = preds[isfinite] == labels[isfinite]

    # The model's "confidence" in its predicted output when right 
    right_label = labels[isfinite][correct].astype(int)
    prob_when_correct = probs[correct, right_label]

    # The model's "confidence" in its predicted output when wrong 
    incorrect = np.logical_not(correct)
    wrong_label = 1 - labels[isfinite][incorrect].astype(int)
    prob_when_incorrect = probs[incorrect, wrong_label]

    # Use these number to answer FRQ 2b
    print("When NBEM is correct:")
    print(prob_when_correct.tolist())
    print("When NBEM is incorrect:")
    print(prob_when_incorrect.tolist())


if __name__ == "__main__":
    q2a()
    q2b()
