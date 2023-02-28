### Setup (2 points)

All you need to do for these points is pass the `test_setup` case. This
requires putting your NetID in the `netid` file and creating two PDF files
titled `XXX_qYYY.pdf` where `XXX` is replaced with your NetID, and `YYY`
ranges from 1 to 2. The content of these PDFs won't be graded, this is just to
ensure that you can set up your repository to be autograded.

Your final submission must also pass the `test_setup` test, or you will lose
these points.

# Coding (16 points)

Start by solving the practice problems in `src/sparse_practice.py`; these
will help you understand how scipy sparse matrices work.

In `src/utils.py`, you will write `softmax` and `stable_log_sum` functions,
making sure that both are numerically stable. These will be helpful in your
Naive Bayes models.

Then, you will implement two versions of a Naive Bayes classifier.  In
`src/naive_bayes.py`, the `NaiveBayes` classifier considers the case where all
the data is labeled.  In `src/naive_bayes_em.py`, `NaiveBayesEM` classifier
will use the Expectation Maximization algorithm to also learn from unlabeled
data.

The lecture slides, as well as [the reading that describes the Naive Bayes EM
algorithm](naive_bayes.pdf) will be helpful.  We have also provided extensive
documentation in the provided code, please read it carefully!  For example,
when implementing the `NaiveBayesEM` classifier, be careful to correctly
initialize your parameters (such that all probabilities are equal) and
correctly update your inferred class distribution during the E-step (do not
overwrite the observed labels with your predicted probabilities).

Your goal is to pass the test suite that is run by `python -m pytest`.
Once the tests are passed, you may move on Q1 and Q2 below.
We suggest that you try to pass the tests in the order they are listed in
`tests/rubric.json`.

Your grade for this section is defined by the autograder. If it says you got an
80/100, you get 80% of the points.  If your code does not run on our autograder
with `python -m pytest` or you do not include your netid in your submitted
code, you may get a zero for the coding portion!

## The speeches dataset

The dataset provided in `data/speeches.zip` (which should be automatically
unzipped by the `src/data.py` code) contains [State of the Union
addresses](https://en.m.wikisource.org/wiki/Portal:State_of_the_Union_Speeches_by_United_States_Presidents)
by United States presidents dating back to 1790. In recent history
(specifically, since the Civil Rights Act of 1964), all US presidents have
belonged to one of two political parties, which have remained relatively
ideologically stable.

In this data, we treat the words of each speech as the features, and the
political party of the speaker as the label.  For presidents prior to 1964, we
will treat their party membership as unobserved, as it does not necessarily
correspond to today's two-party system. The `NaiveBayes` classifier will only
consider fully-labeled data -- it cannot use speeches prior to 1964. The
`NaiveBayesEM` classifier will also use unlabeled speeches to learn its
probabilities.

If the provided code fails to unzip the speeches.zip file, please ask for help.

# Free-response questions (4 points)

To answer some of the free-response questions, you may have to write extra code
(that is not covered by the test cases).  You do *not* have to submit any code
that you write for these questions! If you do add such code to your repository,
double-check that the final version of your default branch code can run
`python -m pytest` with only the provided imports in `requirements.txt`.

## 1. Comparing Naive Bayes with and without unlabeled data (2 points)

See `free_response/q1.py` for code that will help you with part (a) and (b).

The code will create a dataset using `src.data.build_dataset` with 100 documents,
at most 2000 words per document, and a vocabulary size of 1000 words. As mentioned
above, we will consider speeches prior to 1964 as unlabeled. 
You do not need to split this dataset into train and test splits. Instead, you will fit
your `NaiveBayes` and `NaiveBayesEM` models on the `data` matrix and `labels` array.
For the `NaiveBayesEM` model, consider a `max_iter` value of 1, 2, and 10.
For (a) and (b) below, evaluate your models on the entire dataset and add your results to this table.

| Model | `max_iter` | Accuracy | Log Likelihood |
| ---   | ---        | ---      | ---            |
| NB    | N/A        |          |                |
| NB+EM | 1          |          |                |
| NB+EM | 2          |          |                |
| NB+EM | 10         |          |                |

(a) Use each fit model to predict labels for each labeled (post-1964) speech.
Calculate the accuracy of each model's predictions and include them in the table above.
Unlabeled speeches (`labels == np.nan`) should not factor into this accuracy calculation
for **either** model.

(b) Calculate the log likelihood of the entire dataset according each model and
include it in the table above.  Unlabeled examples should not contribute to the
likelihood of the non-EM model, but should contribute to the likelihood of the
EM model.

(c) Discuss the differences in accuracy and likelihood between the `NaiveBayes`
and `NaiveBayesEM` models. Why do the `NaiveBayesEM` models have a lower
likelihood than the `NaiveBayes` model? What explains the differences in
accuracy that you see across all four models?

## 2. Naive Bayes and probabilistic predictions (2 points)

For these questions, use the same dataset as in the previous question.
See `free_response/q2.py` for code that will help you with part (b).

(a) Define `f(j) = p(w_j | y = 1, beta) - p(w_j | y = 0, beta)` as the
difference in "party affiliation" of the `j`th word, which takes positive
values for words that are more likely in speeches by Republican presidents and
takes negative values for words more likely in speeches by Democratic
presidents. Using your NaiveBayesEM model, compute `f(j)` for each word `w_j`
in the vocabulary. Please use probabilities, not log probabilities; `f(j)`
should be between -1 and 1 for all `j` (and quite close to 0). What are the
five words with the highest (positive) and lowest (negative) values of `f(j)`?
What is the value of `f(j)` for each of those words?

(b) Use the code in `free_response/q2.py` to look at the probabilities output by
the NaiveBayesEM model both when it makes a correct prediction and when it
makes an incorrect prediction on the labeled speeches.  We can think about
these probabilities as describing the "confidence" of the classifier -- if it
outputs a probability of 50% for both labels, the model is unconfident. When
the model's probability of the predicted label is close to 100%, it is very
confident in its prediction.  What do you notice about the confidence of your
NaiveBayesEM classifier, both when it is correct and when it is incorrect? What
aspects of the data and/or your model contribute to this behavior?

(c) Suppose we were using a machine learning classifier in a high-stakes domain
such as predicting a clinical diagnosis of hospital patients in the intensive
care unit. What might be one danger of having a classifier that is always
confident, even when incorrect? What might be one benefit of a classifier where
its confidence represents the probability that it makes a correct prediction?

(d) Suppose you wanted to introduce regularization to guide our NaiveBayesEM
model to be less confident in its predictions. How would you do this? How
would your approach work?
