import numpy as np


def softmax(x, axis=1):
    """
    Implements a stabilized softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    NOTE:
    Do not import or use these packages: sklearn, scipy, sys, importlib.

    Note that np.atleast_2d will take an array of shape [K, ]
        and make it an array of shape [1, K]. Thus if you pass
        an array of shape [K, ] to softmax, you should leave
        axis=1 as the default.
    """
    x = np.atleast_2d(x)
    x = x - np.max(x, axis = axis, keepdims=True)
    sm = np.exp(x)/np.exp(x).sum(axis = axis, keepdims = True)
    # print('softmax: ', sm)
    return sm

    

def stable_log_sum(X):
    """
    Implement a stabilized log sum operation.

    When all elements of X are greater than -743, this will be equivalent to:
        >>> np.sum(np.log(np.sum(np.exp(X), axis=1)))

    However, for large negative values of X, your computer will represent
        `np.exp(X)` as 0.0; this is called underflow. Your implementation
        should avoid underflow in one of two ways:

    1. You can assume that X is an array of shape (K, 2) and approximate the
          sum by ignoring the smallest of the two values in each of the K rows.
          That is, while `log(a) + log(b) != log(a + b)`, if a is much bigger
          than b, `log(a)` is a good approximation for `log(a + b)`.
          Thus, `max([log(X[i, 0]), log(X[i, 1])])` provides a decent approximation
          for `log(exp(X[i, 0]) + exp(X[i, 1]))`. You will still need to sum
          over all rows.

    2. To exactly compute the sum without relying on the above approximation,
          you can use some clever math that relies on the properties of
          logarithms to avoid having to ever compute np.exp(X). See the
          following link:
          https://stackoverflow.com/questions/22009862/how-to-calculate-logsum-of-terms-from-its-component-log-terms/22385004
          Note that this approach is more difficult and while more exact,
          that exactness isn't necessary for this assignment.

    NOTE:
    Do not import or use these packages: sklearn, scipy, sys, importlib.

    Args:
        X: an array of shape (K, 2) for some K
    Returns:
        sum(log(sum(exp(X), axis=1))), avoiding underflow
    """
    # You can assume that this array is of shape (K, 2)
    assert X.shape[1] == 2 and len(X.shape) == 2
    # print(X)
    
    # log_sm = np.zeros(())
    if (X>-743).all():
        log_sm = np.sum(np.log(np.sum(np.exp(X), axis=1)))
        # print('log: ', log_sm)
    else:
        log_sm = 0
        for i in range(X.shape[0]):
            if (X[i,:]>-743).all():
                log_sm = log_sm + np.log(np.sum(np.exp(X[i,:])))
            else:
                a=np.max([X[i, 0], X[i, 1]])
                log_sm = log_sm + a

    return log_sm
