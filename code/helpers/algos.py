import numpy as np
import copy
import math
import random as rnd

###################################################
#                                                 #
#               First-Order methods               #
#                                                 #
###################################################


def line_search_mbSGD(model, x0, nbr_epochs, batch_size, verbose=False):

    lls = []
    xs = []
    epochs = []
    x = copy.deepcopy(x0)

    n = len(model.df)

    max_iters = int(np.ceil(nbr_epochs * n / (batch_size)))

    print('Start Line search mini batch SGD for {} epochs ({} iterations)'.format(nbr_epochs, max_iters))

    for n_iter in range(max_iters):
        epochs.append(n_iter * batch_size / n)

        if verbose:
            print('Iteration {}'.format(n_iter))

        ll = model.loglikelihood(x)

        idx = np.random.choice(n, batch_size, replace=False)

        grad = model.grad(x,idx )

        lls.append(ll)
        xs.append(x)

        if np.linalg.norm(grad) < 1e-4:
            print("break")
            break

        if math.isnan(ll):
            print('NAN')
            break

        if verbose:
            print("Step ready to be performed, ll = {:.5f}".format(ll))

        x, alpha = backtracking_line_search(model, x, grad, idx, verbose)

        if verbose:
            print()

    epochs.append(max_iters * batch_size / n)
    ll = model.loglikelihood(x)
    lls.append(ll)
    xs.append(x)

    return epochs, xs, lls


def line_search_adagrad(model, x0, nbr_epochs, batch_size, verbose=False):

    lls = []
    xs = []
    epochs = []
    x = copy.deepcopy(x0)

    n = len(model.df)

    max_iters = int(np.ceil(nbr_epochs * n / (batch_size)))

    print('Start Line search Adagrad for {} epochs ({} iterations)'.format(nbr_epochs, max_iters))

    cache = np.zeros(len(x))

    for n_iter in range(max_iters):
        epochs.append(n_iter * batch_size / n)

        if verbose:
            print('Iteration {}'.format(n_iter))

        ll = model.loglikelihood(x)

        idx = np.random.choice(n, batch_size)

        grad = model.grad(x, idx)

        lls.append(ll)
        xs.append(x)

        if np.linalg.norm(grad) < 1e-4:
            print("break")
            break

        if math.isnan(ll):
            print('NAN')
            break

        if verbose:
            print("Step ready to be performed, ll = {:.5f}".format(ll))

        cache = cache + grad**2

        step = grad/np.sqrt(cache + 1e-8*np.ones(len(grad)))

        x, alpha = backtracking_line_search(model, x, step, idx, verbose)

        if verbose:
            print()

    epochs.append(max_iters * batch_size / n)
    ll = model.loglikelihood(x)
    lls.append(ll)

    return epochs, xs, lls


###################################################
#                                                 #
#          Second-Order methods (Newton)          #
#                                                 #
###################################################


def line_search_mini_batch_SNM(model, x0, nbr_epochs, batch_size, verbose=False):
    lls = []
    xs = []
    epochs = []
    x = copy.deepcopy(x0)

    n = len(model.df)

    max_iters = int(np.ceil(nbr_epochs * n / (batch_size)))

    print('Start Line search mini batch SN for {} epochs ({} iterations)'.format(nbr_epochs, max_iters))

    nbr_grad = 0
    nbr_newton = 0

    for n_iter in range(max_iters):
        epochs.append(n_iter * batch_size / n)

        if verbose:
            print('Iteration {}'.format(n_iter))

        ll = model.loglikelihood(x)

        idx = np.random.choice(n, batch_size, replace=False)
        grad = model.grad(x, idx)
        hess = model.hessian(x, idx)

        lls.append(ll)
        xs.append(x)

        if math.isnan(ll):
            print('NAN')
            break

        if verbose:
            print("Gradient norm: {:.2E}".format(np.linalg.norm(grad)))

        if np.all(np.linalg.eigvals(hess) < 0):

            if verbose:
                print("Newton step, ll = {:.5f}".format(ll))
            step = np.linalg.solve(hess, -grad)
            nbr_newton += 1

        else:
            if verbose:
                print("Gradient Descent step, ll = {:.5f}".format(ll))
            step = grad
            nbr_grad += 1

        x, alpha = backtracking_line_search(model, x, step, idx, verbose)

        if verbose:
            print()

    epochs.append(max_iters * batch_size / n)
    ll = model.loglikelihood(x)
    lls.append(ll)
    xs.append(x)

    return epochs, xs, lls, nbr_newton / (nbr_newton + nbr_grad)


###################################################
#                                                 #
#       Second-Order methods (Quasi-Newton)       #
#                                                 #
###################################################


def bfgs(model, x0, nbr_epochs, start=None, verbose=False):
    lls = []
    xs = []
    epochs = []

    x = copy.deepcopy(x0)

    n = len(model.df)

    if start is 'hessian':
        inv_B = np.linalg.inv(model.hessian(x))
    else:
        inv_B = np.eye(len(model.params))

    max_iters = nbr_epochs

    print('Start BFGS for {} epochs ({} iterations)'.format(nbr_epochs, max_iters))

    I = np.eye(len(model.params))

    for n_iter in range(max_iters):
        epochs.append(n_iter)

        if verbose:
            print('Iteration {}'.format(n_iter))

        ll = model.loglikelihood(x)
        grad = model.grad(x)

        lls.append(ll)
        xs.append(x)

        if verbose:
            print('BFGS step, ll = {:.5E}'.format(ll))

        p = -np.dot(inv_B, grad)

        [x, alpha] = backtracking_line_search(model, x, p, None, verbose)

        s = (alpha * p).reshape(len(model.params), 1)

        y = (model.grad(x) - grad).reshape(len(model.params), 1)

        # Update inverse of B
        beta = (np.dot(np.transpose(y), s))
        M1 = I - np.dot(s, np.transpose(y)) / beta
        M2 = I - np.dot(y, np.transpose(s)) / beta
        M3 = np.dot(s, np.transpose(s)) / beta

        inv_B = np.dot(np.dot(M1, inv_B), M2) + M3

        if verbose:
            print()

    epochs.append(max_iters)
    ll = model.loglikelihood(x)
    lls.append(ll)
    xs.append(x)

    return epochs, xs, lls


def res_bfgs(model, x0, nbr_epochs, batch_size, Gamma=None, delta=None, verbose=False):

    if Gamma is None:
        Gamma = 1
    if delta is None:
        delta = 0.2

    lls = []
    xs = []
    epochs = []

    x = copy.deepcopy(x0)

    n = len(model.df)
    m = len(model.params)

    B = np.eye(m)

    max_iters = int(np.ceil(nbr_epochs * n / batch_size))

    print('Start Regularized stochastic BFGS for {} epochs ({} iterations)'.format(nbr_epochs, max_iters))

    I = np.eye(m)

    for n_iter in range(max_iters):
        epochs.append(n_iter * batch_size / n)

        if verbose:
            print('Iteration {}'.format(n_iter))

        idx = np.random.choice(n, batch_size)

        ll = model.loglikelihood(x)
        grad = model.grad(x, idx)

        lls.append(ll)
        xs.append(x)

        if verbose:
            print('BFGS step, ll = {:.5E}'.format(ll))

        p = np.dot(np.linalg.inv(B) + Gamma * I, grad)

        old_x = x

        [x, alpha] = backtracking_line_search(model, x, p, None, verbose)
        # print(alpha)

        v = (x - old_x).reshape(m, 1)

        r = (model.grad(x, idx) - grad - delta * np.transpose(v)).reshape(m, 1)

        # Update inverse of B
        M1 = np.dot(r, np.transpose(r)) / np.dot(np.transpose(v), r)
        M2 = np.dot(B, np.dot(v, np.dot(np.transpose(v), B))) / np.dot(np.transpose(v), np.dot(B, v))

        B = B + M1 - M2 + delta * I

        if verbose:
            print()

    epochs.append(max_iters * batch_size / n)
    ll = model.loglikelihood(x)
    lls.append(ll)
    xs.append(x)

    return epochs, xs, lls


####################################################
#                                                  #
#                Helpers Functions                 #
#                                                  #
####################################################


def backtracking_line_search(model, x, step, indices, verbose=False):
    ll = model.loglikelihood(x, indices)

    grad = model.grad(x, indices)
    m = np.dot(np.transpose(step), grad)

    c = 0.5

    t = -c*m
    tau = 0.5

    alpha = 10

    while True:
        tmp_x = x + alpha*step
        if ll - model.loglikelihood(tmp_x, indices) <= alpha*t or alpha < 1e-8:
            if verbose:
                print('Line Search -> alpha = {:.2E}'.format(alpha))
            return tmp_x, alpha

        alpha = tau*alpha