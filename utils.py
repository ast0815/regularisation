import numpy as np
from scipy.optimize import minimize

# Different penalty matrices for a given model


def Q1(model, bin_pairs=None):
    # Penalise bin diffrences
    # chi = sum_i( (x_i - x_{i+1})^2 )
    N = len(model)
    if bin_pairs is None:
        bin_pairs = [(i, i + 1) for i in range(N - 1)]
    C = np.zeros((N, N), dtype=float)
    for i, j in bin_pairs:
        C[i, i] += 1
        C[i, j] += -2
        C[j, j] += 1
    return (C + C.T) / 2


def Q2(model, bin_triples=None):
    # Penalise bin difference differences
    # chi = sum_i( (x_i - 2*x_{i+1} + x_{1+2})^2 )
    N = len(model)
    if bin_triples is None:
        bin_triples = [(i, i + 1, i + 2) for i in range(N - 2)]
    C = np.zeros((N, N), dtype=float)
    for i, j, k in bin_triples:
        C[i, i] += 1
        C[i, j] += -4
        C[i, k] += 2
        C[j, j] += 4
        C[j, k] += -4
        C[k, k] += 1
    return (C + C.T) / 2


def Q1s(model, bin_pairs=None):
    # Penalise scaling differences
    # x -> x/m
    # e.g.
    # sum_i( (x_i/m_i - x_{i+1}/m_{i+1})^2 )
    model = model / np.mean(model)
    N = len(model)
    if bin_pairs is None:
        bin_pairs = [(i, i + 1) for i in range(N - 1)]
    C = np.zeros((N, N))
    for i, j in bin_pairs:
        C[i, i] += 1 / model[i] ** 2
        C[i, j] += -2 / (model[i] * model[j])
        C[j, j] += 1 / model[j] ** 2
    return (C + C.T) / 2


def Q2s(model, bin_triples=None):
    # Penalise scaling difference differences
    model = model / np.mean(model)
    N = len(model)
    if bin_triples is None:
        bin_triples = [(i, i + 1, i + 2) for i in range(N - 2)]
    C = np.zeros((N, N))
    for i, j, k in bin_triples:
        C[i, i] += 1 / model[i] ** 2
        C[i, j] += -4 / (model[i] * model[j])
        C[i, k] += 2 / (model[i] * model[k])
        C[j, j] += 4 / model[j] ** 2
        C[j, k] += -4 / (model[j] * model[k])
        C[k, k] += 1 / (model[k] * model[k])
    return (C + C.T) / 2


def chi2(model, data, cov_inv, A=None):
    """Calculate the chi2 of the model in the original data."""

    if A is None:
        A = np.eye(len(model))

    diff = A @ model - data
    return diff.T @ cov_inv @ diff


def xsec_grad(model, data, cov_inv, A=None):
    """Calculate the gradient of the likelihood surface in the XSEC space.

    The length will correspond to the endpoint along the gradient with the lowest chi2.

    """

    if A is None:
        A = np.eye(len(model))

    grad = A.T @ cov_inv @ (data - model)

    def minfun(x):
        y = A @ (model + grad * x) - data
        return y.T @ cov_inv @ y

    ret = minimize(minfun, x0=1.0)

    return grad * ret.x[0]


def scaled_grad(scale_model, model, data, cov_inv, A=None):
    """Calculate the gradient of the likelihood surface in the XSEC/scale_model space.

    The length will correspond to the endpoint along the gradient with the lowest chi2.

    """

    if A is None:
        A = np.eye(len(model))

    grad = scale_model * (A.T @ cov_inv @ (data - model))

    def minfun(x):
        y = A @ (model + scale_model * grad * x) - data
        return y.T @ cov_inv @ y

    ret = minimize(minfun, x0=1.0)

    return grad * ret.x[0]


def log_grad(model, data, cov_inv, A=None):
    """Calculate the gradient of the likelihood surface in the log(XSEC) space.

    The length will correspond to the endpoint along the gradient with the lowest chi2.

    """

    if A is None:
        A = np.eye(len(model))

    grad = model * (A.T @ cov_inv @ (data - model))

    def minfun(x):
        y = A @ (model * np.exp(grad * x)) - data
        return y.T @ cov_inv @ y

    ret = minimize(minfun, x0=1.0)

    return grad * ret.x[0]
