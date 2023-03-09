import numpy as np
from alg import *
from tqdm import tqdm
from case_studies import *
def confidence_interval(dfs, n):
    dfs = dfs[:n]
    count = np.bincount([len(x) for x in dfs])[:-1]
    count = (np.zeros_like(count) + 100) - np.cumsum(count)
    norms = [np.linalg.norm(x, axis=1) for x in dfs]
    max_len = len(count)
    for norm in norms:
        norm.resize(max_len, refcheck=False)
    return np.sum(norms, axis=0) / count
def confidence_interval_steps(steps, n):
    steps = steps[:n]
    count = np.bincount([len(x) for x in steps])[:-1]
    count = (np.zeros_like(count) + 100) - np.cumsum(count)
    max_len = len(count)
    for step in steps:
        step.resize(max_len, refcheck=False)
    return np.sum(steps, axis=0) / count

def test_runner(x, c1, rho, eps, A, use_steepest=False):

    optimizer = (
        lambda f, df, Hf: steepest_lin_eq(x, f, df ,eps, A, c1, rho )
        if (use_steepest) else lambda f, df, Hf: newton_lin_eq(x, f, df,Hf,eps, A, c1, rho )
    )
    f2_opt = (
        lambda f, df, Hf: newton_lin_eq(np.linspace(-x, x, 2),f, df, Hf, eps, A, c1, rho)
    )
    xks1, iters1 = optimizer(f1, df1, Hf1)
    
    #xks2, iters2 = f2_opt(f2, df2, Hf2)
    
    #xks3, iters3 = optimizer(f3, df3, Hf3)
    
    xks4, iters4 = optimizer(f4, df4, Hf4)
    
    xks5, iters5 = optimizer(f5, df5, Hf5)
    
    return (xks1,xks4,xks5, iters1, iters4, iters5)

def test_method(x0, n, A, c1=0.0001, rho=0.5, eps=1.0e-10, use_steepest=True):
    xks1, xks2, xks3, xks4, xks5 = (
        [],
        [],
        [],
        [],
        []
    )
    it1, it2, it3, it4, it5 = (
        [],
        [],
        [],
        [],
        []
    )
    for i in tqdm(np.linspace(-x0, x0, n)):
        (_xks1,  _xks4, _xks5, _it1,  _it4,_it5) = test_runner(
            i, c1, rho, eps, A, use_steepest=use_steepest
        )
        xks1.append(_xks1)
        #xks2.append(_xks2)
        #xks3.append(_xks3)
        xks4.append(_xks4)
        xks5.append(_xks5)
        it1.append(_it1)
       # it2.append(_it2)
       # it3.append(_it3)
        it4.append(_it4)
        it5.append(_it5)
    
    return xks1,xks4,xks5, it1, it4, it5