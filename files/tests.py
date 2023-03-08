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

def test_runner(x, c1,c2, eps):
    #x = np.linspace(-x0, x0, d)
    optimizer = (
        lambda f, df, Hf: BFGS(x, f, df,eps,c1,c2)
    )
    f2_opt = (
        lambda f, df, Hf: BFGS(np.linspace(-x, x, 2),f,df,eps,c1,c2)
    )
    xks1, iters1 = optimizer(f1, df1, Hf1)
    
    xks2, iters2 = f2_opt(f2, df2, Hf2)
    
    xks3, iters3 = optimizer(f3, df3, Hf3)
    
    xks4, iters4 = optimizer(f4, df4, Hf4)
    
    xks5, iters5 = optimizer(f5, df5, Hf5)
    
    return (xks1,xks2,xks3,xks4,xks5, iters1, iters2, iters3, iters4, iters5)
def test_method(x0, n, c1=0.0001, c2=0.25, eps=1.0e-10):
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
        (_xks1, _xks2, _xks3, _xks4, _xks5, _it1, _it2, _it3, _it4,_it5) = test_runner(
            i, c1, c2, eps
        )
        xks1.append(_xks1)
        xks2.append(_xks2)
        xks3.append(_xks3)
        xks4.append(_xks4)
        xks5.append(_xks5)
        it1.append(_it1)
        it2.append(_it2)
        it3.append(_it3)
        it4.append(_it4)
        it5.append(_it5)
    
    return xks1,xks2, xks3,xks4,xks5, it1, it2, it3, it4, it5
""" def test_method_(d, x0, n, cstart=0.0001,cstop=0.5,rho=0.5,eps=1.e-8,use_steepest=True):
    xs1_count, xs2_count, xs3_count, xs4_count, xs5_count = np.zeros((5, 1))
    xs1, xs2, xs3, xs4, xs5 = np.zeros((5, 1))
    dfs1, dfs2, dfs3, dfs4, dfs5 = [], [], [], [], []
    aks1, aks2, aks3, aks4, aks5 = [], [], [], [], []
    j = 0
    for i in tqdm(np.linspace(cstart, cstop, n)):
            (
                _xs1,
                _dfs1,
                _aks1,
                _xs2,
                _dfs2,
                _aks2,
                _xs3,
                _dfs3,
                _aks3,
                _xs4,
                _dfs4,
                _aks4,
                _xs5,
                _dfs5,
                _aks5,
            ) = test_runner(d, x, i, rho, eps, use_steepest=use_steepest)
            dfs1.append(_dfs1)
            dfs2.append(_dfs2)
            dfs3.append(_dfs3)
            dfs4.append(_dfs4)
            dfs5.append(_dfs5)
            #print(j)
            j += 1
    return dfs1, dfs2, dfs3, dfs4, dfs5 """