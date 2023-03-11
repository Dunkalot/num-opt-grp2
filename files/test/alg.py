import numpy as np
import math

# def backtrack(x, p, alpha, c, rho, f, df):
#     a = alpha
#     cnt = 0 
#     while  f(x + p * a) > f(x) + c * a * p.T @df(x):
#         a = rho * a
#         cnt += 1
#     return a


def backtrack(func, grad, x, direction, init_alpha, c, r):
    alpha = init_alpha
    iterations = 0
    while func(x + alpha * direction) > func(x) + c*alpha*np.dot(direction , grad(x)):
        alpha *= r
        iterations += 1
    return alpha

def CG(df, Hf, tolerance): 
    a_k = 1
    x_k = 0
    df_k = df
    p_k = -df_k
    i = 0
    while True:
        a_k = -(np.dot(p_k.T, df_k))/(np.dot(p_k.T, np.dot(Hf,p_k)))
        x_k = x_k + a_k * p_k
        df_k = Hf @ x_k + df
        i+=1
        if np.linalg.norm(df_k) < tolerance:
            break
        p_k = -df_k + np.dot(df_k.T, np.dot(Hf,p_k))/np.dot(p_k.T,np.dot(Hf,p_k)) * p_k
        
    return x_k, i

def approx_newton(x0, c1, rho ,f, df, Hf, tolerance, eta): 
    x = x0
    pks = []
    _i = []
    e_k = 0
    a_k = 0
    p_k = 1
    while np.linalg.norm(df(x)) > tolerance:
        eta_k = eta(df(x))
        e_k = eta_k*np.linalg.norm(df(x))
        p_k, i = CG(df(x),Hf(x),e_k)
        pks.append(p_k)
        _i.append(i)
        a_k = backtrack(x, p_k, 1.0, c1, rho,f,df)
        x = x + a_k*p_k
    return np.array(pks), np.array(_i)

def newton_alg(x, f, c, rho, eps, df, hf):
    xs = []
    dfs = []
    aks = []
    while np.linalg.norm(df(x)) > eps:
        pk = None
        eig_vals, eig_vecs = np.linalg.eig(hf(x))
        eig_vecs = eig_vecs.T
        if np.all(eig_vals > 0):
            pk = -np.dot(np.linalg.pinv(hf(x)), df(x))
        else:
            H = np.zeros((eig_vecs.shape[1], eig_vecs.shape[1]), dtype=np.float32)
            for i, eig_vec in enumerate(eig_vecs):
                H += (1 / np.abs(eig_vals[i])) * np.outer(eig_vec, eig_vec.T)
            pk = -H.dot(df(x))
   
        dfs.append(df(x))
        ak = backtrack(x, pk, 1.0, c, rho, f, df)
        aks.append(ak)
        x = x + ak * pk
        xs.append(x)
    return  np.array(dfs), np.array(xs)

def steepest_descent(x0, f, c, rho, eps, df):
    Bk = 1.0
    x = x0
    xs = []
    dfs = []
    aks = []
    cnt = 0
    while np.linalg.norm(df(x)) > eps:
        pk = -df(x)
        ak = backtrack(x, pk, Bk, c, rho, f, df)
        x = x + ak * pk
        xs.append(x)
        dfs.append(df(x))
        Bk = ak / rho
        aks.append(ak)
        cnt += 1
    return np.array(dfs), np.array(aks)

def wolfe_search(x, f, df, pk, a_init, c1,c2): 
    #print("Calls to wolfe search")
    def g(_a):
        return f(x+_a*pk)
    def dg(_a):
        # TODO: Should we transpose the gradient?
        return df(x+_a*pk).dot(pk)
    l = 0 
    u = a_init 
    i = 0
    while True:
        i +=1
        if (g(u) > g(0) + c1 * u * dg(0)) or (g(u) > g(l)):
            break
        if np.abs(dg(u)) < (c2 * np.abs(dg(0))):
            return u , i
        if dg(u) > 0: 
            break
        else:
            u*=2
        
    while True: 
        i +=1
        # (0,1)
        a = (l+u)/2 # 0.5
        #print(f"a: {a}, l: {l}, u: {u}")
        if (g(a) > (g(0) + c1 * a * dg(0))) or (g(a) > g(l)):
            u = a 
        else:
            if np.abs(dg(a)) < (c2 * np.abs(dg(0))):
                #print(f"A = {a}")
                return a , i
            if dg(a) < 0:
                    # (0.5, 1)
                l = a 
            else: 
                # (0, 0.5)
                u = a
                
def BFGS(x, f, df, eps, c1,c2):
    Hk = np.identity(x.shape[0])
    pk = 0 
    ak = 0 
    yk = 0
    sk = 0
    xk = x
    dfs = []
    iters = []
    while (np.linalg.norm(df(xk)) > eps):
        pk = -Hk.dot(df(xk))
        ak , i = wolfe_search(xk,f,df, pk, 1.0, c1,c2)
        iters.append(i)
        temp_df = df(xk)
        dfs.append(temp_df)
        xk = xk + ak*pk
        yk = df(xk)-temp_df
        sk = ak*pk
        # np.atleast_2d(x) gives a row vector
        fst_top = np.inner(sk,yk) + np.dot(np.dot(np.atleast_2d(yk),Hk), np.atleast_2d(yk).T) 
        # maybe np.dot(yk,sk.T) should be np.outer(yk, sk.T)
        snd_top = np.dot(np.dot(Hk,np.atleast_2d(yk).T),np.atleast_2d(sk)) + np.dot(np.outer(sk,yk),Hk)
        bot = np.inner(sk,yk)
        Hk = Hk + (fst_top/(bot**2))*np.outer(sk,sk) - snd_top/bot
    return np.array(dfs), np.array(iters)

def newton_lin_eq(x,f, df, Hf, eps, A,b, c1, rho, max_iter=1000):
    xk = x
    Bk = 0
    pk = 0
    ak = 0 
    xks = [x]
    dfs = [df(x)]
    lambdas = []
    iters = 0
    Mk=np.eye(A.shape[1]) - A.T@np.linalg.inv(A@A.T)@A

    while True:
        eig_vals, eig_vecs = np.linalg.eig(Hf(xk))
        eig_vecs = eig_vecs.T
        if np.all(eig_vals > 0):
            Bk = Hf(xk)
        else:
            hs = [abs(eig_val) * np.outer(eig_vec, eig_vec) for eig_val, eig_vec in zip(eig_vals, eig_vecs)]
            Bk = np.sum(hs, axis=0)
        M = np.block([[Bk, A.T], [A,np.zeros((A.shape[0],A.shape[0]))]])
        v = np.block([-df(xk), np.zeros(A.shape[0])])
        solved = np.linalg.solve(M,v)
        pk = solved[0:A.shape[1]]
        l = solved[A.shape[1]]
        ak = backtrack(f,df,xk,pk,1.0,c1,rho)
        xk = xk + ak*pk
        xks.append(xk)
        dfs.append(np.linalg.norm(df(xk)))
        iters = iters +1
        rate = np.abs((f(xk) - f(xks[-2])))/f(xks[-2])
        #print(np.dot(A,xk)+b)
        #print(df(xk) + A.T@l)
        if np.linalg.norm(np.dot(A,xk)+b)<eps or iters >= max_iter or rate < eps or np.linalg.norm(df(xk) + np.dot(A,l)) < eps:
            break
    return xks, dfs

def steepest_lin_eq(x, f, df, eps, A, c1, rho, max_iter = 1000):
    Bk = 1.0
    M = np.eye(A.shape[1]) - A.T@np.linalg.inv(A@A.T)@A
    pk = np.zeros((A.shape[1],))
   # ak = 0
    xk = x
    xks = [x]
    dfs = [df(x)]
    iters = 0
    max_iter = max_iter
    print("start")
    while np.abs(np.dot(pk,df(xk))) < eps and iters <= max_iter and np.all(np.dot(A,pk)>eps):  
        pk = -M@df(xk)
        ak = backtrack(f,df,xk,pk,Bk,c1,rho)
        xk = xk + ak*pk
        Bk = ak/rho
        iters = iters +1
        xks.append(xk)
        dfs.append(df(xk))
        iters = iters+1
        rate = np.abs((f(xk) - f(xks[-2])))/f(xks[-2])
        print(iters)
        if rate < eps:
            break
    return xks, dfs

def get_point(A,b,x):
    return x-np.linalg.pinv(A)@(A@x - b)

def generate_A(m,n):
    rd = np.random.RandomState(2023)
    A = rd.uniform(-10,10,(m,n))
    
    #A = np.random.rand(m,n)
    
    while np.linalg.matrix_rank(A) != m:
        A = rd.uniform(-10,10,(m,n))
    return A

