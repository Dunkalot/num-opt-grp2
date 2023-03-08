import numpy as np

#These are the 5 case studies, functions f1 to f4. gradients are computed by the functions df1-df5 and Hessians by Hf1-Hf5
#The optimal value of a function in dimenson n can be obtained via x_opt(funcname,n), for example x_opt("f1",2)

def h(x,q):
    return (np.log(1+np.exp(-np.abs(q*x)))+np.maximum(q*x,0))/q
def dh(x,q):
    qx=np.clip(q*x,-100,100)
    return 1/(1+np.exp(-qx))

def Hh(x,q):
    return q*dh(x,q)*dh(-x,q)
def f1(x,alpha=1000):
    w = alpha**(np.linspace(0,1,len(x)))
    return np.sum(w*x**2)

def f2(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def f3(x,epsilon=10000.0,alpha=1000):
    return np.log1p(f1(x,alpha)*epsilon)

def f4(x, q=1000):
    return np.sum(h(x,q)**2 + 100*h(-x,q)**2)

def f5(x):
    factor = 2*np.linspace(1,2,len(x))
    return np.sum(np.abs(x)**factor)

def df1(x,alpha=1000):
    w = alpha**(np.linspace(0,1,len(x))) 
    return 2*w*x
def Hf1(x,alpha=1000):
    w = alpha**(np.linspace(0,1,len(x)))
    return 2*np.diag(w)

def df2(x):
    (1-x[0])**2+100*(x[1]-x[0]**2)**2
    g0 = -2*(1-x[0])-400*(x[1]-x[0]**2)*x[0]
    g1 = 200*(x[1]-x[0]**2)
    return np.array([g0,g1])

def Hf2(x):
    H00 = 2 -400*x[1]+1200*x[0]**2
    H01 = -400*x[0]
    H11 = 200
    return np.array([[H00,H01],[H01,H11]])

def df3(x,epsilon=10000,alpha=1000):
    fx=f1(x,alpha)
    return df1(x,alpha)/(fx+1.0/epsilon)

def Hf3(x,epsilon=10000,alpha=1000):
    fx=f1(x,alpha)
    dfx=df1(x,alpha)
    Hdx=Hf1(x,alpha)
    g = dfx/(fx+1.0/epsilon)
    return Hdx/(fx+1.0/epsilon)- np.outer(g,g)

def df4(x, q=1000):
    return 2*h(x,q)*dh(x,q) - 200*h(-x,q)*dh(-x,q)

def Hf4(x, q=1000):
    h1 = dh(x,q)**2
    h2 = h(x,q)*Hh(x,q)
    f3 = -dh(-x,q)**2
    f4 = -h(-x,q)*Hh(-x,q)
    return np.diag(2*(h1+h2) - 200*(f3+f4))

def df5(x):
    factor = 2*np.linspace(1,2,len(x))
    return np.sign(x)*factor*np.abs(x)**np.maximum(factor-1,0.0)

def Hf5(x):
    factor = 2*np.linspace(1,2,len(x))
    Hgx =factor*(factor-1)*np.abs(x)**np.maximum(factor-2,0.0)
    return np.diag(Hgx)


def x_opt(fname, dims):
    if fname=="f2":
        return np.ones(2)
    if fname=="f4":
        return np.ones(dims)*0.0019093310847216710905011
    return np.zeros(dims)