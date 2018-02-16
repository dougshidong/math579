import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
from mpltools import annotation
# x'(t) = M*x(t), x(0) = x0, t = [0, tf]
# Memory allocation
def run_case(tvec,M,h):
    xgd[1,:] = gradient_descent(xgd[0,:],tvec[1],h)
    xhb[1,:] = xgd[1,:]
    xnv[1,:] = xgd[1,:]
    print(xnv[0,:])
    print(xnv[1,:])
    for i, t in enumerate(tvec[2:]):
        xgd[i+2,:] = gradient_descent(xgd[i+1,:],t,h)
        xhb[i+2,:] = heavy_ball(xhb[i+1,:],xhb[i,:], t,h)
        xnv[i+2,:] = nesterov(xnv[i+1,:],xnv[i,:], t,h)
        print(xgd[i+2,:])

# Define methods
def gradient_descent(x,t,h): return np.matmul(I - h*M, x)
def heavy_ball(x,xm1,t,h):
    beta = (1.0-np.sqrt(mu/L)) / (1.0+np.sqrt(mu/L))
    c1   = (1.0-beta*beta) / np.sqrt(mu*L)
    c2   = beta*beta
    return x + c2 * (x-xm1) - c1 * np.matmul(M,x)
def nesterov(x,xm1,t,h):
    beta = (1.0-np.sqrt(mu/L)) / (1.0+np.sqrt(mu/L))
    x = ((1.0+beta)*x-beta*xm1 + (0.9/L)*(beta*np.matmul(M,xm1) \
        - (1.0+beta) * np.matmul(M,x))).flatten()
    #    = xm1 - (0.9/L) * np.matmul(M,xm1)
    #yp1  = x   - (0.9/L) * np.matmul(M,x)
    #x    = yp1 + beta * (yp1 - y)
    return x

pp=PdfPages('math579hw2_figures.pdf')



def plota(title):
    plt.figure()
#   plt.semilogy(tvec,abs(xgd[range(nt),0])+1e-17,'-o',label='FE')
#   plt.semilogy(tvec,abs(xgd[range(nt),1])+1e-17,'--o',label='FE')
#   plt.semilogy(tvec,abs(xhb[range(nt),0])+1e-17,'-^',label='HB')
#   plt.semilogy(tvec,abs(xhb[range(nt),1])+1e-17,'--^',label='HB')
#   plt.semilogy(tvec,abs(xnv[range(nt),0])+1e-17,'-s',label='NV')
#   plt.semilogy(tvec,abs(xnv[range(nt),1])+1e-17,'--s',label='NV')

    plt.semilogy(tvec,abs(xgd[range(nt),0])+1e-17,'-',label='FE')
    plt.semilogy(tvec,abs(xhb[range(nt),0])+1e-17,'--',label='HB')
    plt.semilogy(tvec,abs(xnv[range(nt),0])+1e-17,'-.',label='NV')

    plt.xlabel('Iterations')
    plt.ylabel('x1')
    plt.title(title); plt.legend(loc=4); plt.tight_layout(); pp.savefig()
for i in range(6):
    maxt = 10000; nerr = 12;
    xhb = np.empty([maxt,2]); xnv = np.empty([maxt,2]); xgd = np.empty([maxt,2])
    x0 = np.array([0.91,0.5]);
    xhb[0,:] = x0; xnv[0,:] = x0; xgd[0,:] = x0

    I = np.matrix([[1, 0],
                   [0, 1]])
    mu = 10**-i*1.0
    L  = 1.0
    M = np.matrix([[mu, 0],
                   [0, L]])
    C = L/mu#000 # mu = 1; C = L/mu = L
    tf = 1000
    h = 1.9/(L+mu)
    nt = int((tf-0)/h) + 1
    tvec = np.linspace(0,tf,num=nt,endpoint=True)
    run_case(tvec,M,h)
    plota('C = %d'%C)#Solution to x\' = Mx, M = -2.0')

pp.close()
