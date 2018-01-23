import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
from mpltools import annotation
def plota(title):
    plt.figure()
    plt.plot(tvec,xfe[range(nt)],'-o',label='FE')
    plt.plot(tvec,xbe[range(nt)],'-s',label='BE')
    plt.plot(tvec,xtm[range(nt)],'-^',label='TM')
    plt.plot(tvec,exact(tvec),'-k',label='Exact')
    plt.title(title); plt.legend(); plt.tight_layout(); pp.savefig()
def plotb(title):
    plt.figure()
    plt.semilogy(tvec,xfe[range(nt)],'-o',label='FE')
    plt.semilogy(tvec,xbe[range(nt)],'-s',label='BE')
    plt.semilogy(tvec,xtm[range(nt)],'-^',label='TM')
    plt.semilogy(tvec,exact(tvec),'-k',label='Exact')
    plt.title(title); plt.legend(); plt.tight_layout(); pp.savefig()
def plotc(title):
    plt.figure()
    plt.loglog(hvec,efe,'-o',label='FE')
    plt.loglog(hvec,ebe,'-s',label='BE')
    plt.loglog(hvec,etm,'-^',label='TM')
    slope1, intercept, _, _, _ = linregress(np.log(hvec),np.log(ebe))
    slope2, intercept, _, _, _ = linregress(np.log(hvec),np.log(etm))
    annotation.slope_marker((hvec[8], ebe[8]), (slope1, 1))
    annotation.slope_marker((hvec[8], etm[8]), (slope2, 1))
    plt.title(title); plt.legend(); plt.tight_layout(); pp.savefig()

# x'(t) = M*x(t), x(0) = x0, t = [0, tf]
# Memory allocation
maxt = 10000; nerr = 12;
xfe = np.empty(maxt); xbe = np.empty(maxt); xtm = np.empty(maxt)
efe = np.empty(nerr); ebe = np.empty(nerr); etm = np.empty(nerr)
x0 = 0.1;
xfe[0] = x0; xbe[0] = x0; xtm[0] = x0

# Define methods
def forward_euler(x,t,h): return (1.0 + h*M) * x
def backward_euler(x,t,h): return x / (1.0 - h*M)
def trapezoid_method(x,t,h): return (1.0 + 0.5*h*M) / (1.0 - 0.5*h*M) * x
def exact(t): return np.exp(M*t)*x0
def run_case(tvec,M,h):
    for i, t in enumerate(tvec[1:]):
        xfe[i+1] = forward_euler(xfe[i],t,h)
        xbe[i+1] = backward_euler(xbe[i],t,h)
        xtm[i+1] = trapezoid_method(xtm[i],t,h)

pp=PdfPages('math579hw1_figures.pdf')

tf = 2.0; M = -2.0; h = 0.25; nt = int((tf-0)/h) + 1
tvec = np.linspace(0,tf,num=nt,endpoint=True)
run_case(tvec,M,h)
plota('Solution to x\' = Mx, M = -2.0')

tf = 2.0; M = -8.5; h = 0.25; nt = int((tf-0)/h) + 1
tvec = np.linspace(0,tf,num=nt,endpoint=True)
run_case(tvec,M,h)
plota('Solution to x\' = Mx, M = -8.5')

tf = 6.0; M = -2.0; h = 0.25; nt = int((tf-0)/h) + 1
tvec = np.linspace(0,tf,num=nt,endpoint=True)
run_case(tvec,M,h)
plotb('Convergence rate to 0')

tf = 2.0; M = -2.0;
hvec = [1.0/2.0**(i+1) for i in range(nerr)]
for i,h in enumerate(hvec):
    nt = int((tf-0)/h) + 1
    tvec = np.linspace(0,tf,num=nt,endpoint=True)
    run_case(tvec,M,h)
    efe[i] = abs(xfe[nt-1] - exact(tf))
    ebe[i] = abs(xbe[nt-1] - exact(tf))
    etm[i] = abs(xtm[nt-1] - exact(tf))
plotc('Error at t = 2.0 vs timestep h')

pp.close()
