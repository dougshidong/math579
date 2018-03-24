import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Advection-diffusion: u_t + f(u_x) = g(u_xx)
# f(u_x) = c*u_x
# g(u_xx) = s*u_xx

CFL = 0.1
a = -5.0; b = 5.0;
pp=PdfPages('math579hw4_figures.pdf')

def init_solution(icase, x, u_init):
    if(icase=='SmoothBump'): u_init = np.exp(-2*x*x)
    elif(icase=='SharpBump'): u_init = np.maximum(0*x, 1-x*x)
    elif(icase=='FlatSine'):
        for i, xi in enumerate(x):
            if(xi<=-np.pi/2):    u_init[i] = 1
            elif(xi>=np.pi/2):   u_init[i] = -1
            else:                u_init[i] = np.sin(3*xi)
    elif(icase=='Triangle'):
        u_init = [max(1-abs(xi),0) for xi in x]
    return u_init

# Define discretization
def euler_explicit(dt, u, rhs): return u - dt*rhs
def diffusion(u,dx,s,suxx):
    n = len(u)
    for i in range(1,n-1): suxx[i] = (u[i+1] - 2*u[i] + u[i-1])*s/(dx*dx)
    suxx[0] = (u[1] - 2*u[0] + u[0])*s/(dx*dx)
    suxx[n-1] = (u[n-1] - 2*u[n-1] + u[n-2])*s/(dx*dx)
    return suxx
def upwind(u,dx,c,cux):
    n = len(u)
    for i in range(1,n): cux[i] = (u[i] - u[i-1])*c/dx
    cux[0] = 0
    return cux
def upwind2(u,dx,c,cux):
    n = len(u)
    for i in range(2,n): cux[i] = (3*u[i] - 4*u[i-1] + u[i-2])*c/(2*dx)
    cux[1] = (3*u[1] - 4*u[0] + u[0])*c/(2*dx)
    cux[0] = 0
    return cux

def solve(icase, iflux, tf, s, n, u, u0):
    # Domain definition
    x = np.linspace(a,b,n)
    dx = (b-a)/(n-1)
    rhs = np.empty(n,dtype=np.float64())
    cux = np.empty(n,dtype=np.float64())
    suxx = np.empty(n,dtype=np.float64())
    # Advection speed
    if(icase=='FlatSine'): c = 1.5
    if(icase=='Triangle'): c = 2
    # Initialize solution
    u0 = init_solution(icase, x, u0)
    u = u0
    # Setup timestep
    nt = int(np.ceil(tf/(dx/c*CFL)))
    dt = tf/nt

    # Solve using Euler-explicit
    for ti in range(nt):
        if(iflux=='Up1'): cux = upwind(u, dx, c, cux)
        if(iflux=='Up2'): cux = upwind2(u, dx, c, cux)
        rhs = cux - diffusion(u, dx, s, suxx)
        u = euler_explicit(dt, u, rhs)

    return u, u0


def plotFigure4():
    n = 401
    x = np.linspace(a,b,n)
    u = np.empty(n,dtype=np.float64())
    u0 = np.empty(n,dtype=np.float64())

    tf = 1.0
    s = 0.0 # Diffusion

    icase = 'FlatSine'; iflux = 'Up1'
    [u, u0] = solve(icase, iflux, tf, s, n, u, u0)

    plt.figure(); plt.title('Sine Advection')
    plt.plot(x,u0,'-k',label='Initial')
    plt.plot(x,u,'-o',ms=5,label=iflux)

    icase = 'FlatSine'; iflux = 'Up2'
    [u, u0] = solve(icase, iflux, tf, s, n, u, u0)
    plt.plot(x,u,'-^',ms=5,label=iflux)

    if(icase=='FlatSine'): plt.axis([-2,5,-1.2,1.2])
    if(icase=='Triangle'): plt.axis([-2,4,-0.1,1.2])
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout();pp.savefig()

    icase = 'Triangle'; iflux = 'Up1'
    [u, u0] = solve(icase, iflux, tf, s, n, u, u0)

    plt.figure(); plt.title('Triangle Advection')
    plt.plot(x,u0,'-k',label='Initial')
    plt.plot(x,u,'-o',ms=5,label=iflux)

    icase = 'Triangle'; iflux = 'Up2'
    [u, u0] = solve(icase, iflux, tf, s, n, u, u0)
    plt.plot(x,u,'-^',ms=5,label=iflux)

    if(icase=='FlatSine'): plt.axis([-2,5,-1.2,1.2])
    if(icase=='Triangle'): plt.axis([-2,5,-0.1,1.2])
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout();pp.savefig()


def loopDiffusion(icase, iflux, title):
    n = 201
    x = np.linspace(a,b,n)
    u = np.empty(n,dtype=np.float64())
    u0 = np.empty(n,dtype=np.float64())
    tf = 1.0
    s  = 0
    [u, u0] = solve(icase, iflux, tf, s, n, u, u0)

    diffusionvalues = [0,0.005,0.01,0.02,0.04,0.08,0.16]

    plt.figure(); plt.title(title)
    plt.plot(x,u0,'-k',label='Initial')
    for s in diffusionvalues:
        [u, u0] = solve(icase, iflux, tf, s, n, u, u0)
        plt.plot(x,u,'-o',ms=5,label=r'$\sigma^2 = %4.3f$'%s)
    if(icase=='FlatSine'): plt.axis([-2,5,-1.2,1.2])
    if(icase=='Triangle'): plt.axis([-2,5,-0.1,1.2])
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout();pp.savefig()

def plotDiffusion():

    icase = 'FlatSine'; iflux = 'Up1'
    loopDiffusion(icase, iflux,
        'Sine Advection-Diffusion with 1st Order Upwind')

    icase = 'FlatSine'; iflux = 'Up2'
    loopDiffusion(icase, iflux,
        'Sine Advection-Diffusion with 2nd Order Upwind')

    icase = 'Triangle'; iflux = 'Up1'
    loopDiffusion(icase, iflux,
        'Triangle Advection-Diffusion with 1st Order Upwind')

    icase = 'Triangle'; iflux = 'Up2'
    loopDiffusion(icase, iflux,
        'Triangle Advection-Diffusion with 2nd Order Upwind')

def loopDX(icase, iflux, title):
    n = 201;
    x = np.linspace(a,b,n)
    u = np.empty(n,dtype=np.float64())
    u0 = np.empty(n,dtype=np.float64())
    tf = 1.0
    s  = 0.005
    [u, u0] = solve(icase, iflux, tf, s, n, u, u0)

    plt.figure(); plt.title(title)
    plt.plot(x,u0,'-k',label='Initial')

    nvalues = [51,101,151,201,301]
    for n in nvalues:
        x = np.linspace(a,b,n)
        u = np.empty(n,dtype=np.float64())
        u0 = np.empty(n,dtype=np.float64())
        [u, u0] = solve(icase, iflux, tf, s, n, u, u0)
        plt.plot(x,u,'-o',ms=5,label=r'$dx = %4.3f$'%(10.0/(n-1.0)))
    if(icase=='FlatSine'): plt.axis([-2,5,-1.2,1.2])
    if(icase=='Triangle'): plt.axis([-2,5,-0.1,1.2])
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout();pp.savefig()

def plotDX():
    icase = 'FlatSine'; iflux = 'Up1'
    loopDX(icase, iflux,
        'Sine Advection-Diffusion with 1st Order Upwind')

    icase = 'FlatSine'; iflux = 'Up2'
    loopDX(icase, iflux,
        'Sine Advection-Diffusion with 2nd Order Upwind')

    icase = 'Triangle'; iflux = 'Up1'
    loopDX(icase, iflux,
        'Triangle Advection-Diffusion with 1st Order Upwind')

    icase = 'Triangle'; iflux = 'Up2'
    loopDX(icase, iflux,
        'Triangle Advection-Diffusion with 2nd Order Upwind')

plotFigure4()
plotDiffusion()
plotDX()
pp.close()
