# import sympy

# from sympy.abc import z,x,y
import sympy
from sympy.utilities.autowrap import ufuncify
from sympy import I

import numpy as np
from matplotlib import pyplot as plt

import numba

z = sympy.symbols('z', complex=True)
sympy.utilities.codegen.COMPLEX_ALLOWED = True


# def test():
#     x = sympy.abc.x
#     expr = x**2
#     exprd = expr.diff(x)
#     print(exprd)

def rootFinderCompiler(expr):
    """takes a sympy expr, uses the ufuncify method to create a compiled function
    which will perform 1 iteration of newton's method on an entire meshgrid"""
    exprPrime = expr.diff(z)
    iterate = z - (expr/exprPrime)


    compExpr = ufuncify(z,iterate,backend='f2py')
    return compExpr

def displayConvergence(expr, xlim, ylim, xres, yres, iterations, _cmap='inferno'):
    x = np.linspace(xlim[0],xlim[1],xres)
    y = np.linspace(ylim[0],ylim[1],yres)

    xx,yy = np.meshgrid(x,y,indexing = 'xy')

    z = xx + yy*1j

    compExpr = rootFinderCompiler(expr)

    for i in range(iterations):
        z = compExpr(z)

    z = np.angle(z).reshape(xres,yres).T
    plt.imshow(z,cmap=_cmap)
    # plt.xticks(np.linspace(0,xres,5),np.linspace(xlim[0],xlim[1],5))
    # plt.yticks(np.linspace(0,yres,5),-np.linspace(ylim[0],ylim[1],5))
    plt.xticks([])
    plt.yticks([])


    plt.show()