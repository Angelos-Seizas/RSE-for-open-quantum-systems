# -*- coding: utf-8 -*-
"""
This module contains a set of functions useful for the analysis of the resonant states
of open one-dimensional quantum well systems using the Resonant State Expansion (RSE)
and Newton-Raphson's root-finding algorithm.
Usage examples and further details can be found in the examples.py file.

Author: Angelos Seizas
Email: seizasa1@cardiff.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def RSE(γ,even,odd,*p,verbose=True,plot=False,real_range=30,imag_range=1,eig=False):
    """
    Computes the wavenumbers of resonance states in one-dimensional quantum well
    systems using the resonant state expansion (RSE).

    Parameters
    ----------
    γ : float
        Strength of the basis wells

    even : ndarray
           Even states of the unperturbed potential

    odd : ndarray
          Odd states of the unperturbed potential

    *p : tuple of floats
        Perturber parameters represented as pairs of position and strength values.

    verbose : bool, optional
              If True (default), print informative messages during execution.
              Set to False to suppress printing. Default is True.

    plot : bool, optional
           Set to True to generate a complex 2D plot of the wavenumber solutions.

    real_range : float
                 Range of the real axis in the plot. Default is 30.
    
    imag_range : float
                 Range of the imaginary axis in the plot. Default is 1.

    eig : bool, optional
          Set to True to compute the corresponding eigenvectors for each wavenumber
          solution.

    Returns
    -------
    K : Wavenumbers of the resonance states.
    P : The corresponding eigenvectors of the solutions (if 'eig' is set to 1)
    """
    p = np.array(p).reshape(-1,2)
    start_time = time.time()
    k = np.concatenate((even,odd))
    Be = 1/(2*np.sqrt(1-(γ+2j*even)**(-1))) #even normalisation constant
    Bo = 1/(2*np.sqrt(-(1-(γ+2j*odd)**(-1)))) #odd normalisation constant
    μ = [] #even wavefunctions
    ζ = [] #odd wavefunctions
    for b in p[:,0]:
        e_n = Be*np.cos(even*b)
        μ.append(e_n)
        o_n = Bo*1j*np.sin(odd*b)
        ζ.append(o_n)
    μ = np.asarray(μ)
    ζ = np.asarray(ζ)

    'Even-even mixing'
    EE = []
    for n in μ:
        ee = np.zeros((len(even),len(even)),dtype=complex)
        for i in range(0,len(even)):
            ee[i] = 2*n[i]*n
        EE.append(ee)

    'Odd-even mixing'
    OE = []
    for n in range(0,len(ζ)):
        oe = np.zeros((len(odd),len(even)),dtype=complex)
        for i in range(0,len(odd)):
            oe[i] = 2*ζ[n][i]*μ[n]
        OE.append(oe)

    'Even-odd mixing'
    EO = []
    for n in range(0,len(μ)):
        eo = np.zeros((len(even),len(odd)),dtype=complex)
        for i in range(0,len(even)):
            eo[i] = 2*μ[n][i]*ζ[n]
        EO.append(eo)

    'Odd-odd mixing'
    OO = []
    for n in ζ:
        oo = np.zeros((len(odd),len(odd)),dtype=complex)
        for i in range(0,len(odd)):
            oo[i] = 2*n[i]*n
        OO.append(oo)

    K = []
    for i in range(0,len(EE)):
        K1 = np.concatenate((EE[i],OE[i]))
        K2 = np.concatenate((EO[i],OO[i]))
        K.append(np.concatenate((K1,K2),axis=1))

    V = 0
    for i in range(0,len(K)):
        V = V+(-p[:,1][i]*2*K[i])
    H_nm = np.diag(k) + V / (2*np.sqrt(k)*np.sqrt(k.reshape(len(k),1)))
    if eig==False:
        K = np.linalg.eigvals(H_nm)
    elif eig==True:
        K, P = np.linalg.eig(H_nm)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose == True:
        print(f"Elapsed time for RSE computation: {elapsed_time:.3f} seconds.")
    if plot == True:
        plt.figure()
        plt.scatter(K.real, K.imag, marker='x', color='purple')
        plt.xlabel('Re(κ)')
        plt.ylabel('Im(κ)')
        plt.xlim(-real_range,real_range)
        plt.ylim(-imag_range,imag_range)
        plt.grid()

    if eig==True:
        return K, P
    else:
        return K

def d_func(f, x, h=1e-12):
    """
    Computes the symmetric derivative of a function.

    Parameters
    ----------

    f : function
        The function for which to compute the derivative.

    x : float
        The point at which to compute the derivative.

    h : float, optional
        The step size for numerical differentiation. Default its 1e-12.

    """
    return (f(x+h) - f(x-h))/(2*h)

def ComplexCmap(func, x_range, y_range):
    """
    Plots a 2D colormap for a complex function.

    Parameters
    ----------

    func : function
           The complex function to visualise.

    n1 : float
         The range of x values for the colormap.

    n2 : float
         The range of y values for the colormap.

    """
    real_part = np.linspace(-x_range, x_range, 1000)
    imag_part = np.linspace(-y_range, y_range, 1000)
    rp, ip = np.meshgrid(real_part, imag_part)
    domain = rp + ip * 1j

    z = np.abs(func(domain))
    plt.pcolormesh(real_part, imag_part, np.log(z / z.max()), cmap="summer", shading="auto")
    plt.colorbar()

def newton(f,df,roots,n,verbose=True,trace_path=False,tol=1e-10,N=1000):
    """
    The Newton-Raphson algorithm.

    Parameters
    ----------

    f : function
        Input function.

    df : function
         Derivative of the input function.

    roots : list
            The list where the roots will be appended.

    n : int
        Range of real guess values
    
    verbose : bool, optional
              If True (default), print informative messages during execution.
              Set to False to suppress printing. Default is True.

    trace_path : bool, optional
                 If True, then for each iteration it will generate a 2D complex
                 colormap of the function visualising the path taken by the
                 algorithm to find the root. Default is False.

    tol : float, optional
          Tolerance error. Default is 1e-10.

    N : int, optional
        Maximum number of iterations. Default is 1000.

    """
    for k in np.arange(-n, n, 1): #Creating guess values
        k0 = k - 1j*2
        X_path = [k0]
        for l in range(N):
            if df(f,k0)==0.0:
                if verbose == True:
                    print("\nError: Division by zero")
                break
            d = f(k0) / df(f,k0)
            k0 = k0 - d
            X_path.append(k0)
            if abs(k0.real) < tol or abs(k0.imag) < tol:
                pass
        if abs(f(k0).real) < tol and abs(f(k0).imag) < tol:
            if verbose == True:
                print("\nRequired root is ",str(k0))
            roots.append([k0.real, k0.imag])
            Xpath = np.array(X_path)
            if trace_path == True:
                plt.figure()
                ComplexCmap(f,n,10)
                plt.plot(k0.real, k0.imag, 'wo')
                plt.plot(Xpath.real, Xpath.imag, 'r-')
                plt.ylim(-10,10)
                plt.xlabel("Real part of f(k)")
                plt.ylabel("Imaginary part of f(k)")
                plt.title("Newton-Raphson root-finding path")
                plt.show()
                X_path = []
            elif verbose == True:
                print("\nNot convergent")

def basis(γ,p,N,verbose=True,a=1,trace_path=0,tol=1e-10,n=1000):
    """
    Computes the even or odd wavenumber solutions for the double well that is
    used as the basis for the resonant state expansion.

    Parameters
    ----------
    γ : float
        Strength of the basis wells.

    p : float
        Parity

    N : int
        Range of real guess values for Newton-Raphson's method.

    verbose : bool, optional
              If True (default), print informative messages during execution.
              Set to False to suppress printing. Default is True.

    a : float, optional
        Midpoint distance between the basis wells. Default is 1.

    trace_path : bool, optional
                 If True, then for each iteration it will generate a 2D complex
                 colormap of the function visualising the path taken by the
                 algorithm to find the root.

    tol : float, optional
          Tolerance error.

    n : int, optional
        Maximum number of iterations.

    Returns
    -------
    roots : An array containing the computed wavenumbers that form the RSE basis.

    """
    f = lambda k: 1+((2j*k)/γ) + p * np.exp(2j*k*a)
    roots = []
    start_time = time.time()
    newton(f,d_func,roots,N,verbose,trace_path,tol,n)
    end_time = time.time()

    roots = np.array([row[0]+row[1]*1j for row in roots])
    roots = np.sort(np.unique(roots))

    #Remove closely spaced roots up to a certain toelrance.
    tolerance = 1e-13
    diff = np.empty(roots.shape)
    diff[0] = np.inf # always retain the first element
    diff[1:] = np.diff(roots)
    mask = diff > tolerance
    roots = roots[mask]

    elapsed_time = end_time - start_time
    if verbose == True:
        print(f"Elapsed time for the Newton-Raphson computation: {elapsed_time:.3f} seconds.")

    return roots

def RSE_verif(even,odd,b,β,γ,real_range=30,imag_range=1,verbose=True,a=1):
  """
  Computes the wavenumbers of the resonance states of an asymmetric triple quantum
  well using both the resonant state expansion and Newton-Raphson's method and compares
  their results graphically.

  Parameters
  ----------
  even : ndarray
         Even states of the unperturbed potential.

  odd : ndarray
        Odd states of the unperturbed potential.

  b : float
      Position of the perturber

  γ : float
      Strength of the perturber

  real_range : float
               Range of the real axis in the plot. Default is 30.
    
  imag_range : float
               Range of the imaginary axis in the plot. Default is 1.

  verbose : bool, optional
            If True (default), print informative messages during execution.
            Set to False to suppress printing. Default is True.

  a : float, optional
      Midpoint distance between the basis wells. Default is 1.

  Returns
  -------
  K : ndarray
      Wavenumber solutions computed using the resonant state expansion.

  roots : ndarray
          Wavenumber solutions computed using Newton-Raphson's method.
  """
  K = RSE(γ,even,odd,b,β)

  ξ = lambda k: (np.exp(1*2j*k*a))/(1+1j*2*k/γ)
  η = lambda k: (2*1j*k)/β
  f = lambda k: (ξ(k)**2)*(1-η(k))-2*ξ(k)*np.cos(2*k*b)+1+η(k) # the secular equation for an asymmetric triple well system

  roots=[]
  NR2_start_time = time.time()
  newton(f,d_func,roots,100,verbose)
  NR2_end_time = time.time()

  roots = np.array([row[0]+row[1]*1j for row in roots])
  roots = np.sort(np.unique(roots))
  #Remove closely spaced roots up to a certain toelrance.
  tolerance = 1e-13
  diff = np.empty(roots.shape)
  diff[0] = np.inf # always retain the first element
  diff[1:] = np.diff(roots)
  mask = diff > tolerance
  roots = roots[mask]

  fig, ax = plt.subplots()
  ComplexCmap(f,30,2)
  plt.scatter(K.real,K.imag,marker='x',color='r',label='RSE')
  plt.scatter(roots.real,roots.imag,color='yellow',marker='.',label='Newton-Raphson')
  ax.set_xlabel('Re(κ)')
  ax.set_ylabel('Im(κ)')
  plt.grid()
  plt.legend()
  plt.xlim(-real_range,real_range)
  plt.ylim(-imag_range,imag_range)

  textstr = '\n'.join((
    r'$\bf{System\ parameters:}$',
    f'γ = {γ:.3f}',
    f'b = {b:.3f}',
    f'β = {β:.3f}'
  ))
  props = dict(boxstyle='round', facecolor='#999999', alpha=0.5)
  ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props)

  elapsed_time = NR2_end_time - NR2_start_time
  if verbose == True:
    print(f"Elapsed time for the Newton-Raphson computation of the system: {elapsed_time:.3f} seconds.")

  return K,roots
