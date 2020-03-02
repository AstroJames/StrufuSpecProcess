#!/usr/bin/env python2

""""

    Title:          Spectra Average and Fit Code
    Notes:          main file
    Author:         James Beattie
    First Created:  2 / Mar / 2020

"""

import pandas as pd
import py_compile
import emcee
py_compile.compile("header.py")
from header import *

# Command Line Arguments
############################################################################################################################################
ap = argparse.ArgumentParser(description='command line inputs')
ap.add_argument('-spectra', '--spectra',default=None,help='visualisation setting', type=str)
ap.add_argument('-viz', '--viz',default=None,help='visualisation setting', type=str)
args = vars(ap.parse_args())


# Command Examples
############################################################################################################################################
"""

run processSpectra


"""


# Functions
############################################################################################################################################

def powerSpectrumPlot(ax,specData,label,color):
    """


    """

    k       = specData['#01_KStag']
    power   = specData['#15_SpectFunctTot']

    ax.plot(k,power,label=label,color=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$',fontsize=fs)
    ax.set_ylabel(r'$\mathscr{P}_{\text{vel}}$',fontsize=fs)

def fitPowerLaw(specData):
    """


    """

    # first to subset the k-space to fit between k = 10 and k = 20
    subset     = specData[ (specData['#01_KStag'] >= 10) & (specData['#01_KStag'] <= 20) ]
    k          = subset['#01_KStag']
    power      = subset['#15_SpectFunctTot']

    # now fit using whatever method
    #bayesParms = bayesFit(k,power)


    return k,power


def bayesFit(xdata,ydata):
    """


    """

    print("Fitting a bayesian model with MCMC.")

    def logPrior(theta):
        """
        the log prior for the straight line model
        """

        alpha, beta, sigma = theta
        if sigma < 0:
            return -np.inf  # log(0)
        else:
            return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

    def logLikelihood(theta, x, y):
        """
        the log likelihood function
        """

        alpha, beta, sigma = theta
        yModel = alpha + beta * x
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - yModel) ** 2 / sigma ** 2)

    def logPosterior(theta, x, y):
        """
        the logarithm of the posterior is the sum of the prior and likelihood function.
        """

        return logPrior(theta) + logLikelihood(theta, x, y)


    xdata       = np.log10(np.array(xdata)) # transform the x data
    ydata       = np.log10(np.array(ydata)) # transform the y data
    ndim        = 3                         # number of parameters in the model
    nwalkers    = 50                        # number of MCMC walkers
    nburn       = 1000                      # "burn-in" period to let chains stabilize
    nsteps      = 2000                      # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    np.random.seed(0)
    startingGuesses = np.random.random((nwalkers, ndim))
    sampler         = emcee.EnsembleSampler(nwalkers, ndim, logPosterior, args=[xdata, ydata])
    sampler.run_mcmc(startingGuesses, nsteps)
    print("done")

    emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T


    return emcee_trace


# Working Script
############################################################################################################################################

if __name__ == "__main__":

    readDir     = "/Volumes/JamesBe/MHD/M2MA10/Spec/"
    fileName    = "Turb_hdf5_plt_cnt_0050_spect_vels.dat"
    specData    = pd.read_table(readDir + fileName,skiprows=5,sep="\s+")

    k,power       = fitPowerLaw(specData)

    f, ax = plt.subplots(dpi=200)
    powerSpectrumPlot(ax,specData,label=r"$\mathscr{P}_{\text{tot}}$",color='k')
    powerSpectrumPlot(ax,specData,label=r"$\mathscr{P}_{\text{trav}}$",color='red')
    powerSpectrumPlot(ax,specData,label=r"$\mathscr{P}_{\text{long}}$",color='blue')
    plt.legend()
    plt.show()
