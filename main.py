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
ap.add_argument('-spec', '--spec',default='vel',help='which spectra to visualise', type=str)
ap.add_argument('-sim', '--sim',default=None,help='which simulation', type=str)
ap.add_argument('-viz', '--viz',default=None,help='visualisation setting', type=str)
args = vars(ap.parse_args())


# Command Examples
############################################################################################################################################
"""

run main -sim "M2MA0.1"


"""

# Global variables
############################################################################################################################################

labelsMachA         = {"M2MA0.1":r"$\mathcal{M}_{\text{A}0} = 0.1$", "M4MA0.1":"$\mathcal{M}_{\text{A}0} = 0.1$", "M10MA0.1":r"$\mathcal{M}_{\text{A}0} = 0.1$",
                       "M20MA0.1":r"$\mathcal{M}_{\text{A}0} = 0.1$","M2MA0.5":r"$\mathcal{M}_{\text{A}0} = 0.5$","M4MA0.5":r"$\mathcal{M}_{\text{A}0} = 0.5$",
                       "M10MA0.5":r"$\mathcal{M}_{\text{A}0} = 0.5$","M20MA0.5":r"$\mathcal{M}_{\text{A}0} = 0.5$","M2MA1":r"$\mathcal{M}_{\text{A}0} = 1$",
                       "M4MA1":r"$\mathcal{M}_{\text{A}0} = 1$","M10MA1":r"$\mathcal{M}_{\text{A}0} = 1$", "M20MA1":r"$\mathcal{M}_{\text{A}0} = 1$",
                       "M2MA2":r"$\mathcal{M}_{\text{A}0} = 2$","M4MA2":r"$\mathcal{M}_{\text{A}0} = 2$", "M10MA2":r"$\mathcal{M}_{\text{A}0} = 2$",
                       "M20MA2":r"$\mathcal{M}_{\text{A}0} = 2$","M2MA10":r"$\mathcal{M}_{\text{A}0} = 10$","M4MA10":r"$\mathcal{M}_{\text{A}0} = 10$",
                       "M10MA10":r"$\mathcal{M}_{\text{A}0} = 10$","M20MA10":r"$\mathcal{M}_{\text{A}0} = 10$"}
labelsMach          = {"M2MA0.1":r"$\mathcal{M} = 2$", "M4MA0.1":"$\mathcal{M}= 4$", "M10MA0.1":r"$\mathcal{M}= 10$",
                       "M20MA0.1":r"$\mathcal{M} = 20$","M2MA0.5":r"$\mathcal{M} = 2$","M4MA0.5":r"$\mathcal{M} = 4$",
                       "M10MA0.5":r"$\mathcal{M} = 10$","M20MA0.5":r"$\mathcal{M} = 20$","M2MA1":r"$\mathcal{M} = 2$",
                       "M4MA1":r"$\mathcal{M} = 4$","M10MA1":r"$\mathcal{M} = 10$", "M20MA1":r"$\mathcal{M} = 20$",
                       "M2MA2":r"$\mathcal{M} = 2$","M4MA2":r"$\mathcal{M} = 4$", "M10MA2":r"$\mathcal{M} = 10$",
                       "M20MA2":r"$\mathcal{M} = 20$","M2MA10":r"$\mathcal{M} = 2$","M4MA10":r"$\mathcal{M} = 4$",
                       "M10MA10":r"$\mathcal{M} = 10$","M20MA10":r"$\mathcal{M} = 20$"}



# Functions
############################################################################################################################################

def powerSpectraAverage(specType):
    """

    """


    # define each of the directories
    dirs = ["M2MA0.1","M2MA0.5","M2MA1","M2MA2","M2MA10",
            "M4MA0.1","M4MA0.5","M4MA1","M4MA2","M4MA10",
            "M10MA0.1","M10MA0.5","M10MA1","M10MA2","M10MA10",
            "M20MA0.1","M20MA0.5","M20MA1","M20MA2","M20MA10"]

    # for each of the directories, read in a file, save the contents of the
    # power spectra and average it on the last iteration

    for dir in dirs:
        print("Reading from directory: {}".format(dir))
        iterCount = 0
        for iter in xrange(50,110):
            print("On iteration: {}".format(iter))

            specType, sim, iter, iterStr, specData,  contState = extractSpecData(specType,dir,iter)

            if contState == 1:
                print("Reached end of files.")
                print("Beginning averaging.")

                dataFrame = {'k': k,
                            'powerTot_mean': np.mean(powerTot,axis=0),  'powerTot_sd': np.std(powerTot,axis=0),
                            'powerLgt_mean': np.mean(powerLgt,axis=0),  'powerLgt_sd': np.std(powerLgt,axis=0),
                            'powerTrv_mean': np.mean(powerTrv,axis=0),  'powerTrv_sd': np.std(powerTrv,axis=0),
                            'yerrTot_mean': np.mean(yerrTot,axis=0),   'yerrTot_sd': np.std(yerrTot,axis=0),
                            'yerrLgt_mean': np.mean(yerrLgt,axis=0),   'yerrLgt_sd': np.std(yerrLgt,axis=0),
                            'yerrTrv_mean': np.mean(yerrTrv,axis=0),   'yerrTrv_sd': np.std(yerrTrv,axis=0)}
                print("Writing to pandas dataframe")
                df = pd.DataFrame(data=dataFrame)
                print("Saving data as: {}_{}.csv".format(sim,specType))
                df.to_csv('./averagedData/powerSpec/{}_{}.csv'.format(sim,specType))
                break

            if iterCount == 0:
                print("Creating arrays for the first iteration.")
                k           = np.array(specData['#01_KStag'])
                powerTot    = np.array(specData['#15_SpectFunctTot'])
                yerrTot     = np.array(specData['#16_SpectFunctTotSigma'])
                powerLgt    = np.array(specData['#11_SpectFunctLgt'])
                yerrLgt     = np.array(specData['#12_SpectFunctLgtSigma'])
                powerTrv    = np.array(specData['#13_SpectFunctTrv'])
                yerrTrv     = np.array(specData['#14_SpectFunctTrvSigma'])
            else:
                print("Appending arrays from iteration {}".format(iter))
                powerTot    = np.vstack((powerTot,np.array(specData['#15_SpectFunctTot'])))
                yerrTot     = np.vstack((yerrTot,np.array(specData['#16_SpectFunctTotSigma'])))
                powerLgt    = np.vstack((powerLgt,np.array(specData['#11_SpectFunctLgt'])))
                yerrLgt     = np.vstack((yerrLgt,np.array(specData['#12_SpectFunctLgtSigma'])))
                powerTrv    = np.vstack((powerTrv,np.array(specData['#13_SpectFunctTrv'])))
                yerrTrv     = np.vstack((yerrTrv,np.array(specData['#14_SpectFunctTrvSigma'])))


            iterCount += 1



def powerSpectrumPlot(ax,specData,type,label,color,turnoverTime,simLabel,movie):
    """
    DESCRIPTION:

    INPUT:

    OUTPUT:

    """

    # Extract appropriate data
    if movie:
        k       = specData['#01_KStag']
        if type == "tot":
            power   = specData['#15_SpectFunctTot']
            yerr    = specData['#16_SpectFunctTotSigma']
            zorder  = 10
        elif type == "lgt":
            power   = specData['#11_SpectFunctLgt']
            yerr    = specData['#12_SpectFunctLgtSigma']
            zorder  = 11
        elif type == "trv":
            power   = specData['#13_SpectFunctTrv']
            yerr    = specData['#14_SpectFunctTrvSigma']
            zorder  = 12
    else:
        k       = specData['k']
        if type == "tot":
            power   = specData['powerTot_mean']
            yerr    = specData['powerTot_sd']
            zorder  = 10
        elif type == "lgt":
            power   = specData['powerLgt_mean']
            yerr    = specData['powerLgt_sd']
            zorder  = 11
        elif type == "trv":
            power   = specData['powerTrv_mean']
            yerr    = specData['powerTrv_sd']
            zorder  = 12


    # Construct the error bars properly in log space
    upperVar = 10**(np.log10(power) + (1./np.log(10))*(yerr/power))
    lowerVar = 10**(np.log10(power) - (1./np.log(10))*(yerr/power))

    # fit the power spectrum index
    intercept, slope, sigma = fitPowerLaw(specData,'bayes',type,movie)
    linearFit   = lambda x: (10 **intercept[0]) * x ** slope[0]
    K41         = lambda x: (10**-1)*x**(-5/3.)
    Burg        = lambda x: (10**-0.95)*x**(-2.)
    kFit        = np.linspace(1.5,5)

    print("###############################################")
    print("slope     : {} +/- {}".format(slope[0],slope[1]))
    print("intercept : {} +/- {}".format(intercept[0],intercept[1]))
    print("###############################################")

    kComp = k**(-2)

    ax.axvline(x=10,color='blue',linestyle='--',linewidth=0.5,zorder=0)
    ax.axvline(x=20,color='blue',linestyle='--',linewidth=0.5,zorder=0)
    ax.annotate("Bayesian fit domain",color='blue',rotation=90,xy=(16,2),fontsize=fs-4)
    ax.plot(kFit,K41(kFit)/kFit**(-2),'k',linewidth=1)
    ax.annotate(r"$k^{-5/3}$",xy=(kFit[-1] + 0.25,K41(kFit)[-1]/kFit[-1]**(-2)),fontsize=fs-2)
    ax.plot(kFit,Burg(kFit)/kFit**(-2),'k',linewidth=1)
    ax.annotate(r"$k^{-2}$",xy=(kFit[-1] + 0.25,Burg(kFit)[-1]/kFit[-1]**(-2) - 0.5e-1 ),fontsize=fs-2)
    ax.scatter(k, power/ kComp, color=color, s=1,zorder=zorder,label=None)
    ax.plot(k,linearFit(k)/kComp,'blue',linewidth=1,
            label=label + " " +  r"$\sim k^{%s \pm %s}$" % (np.round(slope[0],2),np.round(slope[1],2)),
            color=color)
    ax.fill_between(k, lowerVar / kComp, upperVar / kComp, facecolor=color, alpha=0.2, zorder=zorder-9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$',fontsize=fs)
    ax.set_ylabel(r'$\mathscr{P}_{v}(k) / k^{-2}$',fontsize=fs)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.annotate(labelsMach[simLabel] + ", " + labelsMachA[simLabel], xy = (0.025,0.05),xycoords = xyCoords,fontsize=fs)
    if movie:
        ax.set_xlim(0,256)
        ax.set_ylim(10**-4,10**8)
        ax.annotate(r"$t/T = %s$" % turnoverTime, xy= (0.025,0.1), xycoords = xyCoords,fontsize=fs)

def fitPowerLaw(specData,method,type,movie):
    """
    DESCRIPTION:

    INPUT:

    OUTPUT:

    """

    if movie:
        # First to subset the k-space to fit between k = 10 and k = 20
        subset     = specData[ (specData['#01_KStag'] >= 10) & (specData['#01_KStag'] <= 20) ]
        k          = subset['#01_KStag']
        if type == "tot":
            power   = subset['#15_SpectFunctTot']
            yerr    = subset['#16_SpectFunctTotSigma']
        elif type == "lgt":
            power   = subset['#11_SpectFunctLgt']
            yerr    = subset['#12_SpectFunctLgtSigma']
        elif type == "trv":
            power   = subset['#13_SpectFunctTrv']
            yerr    = subset['#14_SpectFunctTrvSigma']
    else:
        subset      = specData[ (specData['k'] >= 10) & (specData['k'] <= 20) ]
        k           = subset['k']
        if type == "tot":
            power   = subset['powerTot_mean']
            yerr    = subset['powerTot_sd']
        elif type == "lgt":
            power   = subset['powerLgt_mean']
            yerr    = subset['powerLgt_sd']
        elif type == "trv":
            power   = subset['powerTrv_mean']
            yerr    = subset['powerTrv_sd']

    # now fit using whatever method
    if method == "bayes":
        intercept, slope, sigma = bayesFit(k,power)
    elif method == "leastSqu":
        intercept, slope, sigma = leastSquaresFit(k,power)


    return intercept, slope, sigma


def bayesFit(xdata,ydata):
    """
    DESCRIPTION:

    INPUT:

    OUTPUT:

    """

    print("Fitting a bayesian model with MCMC.")

    def logPrior(theta):
        """
        the log prior for the straight line model
        """

        alpha, beta, sigma = theta
        if sigma < 0:
            return -np.inf  # log(0)
        elif alpha < 0:
            return -np.inf
        elif beta > 0:
            return -np.inf
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
    nwalkers    = 10                        # number of MCMC walkers
    nburn       = 1000                      # "burn-in" period to let chains stabilize
    nsteps      = 2000                      # number of MCMC steps to take

    # Set theta near the maximum likelihood
    np.random.seed(0)
    startingGuesses = np.random.random((nwalkers, ndim))
    sampler         = emcee.EnsembleSampler(nwalkers, ndim, logPosterior, args=[xdata, ydata])
    sampler.run_mcmc(startingGuesses, nsteps)

    # Extract parameters
    emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    intercept   = (np.mean(emcee_trace[0]), np.std(emcee_trace[0]))
    slope       = (np.mean(emcee_trace[1]), np.std(emcee_trace[1]))
    sigma       = (np.mean(emcee_trace[2]))

    return intercept, slope, sigma


def leastSquaresFit(xdata,ydata):
    """

    """
    print("Fitting a least-squares model.")

    def linearModel(x,alpha,beta):
        return alpha + beta * x


    xdata = np.log10(np.array(xdata))
    ydata = np.log10(np.array(ydata))

    parms, err = curve_fit(linearModel,xdata=xdata,ydata=ydata)

    intercept   = (parms[0], err[0,0])
    slope       = (parms[1], err[1,1])
    sigma       = (0,0)

    return intercept, slope, sigma

def extractSpecData(specType,sim,iter):
    """
    DESCRIPTION:

    INPUT:

    OUTPUT:

    """

    contState   = 0
    iterStr     = str(iter).zfill(4)
    readDir     = "/Volumes/JamesBe/MHD/{}/Spec/".format(sim)
    try:
        fileName    = "Turb_hdf5_plt_cnt_{}_spect_{}.dat".format(iterStr,specType)
        specData    = pd.read_table(readDir + fileName,skiprows=5,sep="\s+")
    except:
        print("An exception occured in reading the file at iter: {}.".format(iter))
        contState = 1
        return specType, sim, 0, 0, 0, contState

    return specType, sim, iter, iterStr, specData, contState

def createMovie(specType,sim,tend):
    """
    DESCRIPTION:

    INPUT:

    OUTPUT:

    """

    for iter in xrange(50,tend):
        print("Iteration: {}".format(iter))
        specType, sim, iter, iterStr, specData, _ = extractSpecData(specType=specType,sim=sim,iter=iter)

        f, ax = plt.subplots(dpi=200)
        powerSpectrumPlot(ax,specData,"tot",r"$\mathscr{P}_{\text{tot}}$",'black',iter/10.,sim,True)
        powerSpectrumPlot(ax,specData,"trv",r"$\mathscr{P}_{\text{trav}}$",'red',iter/10.,sim,True)
        powerSpectrumPlot(ax,specData,"lgt",r"$\mathscr{P}_{\text{long}}$",'purple',iter/10.,sim,True)
        plt.legend(fontsize=fs-2,loc="upper right")
        plt.tight_layout()
        plt.savefig("plots/spect_{}_{}_{}.png".format(sim,specType,iterStr),dpi=200)
        plt.close()


# Working Script
############################################################################################################################################

if __name__ == "__main__":
    compile = False

    if compile:
        powerSpectraAverage("vels")
        powerSpectraAverage("mags")
        powerSpectraAverage("rho3")

    sims        = ["M2MA0.1","M2MA0.5","M2MA1","M2MA2","M2MA10"]
    specType    = "vels"
    readDir     = "./averagedData/powerSpec/"

    f, ax = plt.subplots(dpi=200)
    for sim in sims:
        specData    = pd.read_csv(readDir + sim + "_" + specType + ".csv")
        powerSpectrumPlot(ax,specData,"tot",r"$\mathscr{P}_{\text{tot}}$",'black',0,sim,False)
    plt.legend(fontsize=fs-2,loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()
