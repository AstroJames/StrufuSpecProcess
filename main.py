#!/usr/bin/env python2

""""

    Title:          Spectra Average and Fit Code
    Notes:          main file
    Author:         James Beattie
    First Created:  2 / Mar / 2020

"""

import pandas as pd
import py_compile
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

if __name__ == "__main__":

    readDir     = "/Volumes/JamesBe/MHD/M20MA0.1/Spec/"
    fileName    = "Turb_hdf5_plt_cnt_0050_spect_vels.dat"
    table       = pd.read_table(readDir + fileName,sep='\t',skiprows=5)
