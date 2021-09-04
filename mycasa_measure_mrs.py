import os, glob
import numpy as np

msfiles = glob.glob("/lfs02/saitots/imaging/ngc1068_b8/*.ms")

for this_msfile in msfiles:
    this_msfile_str = this_msfile.split("/")[-1]

    msmd.open(this_msfile)
    freq_Hz = msmd.meanfreq(0)
    msmd.done()

    wavelength_m = 299792458.0 / freq_Hz
    minbaseline = au.getProjectedBaselineStats(this_msfile)["min"]
    mrk_kl = minbaseline / wavelength_m / 1000.

    print("# ms name = " + this_msfile_str)
    print("# MRS     = " + mrk_kl + " klambda")
