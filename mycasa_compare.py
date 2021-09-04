"""
Standalone routines for comparing FITS and CASA images.

history:
2021-08-12   created by TS
2021-08-13   minor bug fix
2021-08-18   minor bug fix (x) by takano
2021-08-18   delete_intermediate=True changed to delete all CASA images
2021-08-31   major bug fix in unit_to_kelvin, handle "perplanebeams" header
Toshiki Saito@Nichidai/NAOJ
"""


### input
ximage = "ngc1068_b8_1_12m+7m_ci10.image"
yimage = "ngc1068_b8_1_7m_ci10.image"
delete_intermediate = True


#######################
# main part from here #
#######################

import os, sys, glob, copy, inspect
import numpy as np
import pyfits
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
plt.ioff()

#############
# functions #
#############

def fits_to_casa(imagename):
    """
    import input to CASA. Then Stokes axis to the last axis.
    """

    # import fits to casa
    if not os.path.isdir(imagename):
        print("# " + imagename + " is FITS. Create CASA format if not present.")

        if not glob.glob(imagename + ".image"):
            os.system("rm -rf " + imagename + ".image")
            importfits(
                fitsimage = imagename,
                imagename = imagename + ".image",
                )

        imagename += ".image"

    header = imhead(imagename,mode="list")
    shape = header["shape"]

    # move Stokes axis to the last axis
    if "Stokes" in header.values():
        if shape[-1]!=1:
            if not glob.glob(imagename + ".trans"):
                index = np.where(shape==1)[0][0]
                order = np.array(range(len(shape)))
                order = np.array(np.r_[order[order!=index],index], dtype=str)
                order = "".join(order)

                print("# imstrans " + str(order) + " 0123.")
                imtrans(
                imagename = imagename,
                outfile   = imagename + ".trans",
                order     = order,
                )

            imagename += ".trans"

    return imagename

def get_header(imagename):
    """
    get some header info useful for image alignment.
    """

    header = imhead(imagename,mode="list")
    print("# run imhead: " + imagename)

    if "Stokes" in header.values():
        shape = header["shape"][:-1]
    else:
        shape = header["shape"]

    pix   = abs(header["cdelt1"]) * 3600 * 180 / np.pi

    if "beammajor" in header:
        bmaj = np.round(header["beammajor"]["value"],3)
        bmin = np.round(header["beamminor"]["value"],3)
    elif "perplanebeams" in header:
        bminfo = header["perplanebeams"]
        if "beams" in bminfo.keys():
            bmaj = np.round(bminfo["beams"]["*0"]["*0"]["major"]["value"],3)
            bmin = np.round(bminfo["beams"]["*0"]["*0"]["minor"]["value"],3)
        else:
            bmaj = np.round(bminfo["*0"]["major"]["value"],3) + 0.001
            bmin = np.round(bminfo["*0"]["minor"]["value"],3) + 0.001

    box   = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
    unit  = header["bunit"]
    freq  = header["restfreq"][0] / 1e9
    width = np.round(abs(header["cdelt3"]) / 1e9 / freq * 299792, 2) # km/s

    return shape, pix, bmaj, bmin, box, unit, freq, width

def unit_to_kelvin(imagename,unit,beam,freq):
    """
    convert unit from Jy/beam to Kelvin.
    """

    if unit=="K":
        return imagename

    elif unit=="Jy/beam":
        print("# " + imagename + " units to Kelvin if not present.")
        if not glob.glob(imagename + ".K"):
            print("unit conversion")
            immath(
                imagename = imagename,
                outfile   = imagename + ".K",
                expr      = "IM0*" + str(1.222e6 / beam / beam / freq**2),
                )

        imagename += ".K"
        imhead(imagename,mode="put",hdkey="bunit",hdvalue="K")

        return imagename

    else:
        print("# no units in " + imagename + "!")
        return None

def match_beam(ximage,yimage,xbmaj,xbmin,ybmaj,ybmin):
    """
    convolve beams to the same round beam (= largest bmaj).
    """

    beam = np.max([xbmaj, xbmin, ybmaj, ybmin])

    if not xbmaj==xbmin==ybmaj==ybmin:
        print("# convolve images to " + str(beam) + " arcsec if not present.")

        if not xbmaj==xbmin==beam:
            if not glob.glob(ximage + ".smooth"):
                print("# convolve ximage.")
                imsmooth(
                    imagename = ximage,
                    outfile   = ximage + ".smooth",
                    targetres = True,
                    major     = str(beam) + "arcsec",
                    minor     = str(beam) + "arcsec",
                    pa        = "0deg",
                    )

            ximage += ".smooth"

        if not ybmaj==ybmin==beam:
            if not glob.glob(yimage + ".smooth"):
                print("# convolve yimage.")
                imsmooth(
                    imagename = yimage,
                    outfile   = yimage + ".smooth",
                    targetres = True,
                    major     = str(beam) + "arcsec",
                    minor     = str(beam) + "arcsec",
                    pa        = "0deg",
                    )

            yimage += ".smooth"

    return ximage, yimage, beam

def xyalign_maps(ximage,yimage,xpix,ypix):
    """
    aling two images in the xy plane.
    """

    if xpix>ypix:
        print("# regrid " + yimage + " if not present.")
        if not glob.glob(yimage + ".regrid"):
            print("# regrid " + yimage + ".")
            imregrid(
                imagename = yimage,
                template  = ximage,
                output    = yimage + ".regrid",
                axes      = [0,1],
                )

        yimage += ".regrid"

    elif xpix<ypix:
        print("# regrid " + ximage + " if not present.")
        if not glob.glob(ximage + ".regrid"):
            print("# regrid " + ximage + ".")
            imregrid(
                imagename = ximage,
                template  = yimage,
                output    = ximage + ".regrid",
                axes      = [0,1],
                )

        ximage += ".regrid"

    elif xpix==ypix:
        return ximage, yimage

    return ximage, yimage

def vrebin_maps(ximage,yimage,xwidth,ywidth,xshape,yshape):
    """
    align two images in the v axis.
    """

    if len(xshape[xshape!=1])!=3:
        print("# ximage is not 3D!")
        return None

    if len(yshape[yshape!=1])!=3:
        print("# yimage is not 3D!")
        return None

    if xwidth==ywidth:
        return ximage, yimage

    elif xwidth>ywidth:
        print("# rebin " + yimage + " if not present.")
        if not glob.glob(yimage + ".rebin"):
            print("# rebin " + yimage + ".")
            imregrid(
                imagename = yimage,
                template  = ximage,
                output    = yimage + ".rebin",
                axes      = [2],
                )

        yimage += ".rebin"

    elif xwidth<ywidth:
        print("# rebin " + ximage + " if not present.")
        if not glob.glob(ximage + ".rebin"):
            print("# rebin " + ximage + ".")
            imregrid(
                imagename = ximage,
                template  = yimage,
                output    = ximage + ".rebin",
                axes      = [2],
                )

        ximage += ".rebin"

    return ximage, yimage

def plot_hist(data,imagename,snr_for_fit=1.0):
    """
    plot voxel distribution in the input data. Zero will be ignored.
    output units will be mK.
    """

    # prepare
    data = data.flatten()
    data = data[data!=0] * 1000
    output = "plot_hist_" + imagename + ".png"

    # get voxel histogram
    bins   = int( np.ceil( np.log2(len(data))+1 ) * 10 )
    p84    = np.percentile(data, 16) * -1
    hrange = [-3*p84, 3*p84]
    hist   = np.histogram(data, bins=bins, range=hrange)

    x,y   = hist[1][:-1], hist[0]/float(np.sum(hist[0]))
    histx = x[x<p84*snr_for_fit]
    histy = y[x<p84*snr_for_fit]

    # fit
    x_bestfit = np.linspace(hrange[0], hrange[1], bins)
    popt,_ = curve_fit(func1, histx, histy, p0=[np.max(histy),p84], maxfev=10000)

    # best fit
    peak      = popt[0]
    rms_mK    = np.round(popt[1],1)
    y_bestfit = func1(x_bestfit, peak, rms_mK)

    # plot
    ymax  = np.max(y) * 1.1
    cpos  = "tomato"
    cneg  = "deepskyblue"
    cfit  = "black"
    snr_a = 1.0
    snr_b = 3.0

    tpos  = "positive side"
    tneg  = "(flipped) negative side"
    tfit  = "best fit Gaussian"
    ha    = "left"
    w     = "bold"
    tsnr_a = "1$\sigma$ = "+str(np.round(rms_mK*snr_a,1))+" mK"
    tsnr_b = "3$\sigma$ = "+str(np.round(rms_mK*snr_b,1))+" mK"

    fig = plt.figure(figsize=(10,10))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20

    gs = gridspec.GridSpec(nrows=30, ncols=30)
    ax = plt.subplot(gs[0:30,0:30])
    ax.grid(axis="both", ls="--")

    ax.step(x, y, color=cpos, lw=4, where="mid")
    ax.bar(x, y, lw=0, color=cpos, alpha=0.2, width=x[1]-x[0], align="center")
    ax.step(-x, y, color=cneg, lw=4, where="mid")
    ax.bar(-x, y, lw=0, color=cneg, alpha=0.2, width=x[1]-x[0], align="center")
    ax.plot(x_bestfit, y_bestfit, "k-", lw=3)
    ax.plot([snr_a*rms_mK,snr_a*rms_mK], [0,ymax], "k--", lw=1)
    ax.plot([snr_b*rms_mK,snr_b*rms_mK], [0,ymax], "k--", lw=1)

    tf = ax.transAxes
    ax.text(0.45,0.93,tpos,color=cpos,transform=tf,horizontalalignment=ha,weight=w)
    ax.text(0.45,0.88,tneg,color=cneg,transform=tf,horizontalalignment=ha,weight=w)
    ax.text(0.45,0.83,tfit,color=cfit,transform=tf,horizontalalignment=ha,weight=w)
    ax.text(rms_mK*snr_a,ymax*0.9,tsnr_a,rotation=90)
    ax.text(rms_mK*snr_b,ymax*0.5,tsnr_b,rotation=90)

    ax.set_title(imagename)
    ax.set_xlabel("Pixel value (mK)")
    ax.set_ylabel("Pixel count")
    ax.set_xlim([0,5*p84])
    ax.set_ylim([0,ymax])

    plt.savefig(output, dpi=200)

    return rms_mK / 1000. # Kelvin

def func1(x, a, c):
    return a*np.exp(-(x)**2/(2*c**2))

def plot_scatter(xdata,ydata,ximage,yimage,xrms,yrms):
    """
    plot xdata vs. ydata scatter.
    """

    xdata,ydata = xdata.flatten(),ydata.flatten()
    cut = np.where((xdata!=0) & (ydata!=0))

    # plot linear
    xdata,ydata = xdata[cut],ydata[cut]
    lim = [np.min([xdata, ydata]), np.max([xdata, ydata])]
    output = "plot_scatter_"+ximage+"_vs_"+yimage+".png"

    fig = plt.figure(figsize=(10,10))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    ax.grid(axis="both", ls="--")

    thres_u = np.where( (xdata>=xrms*3) & (ydata>=yrms*3) )
    thres_l = np.where( (xdata<xrms*3) | (ydata<yrms*3) )
    ax.scatter(xdata[thres_u], ydata[thres_u], c="tomato", s=30, linewidths=0)
    ax.scatter(xdata[thres_l], ydata[thres_l], c="gray", s=10, linewidths=0)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ximage (K)")
    ax.set_ylabel("yimage (K)")
    ax.set_title(ximage+" vs. \n"+yimage)
    plt.savefig(output, dpi=200)

    # plot log
    xydata = np.r_[xdata[xdata>0], ydata[ydata>0]]
    lim = [np.log10(np.min(xydata)), np.log10(np.max(xydata))]
    output = "plot_scatter_log_"+ximage+"_vs_"+yimage+".png"

    fig = plt.figure(figsize=(10,10))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    ax.grid(axis="both", ls="--")

    ax.scatter(np.log10(xdata[thres_u]), np.log10(ydata[thres_u]), c="tomato", s=30, linewidths=0)
    ax.scatter(np.log10(xdata[thres_l]), np.log10(ydata[thres_l]), c="gray", s=10, linewidths=0)

    ax.plot(lim, [lim[0]+1.0,lim[1]+1.0], "k--", lw=1)
    ax.plot(lim, lim, "k--", lw=3)
    ax.plot(lim, [lim[0]-1.0,lim[1]-1.0], "k--", lw=1)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("log ximage (K)")
    ax.set_ylabel("log yimage (K)")
    ax.set_title(ximage+" vs. \n"+yimage)

    plt.savefig(output, dpi=200)

def plot_spectra(xdata,ydata,ximage,yimage,xrms,yrms):
    """
    plot stats spectra in K units and different spectra in % units.
    Zero will be ignored.
    """

    xdata_linefree = np.where(xdata<xrms*3,xdata,np.nan)
    ydata_linefree = np.where(ydata<yrms*3,ydata,np.nan)

    xdata[xdata==0] = np.nan
    ydata[ydata==0] = np.nan
    axis=(0,1)

    output = "plot_spectra_"+ximage+"_vs_"+yimage+".png"

    # xstats
    cut = np.nan_to_num(np.nanmax(xdata,axis=axis))
    cut = np.where(cut!=0)
    xmax = np.nan_to_num(np.nanmax(xdata,axis=axis))[cut]
    xmin = np.nan_to_num(np.nanmin(xdata,axis=axis))[cut]
    xp16 = np.nan_to_num(np.nanpercentile(xdata,16,axis=axis))[cut]
    xp50 = np.nan_to_num(np.nanpercentile(xdata,50,axis=axis))[cut]
    xp84 = np.nan_to_num(np.nanpercentile(xdata,84,axis=axis))[cut]
    x    = np.array(range(len(xmax[cut])))

    xrms = np.sqrt(np.nanmean(np.square(xdata_linefree),axis=axis))
    x2   = np.array(range(len(xrms[cut])))

    # ystats
    ymax = np.nan_to_num(np.nanmax(ydata,axis=axis))[cut]
    ymin = np.nan_to_num(np.nanmin(ydata,axis=axis))[cut]
    yp16 = np.nan_to_num(np.nanpercentile(ydata,16,axis=axis))[cut]
    yp50 = np.nan_to_num(np.nanpercentile(ydata,50,axis=axis))[cut]
    yp84 = np.nan_to_num(np.nanpercentile(ydata,84,axis=axis))[cut]

    yrms = np.sqrt(np.nanmean(np.square(ydata_linefree),axis=axis))
    y2   = np.array(range(len(yrms[cut])))

    # prepare for plot
    cmax = cm.rainbow(0/4.)
    cp84 = cm.rainbow(1/4.)
    cp50 = cm.rainbow(2/4.)
    cp16 = cm.rainbow(3/4.)
    cmin = cm.rainbow(4/4.)
    crms = "black"
    tmax = "max"
    tp84 = "84$^{th}$"
    tp50 = "50$^{th}$"
    tp16 = "16$^{th}$"
    tmin = "min"
    trms = "rms"
    ha   = "left"
    w    = "bold"
    xlim = [0,len(x)]

    # plot
    fig = plt.figure(figsize=(10,10))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.subplots_adjust(bottom=0.10, left=0.15, right=0.85, top=0.95)

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax1 = plt.subplot(gs[0:9,0:30])
    ax2 = plt.subplot(gs[10:19,0:30])
    ax3 = plt.subplot(gs[20:29,0:30])
    ax1.grid(axis="both", ls="--")
    ax2.grid(axis="both", ls="--")
    ax3.grid(axis="both", ls="--")
    ax1b = ax1.twinx()
    ax2b = ax2.twinx()

    ax1.axes.xaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticklabels([])

    ax1b.step(x2, xrms*1000., color="black", lw=4, alpha=1.0, where="mid")
    ax2b.step(y2, yrms*1000., color="black", lw=4, alpha=1.0, where="mid")

    diff = (xmax-ymax)/xmax*100
    ax1.step(x, xmax, color=cmax, lw=2, alpha=1.0, where="mid")
    ax2.step(x, ymax, color=cmax, lw=2, alpha=1.0, where="mid")
    ax3.step(x, diff, color=cmax, lw=2, alpha=1.0, where="mid")

    diff = (xp84-yp84)/xp84*100
    ax1.step(x, xp84, color=cp84, lw=2, alpha=1.0, where="mid")
    ax2.step(x, yp84, color=cp84, lw=2, alpha=1.0, where="mid")
    ax3.step(x, diff, color=cp84, lw=2, alpha=1.0, where="mid")

    diff = (xp50-yp50)/xp50*100
    ax1.step(x, xp50, color=cp50, lw=2, alpha=1.0, where="mid")
    ax2.step(x, yp50, color=cp50, lw=2, alpha=1.0, where="mid")
    ax3.step(x, diff, color=cp50, lw=2, alpha=1.0, where="mid")

    diff = (xp16-yp16)/xp16*100
    ax1.step(x, xp16, color=cp16, lw=2, alpha=1.0, where="mid")
    ax2.step(x, yp16, color=cp16, lw=2, alpha=1.0, where="mid")
    ax3.step(x, diff, color=cp16, lw=2, alpha=1.0, where="mid")

    diff = (xmin-ymin)/xmin*100
    ax1.step(x, xmin, color=cmin, lw=2, alpha=1.0, where="mid")
    ax2.step(x, ymin, color=cmin, lw=2, alpha=1.0, where="mid")
    ax3.step(x, diff, color=cmin, lw=2, alpha=1.0, where="mid")

    tf = ax1.transAxes
    ax1.text(0.03,0.85,tmax,color=cmax,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.75,tp84,color=cp84,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.65,tp50,color=cp50,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.55,tp16,color=cp16,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.45,tmin,color=cmin,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.35,trms,color=crms,transform=tf,horizontalalignment=ha,weight=w)

    ax1.set_ylabel("ximage (K)")
    ax2.set_ylabel("yimage (K)")
    ax3.set_ylabel("(x-y)/x * 100 (%)")
    ax3.set_xlabel("channel")
    ax1b.set_ylabel("xrms (mK)")
    ax2b.set_ylabel("yrms (mK)")

    ax3.set_ylim([-200,200])
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)
    ax1b.set_xlim(xlim)
    ax2b.set_xlim(xlim)

    plt.savefig(output, dpi=200)

def imval_data(imagename):

    shape = imhead(imagename,mode="list")["shape"]
    box = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)

    print("# imval data (may take time)")
    data = imval(imagename,box=box)
    data = data["data"] * data["mask"]
    data[np.isinf(data)] = 0
    data = np.nan_to_num(data)

    return data

def run_comparison(ximage,yimage,delete_intermediate):
    """
    """
    
    ########
    # main #
    ########

    ### make sure CASA format
    print("#########################")
    print("# make sure CASA format #")
    print("#########################")
    xim = fits_to_casa(ximage)
    yim = fits_to_casa(yimage)

    ### get image info
    print("\n##################")
    print("# get image info #")
    print("##################")
    xshape, xpix, xbmaj, xbmin, xbox, xunit, xfreq, xwidth = get_header(xim)
    yshape, ypix, ybmaj, ybmin, ybox, yunit, yfreq, ywidth = get_header(yim)

    ### make sure round beam
    print("\n####################################")
    print("# make sure round and matched beam #")
    print("####################################")
    xim, yim, beam = match_beam(xim, yim, xbmaj, xbmin, ybmaj, ybmin)

    ### units to Kelvin
    print("\n##########################")
    print("# make sure Kelvin units #")
    print("##########################")
    xim = unit_to_kelvin(xim, xunit, beam, xfreq)
    yim = unit_to_kelvin(yim, yunit, beam, yfreq)

    ### align maps
    print("\n##############")
    print("# align maps #")
    print("##############")
    xim, yim = xyalign_maps(xim, yim, xpix, ypix)

    ### rebin maps
    print("\n##############")
    print("# rebin maps #")
    print("##############")
    xim, yim = vrebin_maps(xim, yim, xwidth, ywidth, xshape, yshape)

    ### analysis
    print("\n############")
    print("# analysis #")
    print("############")
    x = imval_data(xim)
    y = imval_data(yim)

    # survive nonzero pixels in both maps
    xdata = np.where((x!=0)&(y!=0), x, 0)
    ydata = np.where((x!=0)&(y!=0), y, 0)

    # plot
    xrms = plot_hist(xdata, ximage)
    yrms = plot_hist(ydata, yimage)
    plot_scatter(xdata, ydata, ximage, yimage, xrms, yrms)
    plot_spectra(xdata, ydata, ximage, yimage, xrms, yrms)

    ### clean up
    print("\n############")
    print("# clean up #")
    print("############")
    os.system("rm -rf importfits.last")
    os.system("rm -rf imhead.last")
    os.system("rm -rf immath.last")
    os.system("rm -rf imsmooth.last")
    os.system("rm -rf imregrid.last")
    os.system("rm -rf imtrans.last")
    os.system("rm -rf imval.last")

    if delete_intermediate==True:
        xdel = glob.glob(ximage + ".*")
        xdel.sort()
        xdel = " ".join(xdel)
        os.system("rm -rf " + xdel)

        ydel = glob.glob(yimage + ".*")
        ydel.sort()
        ydel = " ".join(ydel)
        os.system("rm -rf " + ydel)

run_comparison(ximage,yimage,delete_intermediate)
