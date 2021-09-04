"""
Standalone routines that evaluate clean output.

history:
2021-08-20   created by TS
2021-08-22   add voxel histograms and Gaussian fit
2021-08-25   add header to the html report
2021-08-27   add stats spectra (min-16-50-84-max, rms)
2021-09-01   add only_html
plan-to-do   add residual auto correlation vs. neighboring channel
plan-to-do   add ms info e.g., uv rms
Toshiki Saito@Nichidai/NAOJ
"""

### input
preimagename = "ngc1068_b8_1_7m_ci10"
imsize = 18 # arcsec
imsize_psf = 5 # arcsec

#######################
# main part from here #
#######################

import os, sys, glob, copy, inspect
import numpy as np
import pyfits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
plt.ioff()

import analysisUtils as aU
mytb = aU.createCasaTool(tbtool)
mymsmd = aU.createCasaTool(msmdtool)

#############
# functions #
#############
def run_evalclean(preimagename,imsize,imsize_psf,only_html=False):
    """
    """

    ########
    # main #
    ########

    # check clean products
    images = glob.glob(preimagename + ".*")
    image, mask, model, pb, psf, residual = check_clean_outputs(images)

    # plot some quantities useful for evaluating clean quality
    outpng_beam = "report_evalclean/png_files/plot_beam.png"
    outpng_rms = "report_evalclean/png_files/plot_rms.png"
    outpng_hist_image = "report_evalclean/png_files/plot_hist_image.png"
    outpng_hist_residual = "report_evalclean/png_files/plot_hist_residual.png"
    outpng_stats_image = "report_evalclean/png_files/plot_stats_image.png"
    outpng_stats_image2 = "report_evalclean/png_files/plot_stats_image2.png"
    outpng_stats_residual = "report_evalclean/png_files/plot_stats_residual.png"

    if only_html==False:
        # initialize output directory
        os.system("rm -rf ./report_evalclean")
        os.system("mkdir ./report_evalclean")
        os.system("mkdir ./report_evalclean/png_files")

        arr_bmaj, arr_bmins = get_beams(image,outpng_beam)
        data_image, data_residual = get_rms(
            image,
            residual,
            arr_bmaj,
            arr_bmins,
            outpng_rms,
            outpng_hist_image,
            outpng_hist_residual,
            )

        get_stats(
            data_image,
            data_residual,
            outpng_stats_image,
            outpng_stats_image2,
            outpng_stats_residual,
            )

        list_png_image, list_png_contour, list_png_residual, list_png_psf = \
        get_channel_map(image,mask,residual,pb,psf,imsize,imsize_psf)

    else:
        list_png_image = glob.glob("report_evalclean/png_files/image_channel_*.png")
        list_png_contour= glob.glob("report_evalclean/png_files/contour_channel_*.png")
        list_png_residual = glob.glob("report_evalclean/png_files/residual_channel_*.png")
        list_png_psf = glob.glob("report_evalclean/png_files/psf_channel_*.png")

        list_png_image = sorted(list_png_image, key=lambda s: int(re.search(r'\d+', s).group()))
        list_png_contour = sorted(list_png_contour, key=lambda s: int(re.search(r'\d+', s).group()))
        list_png_residual = sorted(list_png_residual, key=lambda s: int(re.search(r'\d+', s).group()))
        list_png_psf = sorted(list_png_psf, key=lambda s: int(re.search(r'\d+', s).group()))

    # generate html report
    header = imhead(image,mode="list")
    del header["perplanebeams"]
    html_report_generator(
        preimagename,
        header,
        outpng_beam,
        outpng_rms,
        outpng_hist_image,
        outpng_hist_residual,
        outpng_stats_image,
        outpng_stats_image2,
        outpng_stats_residual,
        list_png_image,
        list_png_contour,
        list_png_residual,
        list_png_psf,
        )

    # clean up
    os.system("rm -rf imhead.last")
    os.system("rm -rf exportfits.last")
    os.system("rm -rf imsubimage.last")
    os.system("rm -rf imval.last")
    os.system("rm -rf immath.last")

def html_report_generator(
    preimagename,
    header,
    outpng_beam,
    outpng_rms,
    outpng_hist_image,
    outpng_hist_residual,
    outpng_stats_image,
    outpng_stats_image2,
    outpng_stats_residual,
    list_outpng_image,
    list_outpng_contour,
    list_outpng_residual,
    list_outpng_psf,
    ):
    """
    """

    print("\n#########################")
    print("# html_report_generator #")
    print("#########################")

    outpng_beam = outpng_beam.replace("report_evalclean/","")
    outpng_rms = outpng_rms.replace("report_evalclean/","")
    outpng_hist_image = outpng_hist_image.replace("report_evalclean/","")
    outpng_hist_residual = outpng_hist_residual.replace("report_evalclean/","")
    outpng_stats_image = outpng_stats_image.replace("report_evalclean/","")
    outpng_stats_image2 = outpng_stats_image2.replace("report_evalclean/","")
    outpng_stats_residual = outpng_stats_residual.replace("report_evalclean/","")

    section1 = "Image header"
    section2 = "Beam size vs. channel"
    section3 = "Noise rms vs. channel"
    section4 = "Stats spectra"
    section5 = "Channel map"

    outhtml = "report_evalclean/report.html"
    outcss = "report_evalclean/report.css"

    # header
    html = \
        "<!DOCTYPE html>\n" + "<html>\n" + \
        "<head>\n" + \
        "<meta charset=\"utf-8\">\n" + \
        "<link rel=\"stylesheet\" href=\"report.css\">\n" + \
        "</head>\n" + \
        "<body>\n" + \
        "\n" + \
        "<section>\n" + \
        "<h2>My QA report for " + preimagename + " clean outputs</h2>\n" + \
        "\n" + \
        "<ol>\n" + \
        "<li>" + section1 + "</li>\n" + \
        "<li>" + section2 + "</li>\n" + \
        "<li>" + section3 + "</li>\n" + \
        "<li>" + section4 + "</li>\n" + \
        "<li>" + section5 + "</li>\n" + \
        "</ol>\n"

    # header
    html = html + \
        "<section>\n" + \
        "<h3>1. "+section1+"</h3>\n" + \
        "<div onclick=\"obj=document.getElementById(\'menu1\').style; obj.display=(obj.display==\'none\')?\'block\':\'none\';\">" + \
        "<a style=\"cursor:pointer;\">&#8595; .image header</a>" + \
        "</div>" + \
        "<div id=\"menu1\" style=\"display:none;clear:both;\">" + \
        "<table>\n" 

    keys = sorted(header.keys())
    for this_key in keys:
        html = html + \
            "<tr>\n" + \
            "<td>" + str(this_key) + "</td>\n" + \
            "<td>" + str(header[this_key]) + "</td>\n" + \
            "</tr>\n"

    html = html + \
        "</table>" + \
        "</section>\n" + \
        "</div>\n"

    # beam vs channel
    html = html + \
        "<section>\n" + \
        "<h3>2. "+section2+"</h3>\n" + \
        "<figure href=\""+outpng_beam+"\" target=\"_blank\"><img src=\""+outpng_beam+"\" height=\"300\"><figcaption><b>Figure 1.</b> Beam size as a function of channel (red = beam major, blue = beam minor, black = axis ratio).</figcaption></figure>\n" + \
        "</section>\n"

    # rms vs channel
    html = html + \
        "<section>\n" + \
        "<h3>3. "+section3+"</h3>\n" + \
        "<figure href=\""+outpng_rms+"\" target=\"_blank\"><img src=\""+outpng_rms+"\" height=\"300\"><figcaption><b>Figure 2.</b> Noise rms level as a function of channel (red = .image, blue = .residual, black = image/residual ratio).</figcaption></figure>\n" + \
        "<figure href=\""+outpng_hist_image+"\" target=\"_blank\"><img src=\""+outpng_hist_image+"\" height=\"400\"><figcaption><b>Figure 3a.</b> Pixel distribution of .image (red = positive side, blue = negative side * -1). Black line shows the best fit Gaussian. vertical dashed lines show 1sigma and 3sigma width of the Gaussian.</figcaption></figure>\n" + \
        "<figure href=\""+outpng_hist_residual+"\" target=\"_blank\"><img src=\""+outpng_hist_residual+"\" height=\"400\"><figcaption><b>Figure 3b.</b> Pixel distribution of .residual.</figcaption></figure>\n" + \
        "</section>\n"

    # stats
    html = html + \
        "<section>\n" + \
        "<h3>4. "+section4+"</h3>\n" + \
        "<figure href=\""+outpng_stats_image2+"\" target=\"_blank\"><img src=\""+outpng_stats_image2+"\" height=\"400\"><figcaption><b>Figure 4a.</b> Spectra of .image (colorized spectra show max, min, and 16th-50th-84th percentiles). Bold black spectrum shows rms varaition (same as rms spectra shown in Figure 2). Grey spectra show all <i>n</i>th percentiles. </figcaption></figure>\n" + \
        "<figure href=\""+outpng_stats_image+"\" target=\"_blank\"><img src=\""+outpng_stats_image+"\" height=\"400\"><figcaption><b>Figure 4b.</b> Same as Figure 4a, but different y-axis scale. </figcaption></figure>\n" + \
        "<figure href=\""+outpng_stats_residual+"\" target=\"_blank\"><img src=\""+outpng_stats_residual+"\" height=\"400\"><figcaption><b>Figure 4c.</b> Spectra of .residual. Note that if the noise distribution is a perfect Gaussian, rms spectrum (bold black) should be consistent with 84th percentile spectrum (blue). </figcaption></figure>\n" + \
        "</section>\n"


    # channel map .image and .residual
    html = html + \
        "<section>\n" + \
        "<h3>5. "+section5+"</h3>\n" + \
        "<figcaption><b>Figure 5.</b> Channel map of .image, .residual, and .psf. Top panels show .image contours + .image grey scale maps. 1sigma value is calculated at each channel using .residual.Middle-top panels show .image color maps with PB=0.5 contour. Middle-bottom panels show .residual maps with .mask contours. Bottom panels show .psf color maps.</figcaption>\n" + \
        "<div class=\"yoko\">\n"

    for i in range(len(list_outpng_image)):
        this_image = list_outpng_image[i].replace("report_evalclean/","")
        this_contour = list_outpng_contour[i].replace("report_evalclean/","")
        this_residual = list_outpng_residual[i].replace("report_evalclean/","")
        this_psf = list_outpng_psf[i].replace("report_evalclean/","")

        html = html + \
            "<div>\n" + \
            "<a href=\""+this_contour+"\" target=\"_blank\"><img src=\""+this_contour+"\" height=\"400\"></a>\n" + \
            "<a href=\""+this_image+"\" target=\"_blank\"><img src=\""+this_image+"\" height=\"400\"></a>\n" + \
            "<a href=\""+this_residual+"\" target=\"_blank\"><img src=\""+this_residual+"\" height=\"400\"></a>\n" + \
            "<a href=\""+this_psf+"\" target=\"_blank\"><img src=\""+this_psf+"\" height=\"400\"></a>\n" + \
            "</div>\n"

    # footer
    html = html + "</div>\n" + \
         "</section>\n" + \
        "\n" + \
         "</section>\n" + \
        "\n" + \
        "</body>\n" + \
        "</html>"

    os.system("rm -rf " + outhtml)
    f = open(outhtml, "w")
    f.write(html)
    f.close()

    css = \
        ".yoko {\n" + \
        "display:flex;\n" + \
        "overflow-x:scroll;\n" + \
        "text-align:center;\n" + \
        "}\n"

    css = css + \
        "img {\n" + \
        "display: block;\n" + \
        "marigin: 0 auto;\n" + \
        "}\n"

    css = css + \
        "h2 {\n" + \
        "position: relative;\n" + \
        "padding: 0.6em;\n" + \
        "background: #e0edff;\n" + \
        "}\n"

    css = css + \
        "h2:after {\n" + \
        "position: absolute;\n" + \
        "content: '';\n" + \
        "top: 100%;\n" + \
        "left: 30px;\n" + \
        "border: 15px solid transparent;\n" + \
        "border-top: 15px solid #e0edff;\n" + \
        "width: 0;\n" + \
        "height: 0;\n" + \
        "}\n"

    css = css + \
        "h3 {\n" + \
        "position: relative;\n" + \
        "padding: 0.6em;\n" + \
        "background: #e0edff;\n" + \
        "}\n"

    css = css + \
        "h3:after {\n" + \
        "position: absolute;\n" + \
        "content: '';\n" + \
        "top: 100%;\n" + \
        "left: 30px;\n" + \
        "border: 15px solid transparent;\n" + \
        "border-top: 15px solid #e0edff;\n" + \
        "width: 0;\n" + \
        "height: 0;\n" + \
        "}\n"

    os.system("rm -rf " + outcss)
    f = open(outcss, "w")
    f.write(css)
    f.close()

    print("\n##########################################################")
    print("# done! check html report (./report_evalclean/report.html)")
    print("##########################################################")

def check_clean_outputs(imagenames):
    """
    """

    print("\n###########################")
    print("# run check_clean_outputs #")
    print("###########################")

    present = []
    present.append(any(s.endswith(".image") for s in imagenames))
    present.append(any(s.endswith(".mask") for s in imagenames))
    present.append(any(s.endswith(".model") for s in imagenames))
    present.append(any(s.endswith(".pb") for s in imagenames))
    present.append(any(s.endswith(".psf") for s in imagenames))
    present.append(any(s.endswith(".residual") for s in imagenames))

    if not all(present):
        print("# one or more clean outputs are missing!")
        return None, None, None, None, None, None

    else:
        print("# found all clean output! Proceed.")
        image    = [s for s in imagenames if s.endswith(".image")][0] 
        mask     = [s for s in imagenames if s.endswith(".mask")][0] 
        model    = [s for s in imagenames if s.endswith(".model")][0] 
        pb       = [s for s in imagenames if s.endswith(".pb")][0] 
        psf      = [s for s in imagenames if s.endswith(".psf")][0] 
        residual = [s for s in imagenames if s.endswith(".residual")][0] 
        return image, mask, model, pb, psf, residual

def plot_hist(data,output,snr_for_fit=1.0):
    """
    plot voxel distribution in the input data. Zero will be ignored.
    output units will be mK.
    """

    # prepare
    data = data.flatten()
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data = data[data!=0] * 1000

    if output.endswith("image.png"):
        title = ".image histogram"
    elif output.endswith("residual.png"):
        title = ".residual histogram"
    else:
        title = "None"

    # get voxel histogram
    bins   = int( np.ceil( np.log2(len(data))+1 ) * 10 )
    p84    = np.percentile(data, 16) * -1
    hrange = [-5*p84, 5*p84]
    hist   = np.histogram(data, bins=bins, range=hrange)

    x,y   = hist[1][:-1], hist[0]/float(np.sum(hist[0])) * 1000
    histx = x[x<p84*snr_for_fit]
    histy = y[x<p84*snr_for_fit]

    # fit
    x_bestfit = np.linspace(hrange[0], hrange[1], bins)
    popt,_ = curve_fit(func1, histx, histy, p0=[np.max(histy),p84], maxfev=10000)

    # best fit
    peak      = popt[0]
    rms_mK    = np.round(popt[1],3)
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
    tsnr_a = "1$\sigma$ = \n"+str(np.round(rms_mK*snr_a,3))+" mJy"
    tsnr_b = "3$\sigma$ = \n"+str(np.round(rms_mK*snr_b,3))+" mJy"

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
    ax.text(rms_mK*snr_a,ymax*0.8,tsnr_a,rotation=90)
    ax.text(rms_mK*snr_b,ymax*0.5,tsnr_b,rotation=90)

    ax.set_title(title)
    ax.set_xlabel("Pixel value (mJy)")
    ax.set_ylabel("Pixel count density * 10$^{3}$")
    ax.set_xlim([0,5*p84])
    ax.set_ylim([0,ymax])

    plt.savefig(output, dpi=100)

def func1(x, a, c):
    return a*np.exp(-(x)**2/(2*c**2))

def get_beams(imagename,outpng):
    """
    """

    print("\n#############")
    print("# get_beams #")
    print("#############")

    if imagename==None:
        print("# one or more clean outputs are missing!")
        return None

    header = imhead(imagename,mode="list")

    if "perplanebeams" in header:
        bminfo = header["perplanebeams"]
        nchan = bminfo["nChannels"]

        list_bmaj,list_bmin = [], []
        for this_chan in range(nchan):
            list_bmaj.append(bminfo["*"+str(this_chan)]["major"]["value"])
            list_bmin.append(bminfo["*"+str(this_chan)]["minor"]["value"])

    if "beammajor" in header:
        index = int([k for k, v in header.items() if v=="Frequency"][0].split("ctype")[1])-1
        nchan = header["shape"][index]

        list_bmaj = [header["beammajor"]["value"]] * nchan
        list_bmin = [header["beamminor"]["value"]] * nchan

    list_bmaj,list_bmin = np.array(list_bmaj),np.array(list_bmin)

    # plot
    x = np.array(range(nchan))

    fig = plt.figure(figsize=(20,6))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.subplots_adjust(bottom=0.15, left=0.07, right=0.93, top=0.88)

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    axb = ax.twinx()
    ax.grid(axis="both", ls="--")

    ax.step(x, list_bmaj, color="red", lw=2, alpha=1.0, where="mid", label="bmaj")
    ax.step(x, list_bmin, color="blue", lw=2, alpha=1.0, where="mid", label="bmin")

    axb.step(x, list_bmaj/list_bmin, color="black", lw=3, alpha=1.0, where="mid", label="ratio")

    if np.max(list_bmaj/list_bmin)<0.1:
    	med = np.median(list_bmaj/list_bmin)
    	ylim = [med-0.2,med+0.2]
    	step = abs(np.max(np.max(list_bmaj/list_bmin)) - np.min(np.max(list_bmaj/list_bmin))) * 0.1
    	ylim = [np.min(np.max(list_bmaj/list_bmin))-step,np.max(np.max(list_bmaj/list_bmin))+step]
    else:
    	ylim = None

    ax.set_xlim([np.min(x),np.max(x)])
    ax.set_ylim(ylim)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Beam Size (arcsec)")
    axb.set_ylabel("Beam Axis Ratio")
    ax.set_title(imagename.split("/")[-1])
    ax.legend(loc="upper left")
    axb.legend(loc="lower left")

    plt.savefig(outpng, dpi=100)

    return np.array(list_bmaj), np.array(list_bmin)

def get_rms(imagename,residual,arr_bmaj,arr_bmins,outpng_rms,outpng_hist_image,outpng_hist_residual):
    """
    """

    print("\n###########")
    print("# get_rms #")
    print("###########")

    if imagename==None or residual==None:
        print("# one or more clean outputs are missing!")
        return None

    # get data
    data_image = imval_data(imagename)
    data_residual = imval_data(residual)

    plot_hist(data_image,outpng_hist_image)
    plot_hist(data_residual,outpng_hist_residual)

    # get rms spectra
    rms_image = np.where(data_image!=0,data_image,np.nan)
    rms_residual = np.where(data_residual!=0,data_residual,np.nan)

    rms_image = np.where(~np.isinf(rms_image),rms_image,np.nan)
    rms_residual = np.where(~np.isinf(rms_residual),rms_residual,np.nan)

    rms_image = np.sqrt(np.nanmean(np.square(rms_image),axis=(0,1)))
    rms_residual = np.sqrt(np.nanmean(np.square(rms_residual),axis=(0,1)))

    # plot
    x = np.array(range(len(rms_image)))

    fig = plt.figure(figsize=(20,6))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.subplots_adjust(bottom=0.15, left=0.07, right=0.93, top=0.88)

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    axb = ax.twinx()
    ax.grid(axis="both", ls="--")

    ax.step(x, rms_image*1000, color="red", lw=2, alpha=1.0, where="mid", label="rms (.image)")
    ax.step(x, rms_residual*1000, color="blue", lw=2, alpha=1.0, where="mid", label="rms (.residual)")

    axb.step(x, rms_image/rms_residual, color="black", lw=3, alpha=1.0, where="mid", label="ratio")

    ax.set_xlim([np.min(x),np.max(x)])

    ax.set_xlabel("Channel")
    ax.set_ylabel("rms (mJy beam$^{-1}$)")
    axb.set_ylabel("rms ratio")
    ax.set_title(imagename.split("/")[-1])
    ax.legend(loc="upper left")
    axb.legend(loc="lower left")

    plt.savefig(outpng_rms, dpi=100)

    return data_image, data_residual

def imval_data(imagename):

    shape = imhead(imagename,mode="list")["shape"]
    box = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)

    print("# imval data (may take time)")
    data = imval(imagename,box=box)
    data = data["data"] * data["mask"]
    data[np.isinf(data)] = 0
    data = np.nan_to_num(data)

    return data

def myax_set(
    ax,
    grid=None,
    xlim=None,
    ylim=None,
    title=None,
    xlabel=None,
    ylabel=None,
    aspect=None,
    adjust=[0.10,0.99,0.10,0.95],
    lw_grid=1.0,
    lw_ticks=2.5,
    lw_outline=2.5,
    fsize=22,
    fsize_legend=20,
    labelbottom=True,
    labeltop=False,
    labelleft=True,
    labelright=False,
    ):

    # adjust edge space
    plt.subplots_adjust(
        left=adjust[0],
        right=adjust[1],
        bottom=adjust[2],
        top=adjust[3],
        )

    # font
    plt.rcParams["font.size"] = fsize
    plt.rcParams["legend.fontsize"] = fsize_legend

    # tick width
    ax.xaxis.set_tick_params(width=lw_ticks)
    ax.yaxis.set_tick_params(width=lw_ticks)

    # labels
    ax.tick_params(
        labelbottom=labelbottom,
        labeltop=labeltop,
        labelleft=labelleft,
        labelright=labelright,
        )

    # outline width
    axis = ["top", "bottom", "left", "right"]
    lw_outlines = [lw_outline, lw_outline, lw_outline, lw_outline]
    for a,w in zip(axis, lw_outlines):
        ax.spines[a].set_linewidth(w)

    # aspect
    if aspect is not None:
        ax.set_aspect(aspect)

    # grid
    if grid is not None:
        ax.grid(axis=grid, lw=lw_grid)

    # xylims
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # xylabels
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # title
    if title is not None:
        ax.set_title(title, x=0.5, y=1.02, weight="bold")

def myfig_fits2png(
    # general
    imcolor,
    outfile,
    imcontour1=None,
    imcontour2=None,
    imsize_as=50,
    ra_cnt=None,
    dec_cnt=None,
    # contour 1
    unit_cont1=None,
    levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
    width_cont1=[2.0],
    color_cont1="black",
    # contour 2
    unit_cont2=None,
    levels_cont2=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
    width_cont2=[1.0],
    color_cont2="black",
    # imshow
    fig_dpi=25,
    set_grid="both",
    set_title=None,
    colorlog=False,
    set_bg_color="white",
    set_cmap="rainbow",
    showzero=False,
    showbeam=True,
    color_beam="black",
    scalebar=None,
    label_scalebar=None,
    color_scalebar="black",
    comment=None,
    comment_color="black",
    # imshow colorbar
    set_cbar=True,
    clim=None,
    label_cbar=None,
    lw_cbar=1.0,
    # annotation
    numann=None,
    textann=True,
    ):
    """
    Parameters
    ----------
    unit_cont1 : float (int) or None
        if None, unit_cont1 is set to the image peak value.

    levels_cont1 : float list
        output contour levels = unit_cont1 * levels_cont1.
        default is 1% to 96% of the image peak value.

    ra_cnt and dec_cnt : str
        user-defined image center in degree units.
    """

    print("# run fits2png")
    print("# imcolor = " + imcolor)

    #################
    ### preparation #
    #################

    # make sure fits format
    if imcolor[-5:]!=".fits":
        run_exportfits(
            imcolor,
            imcolor + ".fits",
            dropstokes = True,
            dropdeg    = True,
            )
        imcolor += ".fits"

    if imcontour1!=None:
        if imcontour1[-5:]!=".fits":
            run_exportfits(
                imcontour1,
                imcontour1 + ".fits",
                dropstokes = True,
                dropdeg    = True,
                )
            imcontour1 += ".fits"

    if imcontour2!=None:
        if imcontour2[-5:]!=".fits":
            run_exportfits(
                imcontour2,
                imcontour2 + ".fits",
                dropstokes = True,
                dropdeg    = True,
                )
            imcontour2 += ".fits"

    # read imcolor
    hdu        = pyfits.open(imcolor)
    image_data = hdu[0].data[:,:]
    image_data = np.where(~np.isnan(image_data), image_data, 0)
    datamax    = np.max(image_data)

    pix_ra_as  = hdu[0].header["CDELT1"] * 3600
    ra_im_deg  = hdu[0].header["CRVAL1"]
    ra_im_pix  = hdu[0].header["CRPIX1"]
    ra_size    = hdu[0].header["NAXIS1"]

    pix_dec_as = hdu[0].header["CDELT2"] * 3600
    dec_im_deg = hdu[0].header["CRVAL2"]
    dec_im_pix = hdu[0].header["CRPIX2"]
    dec_size   = hdu[0].header["NAXIS2"]

    # get centers if None
    if ra_cnt==None:
        ra_cnt = str(ra_im_deg) + "deg"

    if dec_cnt==None:
        dec_cnt = str(dec_im_deg) + "deg"

    # determine contour levels
    if imcontour1!=None:
        contour_data1, levels_cont1 = \
            _get_contour_levels(imcontour1,unit_cont1,levels_cont1)

    if imcontour2!=None:
        contour_data2, levels_cont2 = \
            _get_contour_levels(imcontour2,unit_cont2,levels_cont2)

    # define imaging extent
    extent = _get_extent(ra_cnt,dec_cnt,ra_im_deg,dec_im_deg,
        ra_im_pix,dec_im_pix,pix_ra_as,pix_dec_as,ra_size,dec_size)

    # set lim
    xlim = [imsize_as/2.0, -imsize_as/2.0]
    if float(dec_cnt.replace("deg",""))>0:
        ylim = [imsize_as/2.0, -imsize_as/2.0]
    else:
        ylim = [-imsize_as/2.0, imsize_as/2.0]

    # set colorlog
    if colorlog==True:
        norm = LogNorm(vmin=0.02*datamax, vmax=datamax)
    else:
        norm = None

    # set None pixels
    image_data = np.where(~np.isinf(image_data),image_data,0)
    image_data = np.where(~np.isnan(image_data),image_data,0)
    if showzero==False:
        image_data[np.where(image_data==0)] = None

    ##########
    # imshow #
    ##########

    # plot
    plt.figure(figsize=(13,10))
    gs = gridspec.GridSpec(nrows=10, ncols=10)
    ax = plt.subplot(gs[0:10,0:10])
    ax.set_aspect('equal')

    # set ax parameter
    xl, yl = "R.A. Offset (arcsec)", "Decl. Offset (arcsec)"
    ad = [0.10,0.90,0.10,0.85]
    myax_set(ax,grid=set_grid,xlim=xlim,ylim=ylim,title=set_title,
        xlabel=xl,ylabel=yl,adjust=ad)

    cim = ax.imshow(image_data,cmap=set_cmap,norm=norm,
        extent=extent,interpolation="none")

    if imcontour1!=None:
        ax.contour(contour_data1,levels=levels_cont1,extent=extent,
            colors=color_cont1,linewidths=width_cont1,origin="upper")

    if imcontour2!=None:
        ax.contour(contour_data2,levels=levels_cont2,extent=extent,
            colors=color_cont2,linewidths=width_cont2,origin="upper")

    if set_bg_color!=None:
        ax.axvspan(xlim[0],xlim[1],ylim[0],ylim[1],color=set_bg_color,zorder=0)

    # colorbar
    cim.set_clim(clim)
    if set_cbar==True:
        _myax_cbar(plt,ax,cim,label=label_cbar,clim=clim)

    # add beam size
    if showbeam==True:
        _myax_showbeam(ax,imcolor,ra_cnt,xlim,ylim,color_beam)

    # add scalebar
    if scalebar!=None:
        _myax_scalebar(ax,ra_cnt,xlim,ylim,label_scalebar,scalebar,color_scalebar)

    # add comment
    if comment!=None:
        _myax_comment(ax,dec_cnt,xlim,ylim,comment_color)

    # annotation
    if numann!=None:
        myax_fig2png_ann(ax,numann,textann)

    # save
    plt.savefig(outfile, dpi=fig_dpi)

def run_exportfits(
    imagename,
    fitsimage,
    dropdeg=False,
    dropstokes=False,
    delin=False,
    ):
    """
    """

    # run exportfits
    os.system("rm -rf " + fitsimage)
    exportfits(
        fitsimage  = fitsimage,
        imagename  = imagename,
        dropdeg    = dropdeg,
        dropstokes = dropstokes,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

def get_channel_map(imagename,mask,residual,pb,psf,imsize,imsize_psf):

    print("\n###################")
    print("# get_channel_map #")
    print("###################")

    # get nchan
    header = imhead(imagename,mode="list")
    index = int([k for k, v in header.items() if v=="Frequency"][0].split("ctype")[1])-1
    nchan = header["shape"][index]
    immax,immin = header["datamax"]*1000,header["datamin"]*1000

    header = imhead(residual,mode="list")
    resmax,resmin = header["datamax"]*1000,header["datamin"]*1000

    # plot
    list_outpng_image = []
    list_outpng_contour = []
    list_outpng_residual = []
    list_outpng_psf = []
    for this_chan in range(nchan):
        outpng_image = "report_evalclean/png_files/image_channel_"+str(this_chan)+".png"
        outpng_contour= "report_evalclean/png_files/contour_channel_"+str(this_chan)+".png"
        outpng_residual = "report_evalclean/png_files/residual_channel_"+str(this_chan)+".png"
        outpng_psf = "report_evalclean/png_files/psf_channel_"+str(this_chan)+".png"
        print("# generate " + outpng_image)
        print("# generate " + outpng_residual)

        os.system("rm -rf " + imagename + "_this_chan")
        immath(
            imagename=imagename,
            expr="IM0*1000",
            outfile=imagename+"_this_chan",
            chans=str(this_chan),
            )

        os.system("rm -rf " + imagename + "_this_chan_minus")
        immath(
            imagename=imagename,
            expr="IM0*1000*-1",
            outfile=imagename+"_this_chan_minus",
            chans=str(this_chan),
            )

        os.system("rm -rf " + mask + "_this_chan")
        imsubimage(
            imagename=mask,
            outfile=mask+"_this_chan",
            chans=str(this_chan),
            dropdeg=True,
            )

        os.system("rm -rf " + pb + "_this_chan")
        imsubimage(
            imagename=pb,
            outfile=pb+"_this_chan",
            chans=str(this_chan),
            dropdeg=True,
            )

        os.system("rm -rf " + psf + "_this_chan")
        imsubimage(
            imagename=psf,
            outfile=psf+"_this_chan",
            chans=str(this_chan),
            dropdeg=True,
            )

        os.system("rm -rf " + residual + "_this_chan")
        immath(
            imagename=residual,
            expr="IM0*1000",
            outfile=residual+"_this_chan",
            chans=str(this_chan),
            )

        # get rms of this_chan
        this_residual = imval_data(residual+"_this_chan")
        rms_image = np.where(this_residual!=0,this_residual,np.nan)
        rms_image = np.sqrt( np.nanmean(np.square(rms_image)) )

        # plot
        os.system("rm -rf " + outpng_image)
        os.system("rm -rf " + outpng_residual)

        # .residual + pb=0.5 + mask
        myfig_fits2png(
            imcolor=residual+"_this_chan",
            outfile=outpng_residual,
            imcontour1=mask+"_this_chan",
            imcontour2=pb+"_this_chan",
            imsize_as=imsize,
            levels_cont1=[0.5],
            levels_cont2=[0.5],
            set_grid="both",
            set_title=".residual at channel = "+str(this_chan),
            colorlog=False,
            showbeam=False,
            clim=[resmin,resmax],
            label_cbar="(mJy beam$^{-1}$)",
            )

        # .image + .image s/n=3xn contours
        noise_text = "($\pm$3n$\sigma$, 1$\sigma$="+str(np.round(rms_image,3))+" mJy beam$^{-1}$)"
        myfig_fits2png(
            imcolor=imagename+"_this_chan",
            outfile=outpng_contour,
            imcontour1=imagename+"_this_chan",
            imcontour2=imagename+"_this_chan_minus",
            set_cmap="Greys",
            imsize_as=imsize,
            unit_cont1=rms_image,
            levels_cont1=[3,6,9,12,15,18,21,24,27,30],
            color_cont1="red",
            unit_cont2=rms_image,
            levels_cont2=[3,6,9,12,15,18,21,24,27,30],
            color_cont2="blue",
            set_grid="both",
            set_title=".image at channel = "+str(this_chan) + "\n" + noise_text,
            colorlog=False,
            showbeam=True,
            clim=[immin,immax],
            label_cbar="(mJy beam$^{-1}$)",
            )

        # .image + pb=0.5 + mask
        myfig_fits2png(
            imcolor=imagename+"_this_chan",
            outfile=outpng_image,
            imcontour1=mask+"_this_chan",
            imcontour2=pb+"_this_chan",
            imsize_as=imsize,
            levels_cont1=[0.5],
            levels_cont2=[0.5],
            set_grid="both",
            set_title=".image at channel = "+str(this_chan),
            colorlog=False,
            showbeam=True,
            clim=[immin,immax],
            label_cbar="(mJy beam$^{-1}$)",
            )

        # .psf
        myfig_fits2png(
            imcolor=psf+"_this_chan",
            outfile=outpng_psf,
            imsize_as=imsize_psf,
            unit_cont1=1.0,
            levels_cont1=[0.1,0.3,0.5,0.7,0.9],
            color_cont1="black",
            clim=[0,1],
            set_grid="both",
            set_title=".psf at channel = "+str(this_chan),
            colorlog=False,
            )

        list_outpng_image.append(outpng_image)
        list_outpng_contour.append(outpng_contour)
        list_outpng_residual.append(outpng_residual)
        list_outpng_psf.append(outpng_psf)

        os.system("rm -rf " + imagename + "_this_chan*")
        os.system("rm -rf " + residual + "_this_chan*")
        os.system("rm -rf " + mask + "_this_chan*")
        os.system("rm -rf " + pb + "_this_chan*")
        os.system("rm -rf " + psf + "_this_chan*")

    return list_outpng_image, list_outpng_contour, list_outpng_residual, list_outpng_psf

def get_stats(data_image,data_residual,outpng_stats_image,outpng_stats_image2,outpng_stats_residual):
    """
    plot stats spectra. Zero will be ignored.
    """

    print("\n#############")
    print("# get_stats #")
    print("#############")

    # prepare
    data_image = np.where(data_image!=0,data_image,np.nan)
    data_image = np.where(~np.isinf(data_image),data_image,np.nan)

    data_residual = np.where(data_residual!=0,data_residual,np.nan)
    data_residual = np.where(~np.isinf(data_residual),data_residual,np.nan)

    # get rms spectra
    rms_image = np.sqrt(np.nanmean(np.square(data_image),axis=(0,1)))
    rms_residual = np.sqrt(np.nanmean(np.square(data_residual),axis=(0,1)))

    # get max spectra
    max_image = np.nanmax(data_image,axis=(0,1))
    max_residual = np.nanmax(data_residual,axis=(0,1))

    # get min spectra
    min_image = np.nanmin(data_image,axis=(0,1))
    min_residual = np.nanmin(data_residual,axis=(0,1))

    # get p16 spectra
    p16_image = np.nanpercentile(data_image,16,axis=(0,1))
    p16_residual = np.nanpercentile(data_residual,16,axis=(0,1))

    # get p50 spectra
    p50_image = np.nanpercentile(data_image,50,axis=(0,1))
    p50_residual = np.nanpercentile(data_residual,50,axis=(0,1))

    # get p84 spectra
    p84_image = np.nanpercentile(data_image,84,axis=(0,1))
    p84_residual = np.nanpercentile(data_residual,84,axis=(0,1))

    # x
    x_image = np.array(range(len(rms_image)))
    x_residual = np.array(range(len(rms_residual)))

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

    ylim = [1.1*np.min(min_residual),1.1*np.max(max_residual)]

    # plot image no ylim
    fig = plt.figure(figsize=(20,6))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.subplots_adjust(bottom=0.15, left=0.07, right=0.93, top=0.88)

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax1 = plt.subplot(gs[0:30,0:30])
    ax1.grid(axis="both", ls="--")
    ax1b = ax1.twinx()

    ax1b.step(x_image, rms_image, color="black", lw=4, alpha=1.0, where="mid")
    ax1.step(x_image, max_image, color=cmax, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, p84_image, color=cp84, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, p50_image, color=cp50, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, p16_image, color=cp16, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, min_image, color=cmin, lw=2, alpha=1.0, where="mid")

    for i in range(100):
        this_p = np.nanpercentile(data_image,i,axis=(0,1))
        ax1.plot(x_image, this_p, color="grey", lw=1.0, alpha=0.3)


    tf = ax1.transAxes
    ax1.text(0.03,0.85,tmax,color=cmax,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.75,tp84,color=cp84,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.65,tp50,color=cp50,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.55,tp16,color=cp16,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.45,tmin,color=cmin,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.35,trms,color=crms,transform=tf,horizontalalignment=ha,weight=w)

    ax1.set_ylabel("stats value (Jy beam$^{-1}$)")
    ax1b.set_ylabel("rms (Jy beam$^{-1}$)")
    ax1.set_title(".image stats")

    ax1.set_xlim([np.min(x_image),np.max(x_image)])
    ax1b.set_xlim([np.min(x_image),np.max(x_image)])

    plt.savefig(outpng_stats_image2, dpi=100)

    # plot image
    fig = plt.figure(figsize=(20,6))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.subplots_adjust(bottom=0.15, left=0.07, right=0.93, top=0.88)

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax1 = plt.subplot(gs[0:30,0:30])
    ax1.grid(axis="both", ls="--")
    ax1b = ax1.twinx()

    ax1b.step(x_image, rms_image, color="black", lw=4, alpha=1.0, where="mid")
    ax1.step(x_image, max_image, color=cmax, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, p84_image, color=cp84, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, p50_image, color=cp50, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, p16_image, color=cp16, lw=2, alpha=1.0, where="mid")
    ax1.step(x_image, min_image, color=cmin, lw=2, alpha=1.0, where="mid")

    for i in range(100):
        this_p = np.nanpercentile(data_image,i,axis=(0,1))
        ax1.plot(x_image, this_p, color="grey", lw=1.0, alpha=0.3)

    tf = ax1.transAxes
    ax1.text(0.03,0.85,tmax,color=cmax,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.75,tp84,color=cp84,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.65,tp50,color=cp50,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.55,tp16,color=cp16,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.45,tmin,color=cmin,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.35,trms,color=crms,transform=tf,horizontalalignment=ha,weight=w)

    ax1.set_ylabel("stats value (Jy beam$^{-1}$)")
    ax1b.set_ylabel("rms (Jy beam$^{-1}$)")
    ax1.set_title(".image stats")

    ax1.set_xlim([np.min(x_image),np.max(x_image)])
    ax1b.set_xlim([np.min(x_image),np.max(x_image)])
    ax1.set_ylim(ylim)
    ax1b.set_ylim(ylim)

    plt.savefig(outpng_stats_image, dpi=100)

    # plot residual
    fig = plt.figure(figsize=(20,6))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.subplots_adjust(bottom=0.15, left=0.07, right=0.93, top=0.88)

    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax1 = plt.subplot(gs[0:30,0:30])
    ax1.grid(axis="both", ls="--")
    ax1b = ax1.twinx()

    ax1b.step(x_residual, rms_residual, color="black", lw=4, alpha=1.0, where="mid")
    ax1.step(x_residual, max_residual, color=cmax, lw=2, alpha=1.0, where="mid")
    ax1.step(x_residual, p84_residual, color=cp84, lw=2, alpha=1.0, where="mid")
    ax1.step(x_residual, p50_residual, color=cp50, lw=2, alpha=1.0, where="mid")
    ax1.step(x_residual, p16_residual, color=cp16, lw=2, alpha=1.0, where="mid")
    ax1.step(x_residual, min_residual, color=cmin, lw=2, alpha=1.0, where="mid")

    for i in range(100):
        this_p = np.nanpercentile(data_residual,i,axis=(0,1))
        ax1.plot(x_residual, this_p, color="grey", lw=1.0, alpha=0.3)

    tf = ax1.transAxes
    ax1.text(0.03,0.85,tmax,color=cmax,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.75,tp84,color=cp84,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.65,tp50,color=cp50,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.55,tp16,color=cp16,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.45,tmin,color=cmin,transform=tf,horizontalalignment=ha,weight=w)
    ax1.text(0.03,0.35,trms,color=crms,transform=tf,horizontalalignment=ha,weight=w)

    ax1.set_ylabel("stats value (Jy beam$^{-1}$)")
    ax1b.set_ylabel("rms (Jy beam$^{-1}$)")
    ax1.set_title(".residual stats")

    ax1.set_xlim([np.min(x_residual),np.max(x_residual)])
    ax1b.set_xlim([np.min(x_residual),np.max(x_residual)])
    ax1.set_ylim(ylim)
    ax1b.set_ylim(ylim)

    plt.savefig(outpng_stats_residual, dpi=100)

def _myax_scalebar(ax,ra_cnt,xlim,ylim,label_scalebar,scalebar,color_scalebar):

    if float(ra_cnt.replace("deg",""))>0:
        ax.text(min(xlim)*0.8, max(ylim)*-0.9,
            label_scalebar, horizontalalignment="right")

        e2 = patches.Rectangle(xy = ( min(xlim)*0.8, max(ylim)*-0.8 ),
            width=scalebar, height=0.1, linewidth=4, edgecolor=color_scalebar)

    else:
        ax.text(min(xlim)*0.8, max(ylim)*-0.9,
            label_scalebar, horizontalalignment="right")

        e2 = patches.Rectangle(xy = ( min(xlim)*0.8, max(ylim)*-0.8 ),
            width=scalebar, height=0.1, linewidth=4, edgecolor=color_scalebar)

    ax.add_patch(e2)

def _myax_showbeam(ax,fitsimage,ra_cnt,xlim,ylim,color_beam):

    header = imhead(fitsimage,mode="list")

    if "beammajor" in header.keys():
        bmaj = header["beammajor"]["value"]
        bmin = header["beamminor"]["value"]
        bpa  = header["beampa"]["value"]
    else:
        bmaj,bmin,bpa = 0.01,0.01,0.0

    if float(ra_cnt.replace("deg",""))>0:
        ax.text(min(xlim)*-0.8, max(ylim)*-0.9,
            "beam", horizontalalignment="left", color=color_beam)

        e1 = patches.Ellipse(xy = ( -min(xlim)*0.8-bmin/2.0, -max(ylim)*0.8 ),
            width=bmin, height=bmaj, angle=-bpa, fc=color_beam)

    else:
        ax.text(min(xlim)*-0.8, max(ylim)*-0.9,
            "beam", horizontalalignment="left", color=color_beam)

        e1 = patches.Ellipse(xy = ( -min(xlim)*0.8-bmin/2.0, -max(ylim)*0.8 ),
            width=bmin, height=bmaj, angle=-bpa, fc=color_beam)

    ax.add_patch(e1)

def _get_extent(
    ra_cnt,
    dec_cnt,
    ra_im_deg,
    dec_im_deg,
    ra_im_pix,
    dec_im_pix,
    pix_ra_as,
    pix_dec_as,
    ra_size,
    dec_size,
    ):

    if ra_cnt!=None:
        offset_ra_deg = 0
    else:
        offset_ra_deg = ra_im_deg - float(ra_cnt.replace("deg",""))

    offset_ra_pix = offset_ra_deg * 3600 / float(pix_ra_as)
    this_ra = ra_im_pix - offset_ra_pix

    if dec_cnt!=None:
        offset_dec_deg = 0
    else:
        offset_dec_deg = dec_im_deg - float(dec_cnt.replace("deg",""))

    offset_dec_pix = offset_dec_deg * 3600 / float(pix_dec_as)
    this_dec = dec_im_pix - offset_dec_pix

    xext_min = float(pix_ra_as) * (0.5 - this_ra)
    xext_max = float(pix_ra_as) * (0.5 - this_ra + ra_size)

    if float(dec_cnt.replace("deg",""))>0:
        yext_min = float(pix_dec_as) * (0.5 - this_dec)
        yext_max = float(pix_dec_as) * (0.5 - this_dec + dec_size)
    else:
        yext_min = float(pix_dec_as) * (0.5 - this_dec)
        yext_max = float(pix_dec_as) * (0.5 - this_dec + dec_size)

    extent = [xext_min, xext_max, yext_max, yext_min]

    return extent

def _get_contour_levels(fitsimage,unit_contour,levels_contour):

    hdu = pyfits.open(fitsimage)
    contour_data = hdu[0].data[:,:]

    if unit_contour==None:
        header = imhead(fitsimage, "list")
        if "datamax" in header.keys():
            unit_contour = imhead(fitsimage, "list")["datamax"]
        else:
            unit_contour = 1.0

    output_contours = list(map(lambda x: x * unit_contour, levels_contour))

    return contour_data, output_contours

def _myax_cbar(
    fig,
    ax,
    data,
    label=None,
    clim=None,
    ):
    cb = fig.colorbar(data, ax=ax)

    if label is not None:
        cb.set_label(label)

    if clim is not None:
        cb.set_clim(clim)

    cb.outline.set_linewidth(2.5)

run_evalclean(preimagename,imsize,imsize_psf)
