"""
Standalone routines that manipulate uv data. Run with CASA5.

history:
2021-08-16   created by TS
2021-08-17   small bug fix
2021-08-18   small bug fix
2021-08-20   add drop_edge parameter to run_concat
Toshiki Saito@Nichidai/NAOJ
"""

### output (3 files for each field name)
# 12m array ms    : target_name + "_12m_" + linename + ".ms"
# 7m array ms     : target_name + "_7m_" + linename + ".ms"
# 12m+7m array ms : target_name + "_12m+7m_" + linename + ".ms"

### input
mskey_12m = "/wkm16/saitots/n1068_b8/ms_12m/uid___A002_X*.ms"
mskey_7m  = "/wkm16/saitots/n1068_b8/ms_7m/uid___A002_X*.ms.split.cal"
#mskey_12m = "/wkm16/stakano/saigo-tar/2017.1.00586.S/science_goal.uid___A001_X1284_X20cd/group.uid___A001_X1284_X20ce/member.uid___A001_X1284_X20cf/calibrated/uid___A002_X*.ms" # or None
#mskey_7m  = "/wkm16/stakano/stakano/ACA-20190324/2017.1.00586.S/science_goal.uid___A001_X1284_X20cd/group.uid___A001_X1284_X20ce/member.uid___A001_X1284_X20d1/calibrated/uid___A002_X*.ms.split.cal" # or None

delete_intermediate = True
# False saves time of the next run, True saves the disk space.

drop_edge = 5

z = 0.00379
linename = "ci10"
restfreq = 492.16065100 # GHz
linewidth = 620 #kms

#######################
# main part from here #
#######################

import os, sys, glob, copy, inspect
import numpy as np

import analysisUtils as aU
mytb = aU.createCasaTool(tbtool)
mymsmd = aU.createCasaTool(msmdtool)

#############
# functions #
#############
obsfreq = restfreq / (1+z)

def run_split(list_vis,linename,obsfreq):
    """
    """

    print("\n#############")
    print("# run_split #")
    print("#############")

    list_outputvis = []
    for this_vis in list_vis:
        # check columns
        mytb.open(this_vis,nomodify=True)
        colnames = mytb.colnames()
        mytb.close()

        if "CORRECTED_DATA" in colnames:
            print("# CORRECTED column exists in " + this_vis.split("/")[-1] + ".")
            datacolumn = "CORRECTED"
        else:
            print("# CORRECTED column does not exists in " + this_vis.split("/")[-1] + ".")
            datacolumn = "DATA"

        # get science field names and their spws
        mymsmd.open(this_vis)

        fieldnames = mymsmd.fieldsforintent("OBSERVE_TARGET#ON_SOURCE",True)

        this_spws = mymsmd.spwsforintent("OBSERVE_TARGET#ON_SOURCE")
        alma_spws = mymsmd.almaspws(chavg=True,wvr=True,complement=True)
        science_spws = list(set(this_spws) & set(alma_spws))

        line_spws = []
        for this_sci_spw in science_spws:
            chanfreqs = mymsmd.chanfreqs(this_sci_spw) / 1e9
            minfreq,maxfreq = np.min(chanfreqs),np.max(chanfreqs)

            if obsfreq>=minfreq and obsfreq<=maxfreq:
                line_spws.append(str(this_sci_spw))

        mymsmd.done()
        line_spws = ",".join(line_spws)

        for this_field in fieldnames:
            outputvis = this_vis.split("/")[-1] + "." + this_field + "." + linename

            if not glob.glob(outputvis):
                print("# splitting field=" + this_field + ", spw=" + line_spws + \
                    " from " + this_vis.split("/")[-1] + ".")
                mstransform(
                    vis = this_vis,
                    field = this_field,
                    spw = line_spws,
                    outputvis  = outputvis,
                    datacolumn = datacolumn,
                    keepflags  = False,
                    outframe = "LSRK",
                    veltype = "radio",
                    )
            else:
                print("# skip first mstransform because output exists!")
                print("  output = " + outputvis)

            if glob.glob(outputvis):
                list_outputvis.append(outputvis)

    return list_outputvis, fieldnames

def run_uvcontsub(list_vis,linename,obsfreq,linewidth,delete_intermediate=False):
    """
    """

    print("\n#################")
    print("# run_uvcontsub #")
    print("#################")

    list_outputvis = []
    for this_vis in list_vis:
        # spw info
        mymsmd.open(this_vis)
        spws = mymsmd.almaspws(chavg=True,wvr=True,complement=True)

        fitspw = []
        for this_spw in spws:
            meanfreq = mymsmd.meanfreq(this_spw) / 1e9
            chanfreqs = mymsmd.chanfreqs(this_spw) / 1e9
            minfreq,maxfreq = np.min(chanfreqs),np.max(chanfreqs)

            linewidthfreq = linewidth / 299792.0 * obsfreq # linewidth in GHz
            lineminfreq = obsfreq - linewidthfreq / 2.0
            linemaxfreq = obsfreq + linewidthfreq / 2.0

            if lineminfreq<minfreq:
                this_fitspw = str(this_spw) + ":" + str(linemaxfreq) + "~" + str(maxfreq) + "GHz"
            elif linemaxfreq>maxfreq:
                this_fitspw = str(this_spw) + ":" + str(minfreq) + "~" + str(lineminfreq) + "GHz"
            else:
                this_fitspw = str(this_spw) + ":" + str(minfreq) + "~" + str(lineminfreq) + "GHz;" + \
                    str(linemaxfreq) + "~" + str(maxfreq) + "GHz"

            fitspw.append(this_fitspw)

        mymsmd.done()
        fitspw = ",".join(fitspw)

        # uvcontsub
        if not glob.glob(this_vis+".contsub"):
            print("# uvcontsub for " + this_vis+".contsub" + ".")
            print("# fitspw = " + fitspw)
            uvcontsub(
                vis = this_vis,
                fitspw = fitspw,
                want_cont = False,
                )
        else:
            print("# skip uvcontsub becase output exists!")
            print("  output = " + this_vis+".contsub")

        list_outputvis.append(this_vis+".contsub")

        if delete_intermediate==True:
            os.system("rm -rf " + this_vis)

    return list_outputvis

def run_mstransform(
    list_vis,
    obsfreq,
    linewidth,
    linename,
    arrayname,
    width=None,
    delete_intermediate=False,
    average=2,
    timebin="20s",
    ):

    print("\n###################")
    print("# run_mstransform #")
    print("###################")

    i = 0
    fieldnames = []
    list_outputvis = []
    for this_vis in list_vis:
        # get field name
        mymsmd.open(this_vis)
        fieldname = mymsmd.fieldsforintent("OBSERVE_TARGET#ON_SOURCE",True)[0]
        chanfreqs = mymsmd.chanfreqs(0) / 1e9
        chanwidth = (chanfreqs[1] - chanfreqs[0]) # GHz
        mymsmd.done()

        # get output name
        i += 1
        outputvis = fieldname+"_"+arrayname+"_"+str(i)+"_"+linename+".ms"

        # mstransform
        if not glob.glob(outputvis):
            print("# run mstransform for " + this_vis + ".")
            print(" output = " + outputvis)

            linewidthfreq = linewidth / 299792.0 * obsfreq # linewidth in GHz

            if width==None:
                width = chanwidth * average

            start = obsfreq - linewidthfreq/2.0
            nchan = int(linewidthfreq / width)
            width_kms = str(width / obsfreq * 299792.0) + "km/s"

            print("  start = " + str(start) + "GHz")
            print("  nchan = " + str(nchan))
            print("  width = " + str(width) + "GHz (" + width_kms + ")")

            mstransform(
                vis = this_vis,
                outputvis = outputvis,
                datacolumn = "DATA",
                combinespws = True,
                regridms = True,
                mode = "frequency",
                start = str(start) + "GHz",
                nchan = nchan,
                width = str(width) + "GHz",
                timeaverage = True,
                timebin = timebin,
                )

        else:
            print("# skip mstransform because output exists!")
            print("  output = " + outputvis)

        if glob.glob(outputvis):
            list_outputvis.append(outputvis)

    if delete_intermediate==True:
        os.system("rm -rf " + " ".join(list_vis))

    return list_outputvis,width

def run_concat(
    list_vis,
    fieldnames,
    linename,
    arrayname,
    drop_edge=None,
    delete_intermediate=False,
    ):

    print("\n##############")
    print("# run_concat #")
    print("##############")

    for this_field in fieldnames:
        inputvis = []
        outputvis = this_field + "_" + arrayname + "_" + linename + ".ms"

        for this_vis in list_vis:
            # get field name
            mymsmd.open(this_vis)
            fieldname = mymsmd.fieldsforintent("OBSERVE_TARGET#ON_SOURCE",True)[0]
            mymsmd.done()

            if fieldname==this_field:
                inputvis.append(this_vis)

        if not glob.glob(outputvis):
            print("# run concat")
            print("  input  = " + ", ".join(inputvis))
            print("  output = " + outputvis)
            concat(
                vis = inputvis,
                concatvis = outputvis + "_tmp1",
                freqtol = "5kHz",
                dirtol = "3arcsec",
                )

            if drop_edge!=None:
                mymsmd.open(outputvis+"_tmp1")
                nchan = mymsmd.nchan(0)
                mymsmd.done()

                spw = "0:" + str(drop_edge-1) + "~" + str(nchan-drop_edge-1)
                print("  splitting spw = " + spw)
                mstransform(
                    vis = outputvis + "_tmp1",
                    outputvis = outputvis,
                    spw = spw,
                    datacolumn = "DATA",
                    )
                os.system("rm -rf " + outputvis + "_tmp1")

            else:
                os.system("mv " + outputvis + "_tmp1 " + outputvis)

        else:
            print("# skip concat because output exists!")
            print("  output = " + outputvis)

    if delete_intermediate==True:
        os.system("rm -rf " + " ".join(list_vis))

########
# main #
########

# staging 12m ms
list_12m_vis = glob.glob(mskey_12m)
if glob.glob(mskey_12m):
    print("\n################################")
    print("# Found 12m ms files! Proceed. #")
    print("################################")

    list_12m_vis,fieldnames_12m = run_split(list_12m_vis,linename,obsfreq)

    list_12m_vis = run_uvcontsub(
        list_12m_vis,linename,obsfreq,linewidth,delete_intermediate)

    list_12m_vis,width_12m = run_mstransform(
        list_12m_vis,obsfreq,linewidth,linename,"12m",None,delete_intermediate)

    run_concat(list_12m_vis,fieldnames_12m,linename,"12m",drop_edge)

# staging 7m ms
list_7m_vis = glob.glob(mskey_7m)
if glob.glob(mskey_7m):
    print("\n###############################")
    print("# Found 7m ms files! Proceed. #")
    print("###############################")

    if mskey_12m is None:
        width = None
    else:
        width = width_12m

    list_7m_vis,fieldnames_7m = run_split(list_7m_vis,linename,obsfreq)

    list_7m_vis = run_uvcontsub(
        list_7m_vis,linename,obsfreq,linewidth,delete_intermediate)

    list_7m_vis,_ = run_mstransform(
        list_7m_vis,obsfreq,linewidth,linename,"7m",width,delete_intermediate)

    run_concat(list_7m_vis,fieldnames_7m,linename,"7m",drop_edge)

# create 12m+7m ms
list_12m7m_vis = []
if glob.glob(mskey_12m) and glob.glob(mskey_7m):
    print("\n###################################")
    print("# Found 12m+7m ms files! Proceed. #")
    print("###################################")

    fieldnames_12m7m = list(set(fieldnames_12m) & set(fieldnames_7m))
    list_12m7m_vis.extend(list_12m_vis)
    list_12m7m_vis.extend(list_7m_vis)
    run_concat(list_12m7m_vis,fieldnames_12m7m,linename,"12m+7m",drop_edge,delete_intermediate)

# clean up
os.system("rm -rf mstransform.last")
os.system("rm -rf uvcontsub.last")
os.system("rm -rf concat.last")
