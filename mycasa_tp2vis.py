import os, sys, glob
import numpy as np
execfile("/home02/saitots/scripts/analysis_scripts/tp2vis.py")

uvweight  = np.loadtxt("uvweight.txt",dtype="str")
projects  = uvweight[:,0][uvweight[:,0]!="tbe"]
values_str = uvweight[:,1][uvweight[:,0]!="tbe"]
msfiles   = uvweight[:,2][uvweight[:,0]!="tbe"]

#
dir_current = os.getcwd()

for i in range(len(projects)):
    this_project = projects[i]
    this_value   = float(values_str[i])
    this_7m_ms   = msfiles[i]
    this_12m_ms  = this_7m_ms.replace("_7m_","_12m_")
    os.chdir(this_project)
    #
    print("# tp2viswt " + this_7m_ms + " mode=multiply value=" + str(this_value))
    this_7m_ms_wt = this_7m_ms + ".tp2viswt"
    os.system("cp -r " + this_7m_ms + " " + this_7m_ms_wt)
    concatvis     = this_7m_ms.replace("_7m_","_12m+7m_")
    tp2viswt(this_7m_ms_wt, mode="multiply", value=this_value)
    tp2vispl([this_7m_ms_wt,this_12m_ms])
    concat(vis=[this_7m_ms_wt,this_12m_ms], concatvis=concatvis, freqtol="20MHz", dirtol="4arcsec")
    #
    os.chdir(dir_current)
