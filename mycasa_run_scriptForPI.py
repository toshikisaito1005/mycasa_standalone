import os, sys, glob
import numpy as np
#

this_casa = "541"
galaxy    = "ngc1068_b3"

dir_uv    = "/lfs02/saitots/uvdata/"
proj_path = "/science*/group*/member*/"
uvlist    = np.loadtxt("uvlist.txt", dtype="str")
projects  = [dir_uv+s+proj_path for s in uvlist[uvlist[:,0]==galaxy][:,2]]
versions  = [s for s in uvlist[uvlist[:,0]==galaxy][:,1]]
dir_data  = []
for i in range(len(projects)):
    this_version = versions[i]
    if this_version==this_casa:
        this_path = glob.glob(projects[i])
        dir_data.extend(this_path)

### get current directory and ALMA data directories
dir_current = os.getcwd()

### scriptForPI loop
for i in range(len(dir_data)):
    #
    this_dir_data = dir_data[i]
    this_count = " (" + str(i+1) + "/" + str(len(dir_data)) + ")"
    this_this_dir_data_print = this_dir_data.split("/")[1] + this_count
    #
    os.chdir(this_dir_data)
    done = glob.glob("calibrated/")
    if not done:
        print("run : " + this_this_dir_data_print)
        os.chdir("script")
        execfile(glob.glob("*scriptForPI.py")[0])
    else:
        print("skip: " + this_this_dir_data_print)
    #
    os.chdir(dir_current)
