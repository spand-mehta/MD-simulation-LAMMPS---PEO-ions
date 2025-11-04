import sys
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas import DataFrame

#Global variables
Ex=0.15 #electric field
frame_nos=float('inf')

#Define parameters for block averaging
intrvl=10
num_blocks=11
block_size=301

#Following function returns x,y,z trajectory (unscaled), box dimensions, image flags, type of atoms (polymers/cation/anion), molecule number (pppmd function)
def read_lammpstrj(fname, num_frames=float('inf'), skip_beginning=0, skip_between=0):
    # helper function to read in the header and return the timestep, number of atoms, and box boundaries
    def read_header(f):
        # basic algo
        # read through box dimension initially
        # skip any beginning frames if required by specifying skip_beginning value
        # if not all frames to be read, specify them at num_frames value
        # update box dimensions for each timestep and keep track of x,y,z
        # xs, ys and zs are scaled that is between 0 and 1, unscale them
        # 1000 frames
        # r and ir are 3-d arrays for storing co-ord and image flags
        #

        f.readline()  # ITEM: TIMESTEP (this line is read)
        timestep = int(f.readline()) # value of current timestep

        f.readline()  # ITEM: NUMBER OF ATOMS
        num_atoms = int(f.readline()) # value of number of atoms

        f.readline()  # ITEM: BOX BOUNDS xx yy zz
        line = f.readline() # reads xlo and xhi
        line = line.split()
        xlo = float(line[0])
        xhi = float(line[1])
        line = f.readline()
        line = line.split()
        ylo = float(line[0])
        yhi = float(line[1])
        line = f.readline()
        line = line.split()
        zlo = float(line[0])
        zhi = float(line[1])

        return timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi

    # allow reading from standard input
    if not fname or fname == 'stdin':
        f = sys.stdin
    else:
        f = open(fname, 'r')

    # read in the initial header
    frame = 0
    init_timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_header(f)

    # skip the beginning frames, if requested
    for skippedframe in range(skip_beginning):
        f.readline()  # ITEM: ATOMS
        # loop over the atoms lines
        for atom in range(num_atoms):
            f.readline()
        init_timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_header(f)

    # preallocate arrays, if possible
    if num_frames < float('inf'):
        alloc = num_frames
        inf_frames = False
    else:
        alloc = 1
        inf_frames = True
    timestep = np.zeros(alloc,int)  # 1D array of timesteps
    box_bounds = np.zeros([alloc, 3, 2],
                          float)  # 3D array to store boundaries of the box, indexed by frame, x/y/z, then lower/upper

    timestep[frame] = init_timestep
    box_bounds[frame][0][0] = xlo
    box_bounds[frame][0][1] = xhi
    box_bounds[frame][1][0] = ylo
    box_bounds[frame][1][1] = yhi
    box_bounds[frame][2][0] = zlo
    box_bounds[frame][2][1] = zhi

    # NOTE: using num_atoms+1 here so that the arrays are indexed by their LAMMPS atom id
    r = np.zeros([alloc, num_atoms + 1, 3], float)  # 3D array of x, y, z coordinates, r[frame][id][coordinate]
    ir = np.zeros([alloc, num_atoms + 1, 3], int)  # 3D array of x, y, z image flags, r[frame][id][coordinate]

    id2mol = np.zeros(num_atoms + 1,
                      int)  # array to map from atom id to molecule id, builds this from the first frame, if available
    id2type = np.zeros(num_atoms + 1,
                    int)  # array to map from atom id to type, builds this from the first frame, if available

    # separately do the first ATOMS section so that we can initialize things, build the id2mol and id2type arrays, and so that the main loop starts with reading in the header
    line = f.readline()
    line = line.split()
    id_index = line.index("id") - 2
    if "mol" in line:
        mol_index = line.index("mol") - 2
    else:
        mol_index = None
    if "type" in line:
        type_index = line.index("type") - 2
    else:
        type_index = None

    if "x" in line:
        scaled = False
        x_index = line.index("x") - 2
        y_index = line.index("y") - 2
        z_index = line.index("z") - 2
    elif "xs" in line:
        scaled = True
        x_index = line.index("xs") - 2
        y_index = line.index("ys") - 2
        z_index = line.index("zs") - 2
    else:
        print (sys.stderr, "ERROR: x image flag not found in lammps trajectory")
        return

    if "ix" in line:
        ix_index = line.index("ix") - 2
        iy_index = line.index("iy") - 2
        iz_index = line.index("iz") - 2
    else:
        print (sys.stderr, "ERROR: x image flag not found in lammps trajectory")
        return

    # loop over the atoms lines for the first frame separately, the rest of the frames will be read in below
    for atom in range(num_atoms):
        line = f.readline()
        line = line.split()

        # get the atom id
        my_id = int(line[id_index])

        # x, y, z coordinates
        r[frame][my_id][0] = float(line[x_index])
        r[frame][my_id][1] = float(line[y_index])
        r[frame][my_id][2] = float(line[z_index])

        # unscale, if necessary
        if scaled:
            r[frame][my_id][0] = r[frame][my_id][0] * (box_bounds[frame][0][1] - box_bounds[frame][0][0]) + \
                                 box_bounds[frame][0][0]
            r[frame][my_id][1] = r[frame][my_id][1] * (box_bounds[frame][1][1] - box_bounds[frame][1][0]) + \
                                 box_bounds[frame][1][0]
            r[frame][my_id][2] = r[frame][my_id][2] * (box_bounds[frame][2][1] - box_bounds[frame][2][0]) + \
                                 box_bounds[frame][2][0]

        # x, y, z image flags
        ir[frame][my_id][0] = int(line[ix_index])
        ir[frame][my_id][1] = int(line[iy_index])
        ir[frame][my_id][2] = int(line[iz_index])

        if mol_index is not None:
            id2mol[my_id] = int(line[mol_index])
        if type_index is not None:
            id2type[my_id] = int(line[type_index])

    # build the reverse of the id2mol array
    # this is a 2D array with rows of (potentially) varying length, so nest a numpy array into a python list
    if mol_index is not None:
        num_mols = id2mol.max()
        mol2ids = [[]]
        for molid in range(1, num_mols + 1):
            mol2ids.append(np.where(id2mol == molid)[0])
    else:
        num_mols = None
        mol2ids = None

    # loop over number of num_frames frames, if num_frames is infinite, will loop over all the frames in the file
    frame = 1  # this is the frame counter for frames actually read in
    frame_attempt = 0  # this is the actual frame count in the file (not counting the ones skipped in the beginning
    while frame < num_frames:

        frame_attempt += 1

        # try to read in a new header
        try:
            my_timestep, my_num_atoms, my_xlo, my_xhi, my_ylo, my_yhi, my_zlo, my_zhi = read_header(f)
        except:
            print (sys.stderr, "WARNING: hit end of file when reading in", fname, "at frame", skip_beginning + frame_attempt)
            break

        #print(my_timestep,(my_timestep)*0.005)
        # skip the frame if between frames to be read in and restart the loop
        if frame_attempt % (skip_between + 1) > 0:
            f.readline()  # ITEM: ATOMS
            # loop over the atoms lines
            for atom in range(num_atoms):
                f.readline()
            continue

        # if we don't know how many frames to read in, have to allocate more memeory for the arrays
        if inf_frames:
            timestep = np.append(timestep, 0)

            box_bounds = np.concatenate((box_bounds, np.zeros([1, 3, 2], float)))

            r = np.concatenate((r, np.zeros([1, num_atoms + 1, 3], float)))
            ir = np.concatenate((ir, np.zeros([1, num_atoms + 1, 3], float)))

        # update the timestep and box size arrays
        timestep[frame] = my_timestep
        box_bounds[frame][0][0] = my_xlo
        box_bounds[frame][0][1] = my_xhi
        box_bounds[frame][1][0] = my_ylo
        box_bounds[frame][1][1] = my_yhi
        box_bounds[frame][2][0] = my_zlo
        box_bounds[frame][2][1] = my_zhi

        f.readline()  # ITEM: ATOMS
        # loop over the atoms lines
        for atom in range(num_atoms):
            line = f.readline()
            line = line.split()

            # get the atom id
            my_id = int(line[id_index])

            # x, y, z coordinates
            r[frame][my_id][0] = float(line[x_index])
            r[frame][my_id][1] = float(line[y_index])
            r[frame][my_id][2] = float(line[z_index])

            # unscale, if necessary
            if scaled:
                r[frame][my_id][0] = r[frame][my_id][0] * (box_bounds[frame][0][1] - box_bounds[frame][0][0]) + \
                                     box_bounds[frame][0][0]
                r[frame][my_id][1] = r[frame][my_id][1] * (box_bounds[frame][1][1] - box_bounds[frame][1][0]) + \
                                     box_bounds[frame][1][0]
                r[frame][my_id][2] = r[frame][my_id][2] * (box_bounds[frame][2][1] - box_bounds[frame][2][0]) + \
                                     box_bounds[frame][2][0]

            # x, y, z image flags
            ir[frame][my_id][0] = int(line[ix_index])
            ir[frame][my_id][1] = int(line[iy_index])
            ir[frame][my_id][2] = int(line[iz_index])

        frame += 1

    return r, ir, timestep, box_bounds, id2type, id2mol, mol2ids

#Function returns unwrapped co-ordinates
def r_unwrap(r,ir,box,id2type,id2mol,pol):
    frames = len(r)  # number of frames
    atoms = len(id2type)  # total atoms + 1
    if (pol=='bcp'):
        molecules = len(np.where(id2type == 1)[0])+len(np.where(id2type == 2)[0]) # total polymer beads
    else:
        molecules = len(np.where(id2type == 1)[0])  # total polymer beads

    nchain = len(np.where(id2mol == 1)[0])  # chain length
    ncord = len(r[0][0])  # x,y,z
    totalpoly = int(molecules / nchain)  # total number of polymers
    r_unwrap = np.zeros((frames, atoms, ncord))
    diffbox = np.diff(box)
    for i in range(frames):
        r_unwrap[i] = r[i] + ir[i] * diffbox[i].T
    return r_unwrap

#Returns MSD
def MSD(r, ir, box_bounds, id2type=[]):
    # set up some constants
    frames = len(r)
    box_size = np.array([ box_bounds[0][0][1] - box_bounds[0][0][0], box_bounds[0][1][1] - box_bounds[0][1][0], box_bounds[0][2][1] - box_bounds[0][2][0] ])

    #  allocate an array for the box center of mass which needs to be subtracted off
    box_com = np.zeros([frames,3], np.float)

    # preallocate msd vectors
    msd_dict = {}
    for type_id in set(id2type):
        msd_dict[type_id] = np.zeros(frames, np.float)

    # loop over frames
    for t in range(frames):
        # calculate the center of mass of the entire box
        for atom in range(1, len(r[0])):
            box_com[t] += r[t][atom] + ir[t][atom]*box_size
        box_com[t] = box_com[t]/(len(r[0])-1)


        # loop over atoms
        for atom in range(1, len(id2type)):
            # calculate how much the bead has moved reletive to the center of mass (note that this is a vector equation)
            diff = (r[t][atom] + ir[t][atom]*box_size - box_com[t]) - (r[0][atom] + ir[0][atom]*box_size - box_com[0])
            # the mean squared displacement is this difference dotted with itself
            msd_dict[id2type[atom]][t] += diff.dot(diff)


    # scale MSD by the number of beads of each type, to get the average MSD
    for type_id in set(id2type):
        msd_dict[type_id] = msd_dict[type_id]/sum(id2type == type_id)
    del msd_dict[0] # this is needed since id2type has a dummy entry of 0 at index 0 so that it is indexed by LAMMPS atom_id

    return msd_dict

#Returns MSD in direction of electric field (x) and yz directions
def MSD_kevin(r, ir, box_size, box_com, id2type, nofield_dir): 
    # set up some constants
    frames = len(r)
    # preallocate msd vectors
    dr_dict_efield = {}
    msd_dict_efield = {}
    msd_dict_noefield ={}
    dr_dict_efield1 = {}
    msd_dict_efield1 = {}
    msd_dict_noefield1 = {}
    for type_id in set(id2type):
        dr_dict_efield[type_id] = np.zeros(frames,float)
        msd_dict_efield[type_id] = np.zeros(frames, float)
        msd_dict_noefield[type_id] = np.zeros(frames, float)
        dr_dict_efield1[type_id] = np.zeros(frames,float)
        msd_dict_efield1[type_id] = np.zeros(frames, float)
        msd_dict_noefield1[type_id] = np.zeros(frames, float)

    # loop over frames
    for t in range(frames):
        # loop over atoms
        for atom in range(1, len(id2type)):
            # calculate how much the bead has moved reletive to the center of mass (note that this is a vector equation)
            r_t = r[t][atom] + ir[t][atom]*box_size[t] - box_com[t]
            r_0 = r[0][atom] + ir[0][atom]*box_size[0] - box_com[0]
            diff = r_t - r_0

            dr_dict_efield[id2type[atom]][t] += diff[0]
            # the mean squared displacement is this difference dotted with itself
            msd_dict_efield[id2type[atom]][t] += diff[0]*diff[0]
            if nofield_dir == 'yz':
                msd_dict_noefield[id2type[atom]][t] += (diff[1]*diff[1]+diff[2]*diff[2])
            elif nofield_dir == 'y':
                msd_dict_noefield[id2type[atom]][t] += diff[1]*diff[1]

        #print(t,msd_dict_noefield[1][t],msd_dict_noefield[2][t],msd_dict_noefield[3][t])

    # scale MSD by the number of beads of each type, to get the average MSD
    for type_id in set(id2type):
        dr_dict_efield[type_id] = dr_dict_efield[type_id]
        msd_dict_efield[type_id] = msd_dict_efield[type_id]
        msd_dict_noefield[type_id] = msd_dict_noefield[type_id]
        dr_dict_efield1[type_id] = dr_dict_efield[type_id]/sum(id2type == type_id)
        msd_dict_efield1[type_id] = msd_dict_efield[type_id]/sum(id2type == type_id)
        msd_dict_noefield1[type_id] = msd_dict_noefield[type_id]/sum(id2type == type_id)
    #for i in range(frames):
        #print(i,"msd_x,polymer",msd_dict_efield[1][i],"cation:",msd_dict_efield[2][i],"anion:",msd_dict_efield[3][i])

    del dr_dict_efield[0] # this is needed since id2type has a dummy entry of 0 at index 0 so that it is indexed by LAMMPS atom_id
    del msd_dict_efield[0]
    del msd_dict_noefield[0]
    del dr_dict_efield1[0] # this is needed since id2type has a dummy entry of 0 at index 0 so that it is indexed by LAMMPS atom_id
    del msd_dict_efield1[0]
    del msd_dict_noefield1[0]

    return dr_dict_efield1, msd_dict_efield1, msd_dict_noefield1,dr_dict_efield, msd_dict_efield, msd_dict_noefield

#Calculation of radius of gyration
def rg(r,ir,box,id2type,id2mol,pol):
    frames=len(r)
    atoms=len(id2type)
    if (pol=='bcp'):
        molecules=len(np.where(id2type==1)[0])+len(np.where(id2type==2)[0])
    else:
        molecules = len(np.where(id2type == 1)[0])
    nchain = len(np.where(id2mol==1)[0])
    ncord=len(r[0][0])
    totalpoly=int(molecules/nchain)
    rgyr=np.zeros((frames,totalpoly+1))
    r_unwrap = np.zeros((frames, atoms, ncord))
    diffbox = np.diff(box)
    for i in range(frames):
        r_unwrap[i] = r[i] + ir[i] * diffbox[i].T
    for i in range(frames):
        for j in range(totalpoly):
            poly=np.array_split(r_unwrap[i][1:molecules+1],totalpoly)[j]
            polymean=np.mean(poly,axis=0)
            rcom=(poly-polymean)**2
            rgyr[i][j]=np.sqrt(np.sum(rcom)/nchain)

    return r_unwrap,rgyr

#Calculate end-to-end distance
def endtoend(r,ir,box,id2type,id2mol,pol):
    frames = len(r)
    atoms = len(id2type)
    if (pol=='bcp'):
        molecules=len(np.where(id2type==1)[0])+len(np.where(id2type==2)[0])
    else:
        molecules = len(np.where(id2type == 1)[0])
    nchain = len(np.where(id2mol == 1)[0])
    ncord = len(r[0][0])
    totalpoly = int(molecules / nchain)
    r_unwrap = np.zeros((frames, atoms, ncord))
    diffbox=np.diff(box)
    for i in range(frames):
        r_unwrap[i] = r[i] + ir[i] * diffbox[i].T
    rendtoend = np.zeros((frames, totalpoly, ncord))
    for i in range(frames):
        rendtoend[i] = (r_unwrap[i][0:molecules+1][nchain::nchain] - r_unwrap[i][0:molecules+1][1::nchain]) ** 2
    endtoend = np.zeros((frames, 1, totalpoly))
    for i in range(frames):
        endtoend[i] = np.sum(rendtoend[i], axis=1)
    return np.sqrt(endtoend)

#Fitting function, required if you are calculating drift velocity based on K.Shen 2020 paper
def func(t,a,b):
    return a+b*t**2

#Block averaging MSD
def block_average(intrvl, num_blocks, block_size, timescale,r,ir,timestep,boxbds,id2type,id2mol,mol2ids):

    if not (num_blocks-1)*intrvl + block_size == len(r):
        print (sys.stderr, "WARNING: not all timesteps in the dump file are utilized.")


    msd = {}
    msd_avg = {}
    time_range = np.zeros(num_blocks)

    for n in range(num_blocks):
        low = n*intrvl
        high = n*intrvl + block_size
        print ("calculating MSD in block ", n, "with timesteps between:", timestep[low], timestep[high-1])
        msd[n] = MSD(r[low:high], ir[low:high], boxbds[low:high], id2type)
        time_range[n]=(timestep[low]+timestep[high-1])/2*timescale
        #print (n,low,high,time_range[n])

    for n in range(num_blocks):
        for type_id in sorted(msd[n]):
            if n ==0:
                msd_avg[type_id] = np.zeros(block_size, np.float)
            print (n,type_id,msd_avg[type_id],msd[n][type_id])
            msd_avg[type_id] += msd[n][type_id]/float(num_blocks)

    return num_blocks, time_range, msd, msd_avg

#End-to-end ACF
def end2end_autocorr(r, ir, box_bounds, mol2ids):
    frames = len(r)
    mols = len(mol2ids)

    # preallocate e2e vector arrays and autocorr array
    e2e_t = np.zeros([mols, 3], np.float)
    e2e_0 = np.zeros([mols, 3], np.float)
    e2e_autocorr = np.zeros(frames, np.float)

    # loop over time
    for t in range(41,frames):
    #for t in range(frames):
        box_size = np.array([ box_bounds[t][0][1] - box_bounds[t][0][0], box_bounds[t][1][1] - box_bounds[t][1][0], box_bounds[t][2][1] - box_bounds[t][2][0] ])

        # loop over molecules
        for molid in range(1, mols):
            # assume that the ends of the chain have the maximum and minimum id numbers
            id1 = mol2ids[molid].min()
            id2 = mol2ids[molid].max()

            # calculate the end-to-end vector
            r1 = r[t][id1] + ir[t][id1]*box_size
            r2 = r[t][id2] + ir[t][id2]*box_size

            e2e_t[molid] = r2 - r1
            if t == 41:
            #if t==0:
                e2e_0[molid] = e2e_t[molid]

            # take dot products
            e2e_autocorr[t] += np.dot(e2e_0[molid], e2e_t[molid])
            #print ("ACF:",t,molid,box_size,e2e_autocorr[t])

    # scaling
    e2e_autocorr = e2e_autocorr/(mols-1)
    e2e_autocorr = e2e_autocorr/e2e_autocorr[41]
    #e2e_autocorr = e2e_autocorr / e2e_autocorr[0]
    #print(e2e_autocorr[11])
    return e2e_autocorr

#Block averaged end-to-end ACF
def block_average_end(intrvl, num_blocks, block_size, timescale,r,ir,timestep,boxbds,id2type,id2mol,mol2ids):

    if not (num_blocks-1)*intrvl + block_size == len(r):
        print (sys.stderr, "WARNING: not all timesteps in the dump file are utilized.")

    intrvl = 31
    num_blocks = 11
    block_size = 50
    e2eavg = np.zeros(block_size, np.float)
    e2eacf = {}

    skip=41

    reqdn=-1
    for n in range(num_blocks):
        low = skip+n*intrvl
        high = skip+n*intrvl + block_size

        if(low<401):
            reqdn+=1
            print(n, reqdn,low, high)
            #print("calculating <R^2ACF> in block ", n, "with timesteps between:", timestep[low], timestep[high - 1])
            e2eacf[n] = end2end_autocorr(r[low:high], ir[low:high], boxbds[low:high],mol2ids)


    sum=0
    #reqdn <-> num_blocks
    for n in range(reqdn):
        #print (n,type_id,e2eavg[type_id],e2eacf[n])
        e2eavg += e2eacf[n]
        #print(n,e2eavg,e2eacf[n])
    e2eavg=e2eavg/float(reqdn)

    return e2eavg
#End of functions


file="hp_Li_divalent_delta_1_S_50.lammpstrj"  #filename
poltype='hp'  #type of polymer-homopolymer (hp) or blockcopolymer (bcp)
model = 1  #type of model for TFSI representation - model 1 is assuming it as single bead and model 2 is assuming it as 3 connected beads. For other anions, model 1 is used.
data=read_lammpstrj(file, num_frames=frame_nos, skip_beginning=skip_starting, skip_between=0) #read the trajectory file in unscaled
id2type=data[4]  # array of type of atom:polymer/cation/anion
unwrapped=r_unwrap(data[0],data[1],data[3],data[4],data[5],poltype) #function to unwrap the co-ordinates if required
box_com=np.mean(unwrapped,axis=1) #function to find COM of Box
box_size=np.diff(data[3]).reshape(len(data[0]),3) #find length, breaddth and height of box
end=end2end_autocorr(data[0], data[1], data[3], data[6])  #find end-to-end ACF
#end_to_end=endtoend(data[0],data[1],data[3],data[4],data[5],poltype) #find end-to-end distance (uncomment to use)
#radius=rg(data[0],data[1],data[3],data[4],data[5],pol) #find radius of gyration (uncomment to use)

#Calculations begin
time1=data[2]*0.005  #time in tau units (should be changed for NPT file since, initial timesteps are run with dt=0.001 and then dt=0.005 used for remaining)
time=time1-time1[0]  #time array in tau units
nofield_dir = 'yz'  #direction perpendicular to electric field for nvt system
msd_kevin=MSD_kevin(data[0], data[1], box_size, box_com, data[4], nofield_dir) #returns MSD based on direction of nofield
#msd_block=block_average(intrvl,num_blocks,block_size,0.005,data[0],data[1],data[2],data[3],data[4],data[5],data[6]) #block averaged MSD(uncomment to use)
#end_to_end_avg=block_average_end(intrvl,num_blocks,block_size,0.005,data[0],data[1],data[2],data[3],data[4],data[5],data[6]) #block averaged end-to-end


dispsum=0
dispsum_an=0

#below is calculation of number of cations and anions for homopolymer and block co-polymer systems
if(poltype=='hp'):
    start=len(np.where(data[4]==1)[0])
    if(model==1):
        cations=len(np.where(data[4]==2)[0])
        anions=len(np.where(data[4]==3)[0])
    else:
        cations = len(np.where(data[4] == 2)[0])
        anions = len(np.where(data[4] == 3)[0])+len(np.where(data[4] == 4)[0])
else:
    start = len(np.where(data[4] == 1)[0])+len(np.where(data[4] == 2)[0])
    cations = len(np.where(data[4] == 3)[0])

#calculation of displacement of cations and anions for calculation of drift velocity and then ion mobility based on drift velocity
for i in range(start+1, start+cations+1):
    disp = unwrapped[-1][i][0]-unwrapped[0][i][0]
    dispsum += disp

for i in range(start+cations+1,start+cations+anions+1):
    disp_an = unwrapped[-1][i][0]-unwrapped[0][i][0]
    dispsum_an += disp_an

tot_time=time[-1]-time[0] #total time for displacement in tau units

print("Cation mobility:",dispsum/cations/tot_time/0.15,
      "Anion mobility:",dispsum_an/anions/tot_time/0.15,
      "Cation drift velocity:",dispsum/cations/tot_time,
      "Anion drift velocity:",dispsum_an/anions/tot_time)

# Calculation of total number of ions
if(poltype!='bcp'):
    if (model==1):
        N_ions = len(np.where(data[4] == 2)[0]) + len(np.where(data[4] == 3)[0])
    else:
        N_ions = len(np.where(data[4] == 2)[0]) + len(np.where(data[4] == 3)[0])+len(np.where(data[4] == 4)[0])
else:
    N_ions=len(np.where(data[4]==3)[0])+len(np.where(data[4]==4)[0])

# Calculation of molar and total conductivities
molar_conductivity=(abs(ion_mobility_cat)+abs(ion_mobility_an))/np.prod(box_size[0])
total_conductivity=(molar_conductivity)*N_ions
print ("cation:",diffusivity_cat,"anion:",diffusivity_an)
print("total conductivity:",total_conductivity)
print("box volume:",np.prod(box_size[0]))


