import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import seaborn as sns
from pylab import setp
import argparse

fs = 22

sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['savefig.format'] = "pdf"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    lw=5
    setp(bp['boxes'][0], color='red',linewidth=lw)
    setp(bp['medians'][0], color='orange',linewidth=lw)

    setp(bp['boxes'][1], color='cyan',linewidth=lw)
    setp(bp['medians'][1], color='orange',linewidth=lw)

    # setp(bp['boxes'][2], color='blue',linewidth=lw)
    # setp(bp['medians'][2], color='orange',linewidth=lw)

    # setp(bp['boxes'][3], color='green',linewidth=lw)
    # setp(bp['medians'][3], color='orange',linewidth=lw)

showfliers= False

parser = argparse.ArgumentParser()
parser.add_argument('--noise_levels',type=float, nargs='+', required=True)
parser.add_argument('--num_reduced_sets',type=int, nargs='+', required=True)
parser.add_argument('--num_obs',type=int, nargs='+', required=True)
parser.add_argument('--num_prime',type=int, nargs='+', required=True)
parser.add_argument('--noises',type=str, nargs='+', required=True)

args = parser.parse_args()

list_noises = args.noises
list_num_prime = args.num_prime
list_noise_levels = args.noise_levels
list_num_reduced = args.num_reduced_sets
list_num_obs = args.num_obs 

len_num_red = len(list_num_reduced)
len_num_noise_levels = len(list_noise_levels)

pos = 4

num_exps = 1

for noise in list_noises:
    for num_prime in list_num_prime:
        
        fig, axs = plt.subplots(len_num_red,len_num_noise_levels, figsize=(12,6),layout="constrained")
        
        if noise=="gaussian":
            # add a big axis, hide frame
            fig.add_subplot(len_num_red,len_num_noise_levels,5, frameon=False)

            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel('(a) Low gaussian noise', fontweight = "bold", fontsize = fs,loc="center")

            # add a big axis, hide frame
            fig.add_subplot(len_num_red,len_num_noise_levels,6, frameon=False)

            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel('(b) High gaussian noise', fontweight = "bold", fontsize = fs,loc="center")
        
        else:
            # add a big axis, hide frame
            fig.add_subplot(len_num_red,len_num_noise_levels,5, frameon=False)

            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel('(a) Low beta noise', fontweight = "bold", fontsize = fs,loc="center")

            # add a big axis, hide frame
            fig.add_subplot(len_num_red,len_num_noise_levels,6, frameon=False)

            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel('(b) High beta noise', fontweight = "bold", fontsize = fs,loc="center")
    
        for i,num_reduced in enumerate(list_num_reduced):
            for j,noise_level in enumerate(list_noise_levels):
                coll_mmd_opt,coll_cvar,coll_mmd_random = [],[],[]
               
                filename = "./stats/cut_in/{}_noise/noise_{}/ts_{}/{}_samples_{}_obs.npz".format(noise,int(noise_level*100),
                                    num_prime,
                                    num_reduced,list_num_obs[0])

                temp_file = np.load(filename)

                coll_mmd_opt.extend((temp_file["coll_mmd_opt"]/1000)*100)
                # coll_mmd_random.extend((temp_file["coll_mmd_random"]/1000)*100)
                coll_cvar.extend((temp_file["coll_cvar"]/1000)*100)

                filename = "./stats/overtake/{}_noise/noise_{}/ts_{}/{}_samples_{}_obs.npz".format(noise,int(noise_level*100),
                                    num_prime,
                                    num_reduced,list_num_obs[0])

                temp_file = np.load(filename)

                coll_mmd_opt.extend((temp_file["coll_mmd_opt"]/1000)*100)
                # coll_mmd_random.extend((temp_file["coll_mmd_random"]/1000)*100)
                coll_cvar.extend((temp_file["coll_cvar"]/1000)*100)
            
                if len(coll_mmd_opt)==0 or len(coll_cvar)==0:
                    continue

                data = [coll_mmd_opt,coll_cvar]

                x_synthetic = np.array([0.5,1.5]) # the label locations
                widths = 0.8
                width= 1.0
                whiskerprops = dict(linestyle='-',linewidth=5.0, color='black')

                bp = axs[i,j].boxplot(data,showfliers=showfliers,widths=widths,whiskerprops=whiskerprops)
                setBoxColors(bp)

                axs[i,j].set_xticklabels([])

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                # place a text box in upper left in axes coords
                textstr = "$N={}$".format(num_reduced)
                axs[i,j].text(0.05, 0.95, textstr, transform=axs[i,j].transAxes, fontsize=fs,
                verticalalignment='top', bbox=props)

                labels_synthetic = ["$r_{MMD}^{emp}$","$r_{CVaR}^{emp}$"]
                axs[i,j].set_xticks(x_synthetic + width / 2, labels_synthetic)
                
                axs[i,j].set_ylabel('%Collisions', fontweight = "bold", fontsize = fs,loc="center")


plt.show()

