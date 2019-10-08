import  analyze_IV as AIV
import  analyze_opticaleff as AOE
import cPickle as pickle
import argparse as ap
import os.path
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sb

# parse arguments
P0 = ap.ArgumentParser(description='Analyze optical efficiency data',
                       formatter_class=ap.ArgumentDefaultsHelpFormatter)
P0.add_argument('hkdata', action='store', default=None,
                help='Pickle file with housekeeping data.')
P0.add_argument('--output-dir', action='store', default=os.getcwd(),
                help='Directory to store output.')
P0.add_argument('--file-prefix', action='store', default='',
                help='Prefix to append to output filenames')
args = P0.parse_args()
                
jT_exclude = [] #[0] #[0, 1, 9, 10, 11]

# aesthetics
sb.set_style('white')
sb.set_context("notebook", font_scale=1.5)
cmap3 = sb.color_palette("Set1", n_colors=3)



# load and wrangle housekeeping data
hk_data = pickle.load(file(args.hkdata, 'r'))
coldloadtemp = (np.array(hk_data['blackbody start temp']) +
                np.array(hk_data['blackbody stop temp'])) / 2.0

drop_bolos_pkl_files = [os.path.basename(fname)
                        for fname in glob.glob('{}/data/*OUTPUT.pkl'
                                               .format(hk_data['drop_bolos datadir'][-1]))] # if 'Mezz_1' in os.path.basename(fname)]

# loop over temperatures and fit for Psat at each
all_fit_params = dict()
all_linear_params = dict()
for filename in drop_bolos_pkl_files:
    IV_datafiles = {coldloadtemp[jT]: '{}/data/{}'
                    .format(hk_data['drop_bolos datadir'][jT], filename)
                    for jT in range(len(coldloadtemp))}
    fit_params, linear_params, PsatVtemp = AOE.fit_optical_eff(IV_datafiles,
                                                plot_path='{}/{}_{}_opticaleff.png'
                                                .format(args.output_dir,
                                                        args.file_prefix,
                                                        os.path.basename(filename.strip('.pkl'))))
    all_fit_params.update(fit_params)
    all_linear_params.update(linear_params)

# reformat the dictionary of fit parameters to be more legible
save_fit_params = dict()
for bolo in all_fit_params.keys():
    save_fit_params[bolo] = dict()
    save_fit_params[bolo]['Psat270_CL'] = all_fit_params[bolo][0]
    save_fit_params[bolo]['opteff_CL'] = all_fit_params[bolo][1]
    save_fit_params[bolo]['slope_CL'] = all_linear_params[bolo][0]

data_file = file('{}/{}_optical_eff_fit_results.pkl'.format(args.output_dir, args.file_prefix), 'w')
pickle.dump(save_fit_params, data_file)
data_file.close()

opteff_CL = np.array([save_fit_params[boloname]['opteff_CL'] for boloname in save_fit_params.keys()])
slope_CL = np.array([save_fit_params[boloname]['slope_CL'] for boloname in save_fit_params.keys()])
c90 = np.array([boloname.split('.')[1] == '90' for boloname in save_fit_params.keys()])
c150 = np.array([boloname.split('.')[1] == '150' for boloname in save_fit_params.keys()])
c220 = np.array([boloname.split('.')[1] == '220' for boloname in save_fit_params.keys()])

varlist = [opteff_CL, slope_CL*1e12]
labellist = ['optical efficiency', 'slope [pW / K]']
savelabel = ['efficiency', 'slope']
rangelist = [(0, 1.8), (-1,0)]
nbins = [36, 36]
nvar = len(varlist)
for ivar, xvar in enumerate(varlist):
    plt.figure()
    plt.hist(xvar[c90==True], bins=np.linspace(rangelist[ivar][0], rangelist[ivar][1], nbins[ivar]), histtype='stepfilled', alpha=0.3, color=cmap3[0])
    plt.hist(xvar[c90==True], bins=np.linspace(rangelist[ivar][0], rangelist[ivar][1], nbins[ivar]), histtype='step', label='90 GHz', color=cmap3[0], linewidth=2)
    plt.hist(xvar[c150==True], bins=np.linspace(rangelist[ivar][0], rangelist[ivar][1], nbins[ivar]), histtype='stepfilled', alpha=0.3, color=cmap3[1])
    plt.hist(xvar[c150==True], bins=np.linspace(rangelist[ivar][0], rangelist[ivar][1], nbins[ivar]), histtype='step', label='150 GHz', color=cmap3[1], linewidth=2)
    plt.hist(xvar[c220==True], bins=np.linspace(rangelist[ivar][0], rangelist[ivar][1], nbins[ivar]), histtype='stepfilled', alpha=0.3, color=cmap3[2])
    plt.hist(xvar[c220==True], bins=np.linspace(rangelist[ivar][0], rangelist[ivar][1], nbins[ivar]), histtype='step', label='220 GHz', color=cmap3[2], linewidth=2)
    plt.xlim(rangelist[ivar])
    plt.xlabel(labellist[ivar])
    plt.legend()
    plt.savefig('{}/{}_{}_CL_freq.png'.format(args.output_dir, args.file_prefix, savelabel[ivar]))
    
