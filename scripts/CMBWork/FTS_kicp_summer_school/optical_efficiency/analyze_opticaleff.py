# analyze_opticaleff.py
#
# Some tools for measuring optical efficiencies from cold load data. The main
# outward-facing function in this module is 'fit_optical_eff', which will extract
# Psat values from 'drop_bolos' data and fit it to a functional form to measure
# the optical efficiencies.
#
# Adam Anderson
# adama@fnal.gov
# 4 May 2016

import analyze_IV
import analyze_GofT
import os
import numpy as np
from scipy.optimize import curve_fit
import scipy.integrate as sciint
import matplotlib.pyplot as plt
import cPickle as pickle
import re

# Load simulated transmission data for the triplexer. Simulations were
# performed by Donna Kubik, and can be obtained on the wiki here:
# https://pole.uchicago.edu/spt3g/index.php/Sonnet_Simulations_related_to_SPT-3G_pixels#08Sep2015_Triplexer_for_Wafer_20.2B
path, _ = os.path.split(__file__)
bands = np.loadtxt(path+'/triplexer_bands.csv', delimiter=',', skiprows=1)
transmission_freq = bands[:,0]*1e9
transmission = {'90': bands[:,1], '150': bands[:,2], '220': bands[:,3]}

# physical constants
hPlanck = 6.626e-34 # [J s]
kB = 1.381e-23 # [J / K]


def fit_optical_eff(data_files, T_ref=None, T_stage=None, GofT_data=None, plot_path=None, filter_data=None):
    '''
    Fits cold load data in order to extract optical efficiencies. The input data
    consist of IV curves taken at a set of cold load temperatures. After
    extracting Psat from each measurement, Psat is fit as a function of the cold
    load temperature.

    The model is a blackbody times the triplexer transmission function from
    Donna's Sonnet simulations. This function also does a standard linear fit to
    the cold load data to extract the response to optical power.

    An optional correction can be applied for heating of the wafer by optical
    loading. To perform this correction, you must specify ALL of the arguments
    'T_ref', 'T_stage', and 'GofT_data' described below. If all of the arguments
    are set to None, then no correction will be applied.

    Parameters:
    -----------
    data_files : dict
        {temperatures -> full path to pickle produced by 'drop_bolos'}
    T_ref (default=None) : float
        Stage temperature to which Psats should be adjusted or referenced (this
        is a correction for drift in the stage temperature due to loading from
        the cold load)
    T_stage (default=None) : dict
        {cold load temperatures -> stage temperatures}
    GofT_data (default=None) : dict
        {channel name -> list of fit parameters}
        Results of analyze_GofT.find_G_params
    plot_path (default=None) : str
        Name of figure to save

    Returns:
    --------
    model_params : dict
        {channel name -> list of fit parameters}
        Model fit parameters, list of the form:
        [Psat (without loading), optical eff.]
    linear_params : dict
        Linear fit parameters, list of the form:
        [slope, intercept]
    PsatVtemp : dict
        {channel name -> {'T': coldload temp, 'Psat': Psat at this temp.}}
        Psat data extracted from IV curves, which is used in the fit
    '''

    PsatVtemp = dict()
    # loop over temperatures and data to find Psats at each point
    for temp in data_files:
        Psats = dict()
        IV_file = file(data_files[temp])
        IV_data = pickle.load(IV_file)
        IV_file.close()
        Psat_at_T = analyze_IV.find_Psat(data_files[temp], R_threshold='pturn', Zp=[], plot_dir=None)
        
        for channum in Psat_at_T:
            boloname = IV_data['subtargets'][channum]['physical_name']
            if boloname not in PsatVtemp:
                    PsatVtemp[boloname] = dict()
                    PsatVtemp[boloname]['T'] = np.array(data_files.keys())
                    PsatVtemp[boloname]['Psat'] = np.zeros(len(PsatVtemp[boloname]['T']))
            PsatVtemp[boloname]['Psat'][temp==PsatVtemp[boloname]['T']] = Psat_at_T[channum]

    # now do the fitting of Psat(T)
    model_params = dict()
    linear_params = dict()
    for boloname in PsatVtemp:
        print boloname
        band = boloname.split('/')[1].split('.')[1]
        #reObj = re.search(r'Bolometer\((\S+)\)', boloname)
        #band = reObj.group(1).split('.')[1]
        cNotNormal = np.array(PsatVtemp[boloname]['Psat']) > 0.5e-12 # don't fit data for which bolometer may be normal
        if np.sum(cNotNormal == True) > 1:
            T_tofit = np.array(PsatVtemp[boloname]['T'])[cNotNormal]
            Psat_tofit = np.array(PsatVtemp[boloname]['Psat'])[cNotNormal]
            linear_params[boloname] = np.polyfit(T_tofit, Psat_tofit, 1)
            try:
                model_params[boloname], fit_cov = curve_fit(lambda T, Psat_el, eff: PsatofT_coldload(T, Psat_el, eff, band, filter_data),
                                                            np.array(PsatVtemp[boloname]['T'])[cNotNormal], np.array(PsatVtemp[boloname]['Psat'])[cNotNormal], p0=[50e-12, 1.2])
            except RuntimeError as err:
                print err
        else:
            model_params[boloname] = np.zeros(2)
            fit_cov = np.zeros((2,2))
            linear_params[boloname] = np.zeros(2)
    # make plots if desired
    # (don't make empty plots)
    if plot_path != None and len(PsatVtemp) > 0:
        f = plot_optical_eff(data_files.keys(), PsatVtemp, model_params, linear_params, filter_data)
        f.savefig(plot_path)

    return model_params, linear_params, PsatVtemp


def plot_optical_eff(T, Psat, model_params, linear_params, filter_data=None):
    '''
    Make plots of the results of the Psat(T) analysis.

    Parameters
    ----------
    T : numpy array
        Temperature data
    Psat : python dictionary
        Psat values corresponding to T, indexed by bolometer
    model_params : python list
        {channel name -> list of fit parameters}
        Model fit parameters, list of the form:
        [Psat (without loading), optical eff.]
    linear_params : python list
        Linear fit parameters, list of the form:
        [slope, intercept]

    Returns
    -------
    fig : handle to figure
    '''
    ncols = np.ceil(np.sqrt(len(Psat)))
    if ncols == 0:
        nrows = 0
    else:
        nrows = np.ceil(len(Psat) / ncols)

    fig = plt.figure(figsize=(ncols*5., nrows*3.))
    for jbolo, bolo in enumerate(Psat):
        #reObj = re.search(r'Bolometer\((\S+)\)', bolo)
        #boloname = reObj.group(1)
        #band = boloname.split('.')[1]
        band = bolo.split('/')[1].split('.')[1]

        plt.subplot(nrows, ncols, jbolo+1)
        plt.plot(Psat[bolo]['T'], 1.0e12*(Psat[bolo]['Psat']), marker='o', color='k', linestyle='None')

        plot_T = np.linspace(0., 30, 100)
        plot_Psat_linear = np.polyval(linear_params[bolo], plot_T)
        plot_Psat = PsatofT_coldload(plot_T, model_params[bolo][0], model_params[bolo][1], band, filter_data=filter_data)
        plt.plot(plot_T, 1.0e12*plot_Psat, color='b')
        plt.plot(plot_T, 1.0e12*plot_Psat_linear, color='r', linestyle='--')
        plt.grid(True)

        plt.xlabel('coldload temperature [K]')
        plt.ylabel('Electrical power on TES [pW]')

        ax = plt.gca()
        ax.annotate('Psat = %.1f' % (1e12*model_params[bolo][0]) ,xy=(.05,.4), xycoords='axes fraction')
        ax.annotate('$\eta$ = %.3f' % (model_params[bolo][1]), xy=(.05,.3), xycoords='axes fraction')
        ax.annotate('slope = %.2f pW/K' % (1e12*linear_params[bolo][0]), xy=(.05,.2), xycoords='axes fraction')
        ax.annotate(bolo, fontsize=14,xy=(.02,.89),xycoords='axes fraction')
    return fig


def PsatofT_coldload(T, Psat_electrical, optical_eff, band, filter_data=None):
    '''
    Functional form for fitting Psat as a function of coldload temperature in
    order to extract optical efficiency. The model is a blackbody times the
    triplexer transmission function from Donna's Sonnet simulations, and we
    assume that the cold load is beam-filling.

    Parameters
    ----------
    T : float
        Temperature of cold load
    Psat_electrical : float
        TES saturation power at zero optical loading (i.e. with
        loading from Joule heating only)
    optical_eff : float
        Optical efficiency
    band : str
        Frequency band '90', '150', or '220'
    filter_data : 2-tuple
        Filter transmission data: first entry is frequency of points, second
        entry is the transmission values of the points

    Returns
    -------
    Psat : float
        Saturation power
    '''
    if band not in ['90', '150', '220']:
        print('WARNING: Invalid band frequency supplied. Defaulting to 220GHz.')
        band = '220'

    def spectral_density(nu, temp):
        if filter_data is None:
            filter_TF = 1.0
        else:
            filter_TF = np.interp(nu, filter_data[0], filter_data[1])
        dPdnu = hPlanck * nu / (np.exp(hPlanck * nu / (kB * temp)) - 1) * \
            np.interp(nu, transmission_freq, transmission[band]) * \
            filter_TF * 1e12
        return dPdnu

    P_optical = np.zeros(len(T))
    for jT in range(len(T)):
        P_optical[jT], _ = sciint.quad(spectral_density, a=1e10, b=5e11, args=(T[jT]))
        P_optical[jT] = P_optical[jT] * optical_eff

    Psat = Psat_electrical - P_optical/1e12
    return Psat
