# Import modules
import csv
import numpy as np
import os
from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import itertools

def reference_values():
    """Returns dict with hardcoded parameters per function."""

    NoiseTypes = ['baseline', '1AM', '3AM', '5AM', '7AM', 'DNT', '11PM', '1PM', '3PM', '5PM', '7PM', 'NNT', '11PM']
    noisefloor = {8e3: -5, 12e3: -5, 16e3: 0, 24e3: 0, 32e3: 5}

    # colormap
    colors = [(0, 0, 1), (1, .9, 0), (1, .5, 0), (1, 0, 0), (0, 0, 1)]  # B -> Y -> O -> R -> B
    cmap = LinearSegmentedColormap.from_list('circadian', colors)
    colormap = dict(zip(NoiseTypes[1:], cmap(np.linspace(0, 1, len(NoiseTypes)))))
    colormap['ZT3'] = colormap.pop('DNT')
    colormap['ZT15'] = colormap.pop('NNT')
    colormap['baseline'] = np.ones(3)*.6
    colormap_alternate = {'baseline': (.4, .4, .4), 'ZT3': 'r', 'ZT15': 'b', 'ZT23': 'r', 'ZT1': 'r', 'ZT11': 'b', 'ZT13': 'b'}

    # load_data_dp
    load_data_dp_dict = {}
#    MasterExperimentList_dir = 'G:\\Forskning\\Canlons Group\\Auditory Physiology\\Projects\\DoD\\CVanalysis\\MasterExperimentList.csv'
    MasterExperimentList_dir = '/Users/corstiaen/OneDrive - Karolinska Institutet/CVanalysis/MasterExperimentList.csv'
    with open(MasterExperimentList_dir, 'r') as f:
        reader = csv.reader(f)
        experimenter = {}
        for row in reader:
            experimenter[row[0]] = row[1:]
    load_data_dp_dict['experimenter'] = experimenter
    load_data_dp_dict['supplier'] = np.array(['Jackson', 'Janvier', 'Scanbur'])
    load_data_dp_dict['NoiseSPL'] = np.array(['100 dB', '103 dB', '105 dB'])
    load_data_dp_dict['ABRtimes'] = np.array(['baseline', '24h', '2w'])
    load_data_dp_dict['NoiseTypes'] = np.array(NoiseTypes)
    
    # detect_peaks
    detect_peaks_dict = {}
    detect_peaks_dict['nf_neighbors'] = 16
    detect_peaks_dict['comp_neighbors'] = 1
    detect_peaks_dict['thr_dB_anf'] = 10

    # plot_io_dp
    plot_io_dp = {}
    plot_io_dp['jitter_factor'] = 1.
    plot_io_dp['noisefloor'] = noisefloor

    # plot_area_under_curve_dp
    plot_area_under_curve_dp = {}
    plot_area_under_curve_dp['noisefloor'] = noisefloor
    plot_area_under_curve_dp['alpha'] = .05

    # plot_threshold_dp
    plot_threshold_dp = {}
    plot_threshold_dp['noisefloor'] = noisefloor
    plot_threshold_dp['alpha'] = .05

    # Pool into single output
    reference_dict = {'colormap': colormap,
                      'colormap_alternate': colormap_alternate,
                      'load_data_dp': load_data_dp_dict,
                      'detect_peaks': detect_peaks_dict,
                      'plot_io_dp': plot_io_dp,
                      'plot_area_under_curve_dp': plot_area_under_curve_dp,
                      'plot_threshold_dp': plot_threshold_dp}

    return reference_dict


def awfread(path):
    """TDT .awf file reader.

    Parameters
    ----------
    path : string
        String containing the path to the awf file to be imported.
    
    Returns
    -------
    data : dictionary
        Dictionary containing all data from specified awf file. The waveforms
        can be found in
        
            data['groups'][i]['wave']
            
        where i will be in range(0, 30).
    """
    
    # Initialize parameters
    isRZ = False
    
    RecHead = dict()
    groups = []
    data = dict()
    
    with open(path, 'rb') as fid:
        
        # Read RecHead data
        RecHead['nens'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead['ymax'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        RecHead['ymin'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        RecHead['autoscale'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        
        RecHead['size'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead['gridsize'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead['showgrid'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        
        RecHead['showcur'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        
        RecHead['TextMargLeft'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead['TextMargTop'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead['TextMargRight'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead['TextMargBottom'] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        bFirstPass = True
        
        for x in range(0, 30):
            
            # Create dict for looping
            loop_groups = dict()
            
            loop_groups['recn'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['grpid'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            
            # Read temporary timestamp
            if bFirstPass:
                ttt = np.fromfile(fid, dtype=np.int64, count=1)
                fid.seek(-8, 1)
                # Make sure timestamps make sense.
                if datetime.now().toordinal() - (ttt / 86400 + date.toordinal(date(1970, 1, 1))) > 0:
                    isRZ = True
                    data['fileTime'] = datetime.fromtimestamp(ttt).strftime("%Y-%m-%d %H:%M:%S")
                    data['fileType'] = 'BioSigRZ'
                else:
                    ttt = np.fromfile(fid, dtype=np.uint32, count=1)
                    data['fileTime'] = datetime.fromtimestamp(ttt).strftime("%Y-%m-%d %H:%M:%S")
                    fid.seek(-4, 1)
                    data['fileType'] = 'BioSigRP'
                bFirstPass = False
            
            if isRZ:
                loop_groups['grp_t'] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups['grp_t'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            loop_groups['newgrp'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['sgi'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            
            # CAREFUL! TROUBLE IN CONVERTING FROM MATLAB CODE.
            # I get a '\01' byte, which translates to "start of heading"
            # Using int8 seems to give the right channel number, but this might be wrong.
            # MATLAB code:
                # groups['chan'].iloc[x] = int16(fread(fid,1,'char'))
                # groups['rtype'].iloc[x] = int16(fread(fid,1,'char'))
            loop_groups['chan'] = np.fromfile(fid, dtype=np.int8, count=1)[0]
            loop_groups['rtype'] = np.fromfile(fid, dtype=np.int8, count=1)[0]
            
            loop_groups['npts'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['osdel'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups['dur'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups['srate'] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            loop_groups['arthresh'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups['gain'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups['accouple'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            
            loop_groups['navgs'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['narts'] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            if isRZ:
                loop_groups['beg_t'] = np.fromfile(fid, dtype=np.int64, count=1)[0]
                loop_groups['end_t'] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups['beg_t'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                loop_groups['end_t'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            
            tmp = np.zeros(10)
            for i in range(0,10):
                tmp[i] = np.fromfile(fid, dtype=np.float32, count=1)
            loop_groups['vars'] = tmp
            
            cursors = []
            for i in range(0,10):
                loop_cursors = dict()
                loop_cursors['tmar'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
                loop_cursors['val'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
                tmp_str = fid.read(20).decode('utf-8').split('\0')
                loop_cursors['desc'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
                loop_cursors['xpos'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors['ypos'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors['hide'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors['lock'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                cursors.append(loop_cursors)
            loop_groups['cursors'] = cursors
            
            # Open the group
            loop_groups['grpn'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['frecn'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['nrecs'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            tmp_str = fid.read(16).decode('utf-8').split('\0')
            loop_groups['ID'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(16).decode('utf-8').split('\0')
            loop_groups['ref1'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(16).decode('utf-8').split('\0')
            loop_groups['ref2'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(50).decode('utf-8').split('\0')
            loop_groups['memo'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            
            if isRZ:
                loop_groups['beg_t'] = np.fromfile(fid, dtype=np.int64, count=1)[0]
                loop_groups['end_t'] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups['beg_t'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                loop_groups['end_t'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            
            tmp_str = fid.read(100).decode('utf-8').split('\0')
            loop_groups['sgfname1'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(100).decode('utf-8').split('\0')
            loop_groups['sgfname2'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName1'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName2'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName3'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName4'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName5'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName6'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName7'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName8'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName9'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(15).decode('utf-8').split('\0')
            loop_groups['VarName10'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit1'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit2'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit3'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit4'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit5'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit6'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit7'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit8'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit9'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            tmp_str = fid.read(5).decode('utf-8').split('\0')
            loop_groups['VarUnit10'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            
            loop_groups['SampPer_us'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            
            loop_groups['cc_t'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            loop_groups['version'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['postproc'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            tmp_str = fid.read(92).decode('utf-8').split('\0')
            loop_groups['dump'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            
            loop_groups['bid'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['comp'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['x'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['y'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            
            loop_groups['traceCM'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups['tokenCM'] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups['col'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            loop_groups['curcol'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            
            blurb = []
            for i in range(0, 5):
                loop_blurb = dict()
                loop_blurb['type'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb['incid'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb['hide'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb['x'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb['y'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb['manplace'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                tmp_str = fid.read(12).decode('utf-8').split('\0')
                loop_blurb['txt'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
                blurb.append(loop_blurb)
            loop_groups['blurb'] = blurb
            loop_groups['ymax'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups['ymin'] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            tmp_str = fid.read(100).decode('utf-8').split('\0')
            loop_groups['equ'] = [x for x in tmp_str if x and np.size(tmp_str)>1]
            
            groups.append(loop_groups)
        
        for x in range(0, 30):
            if groups[x]['bid'] > 0 and groups[x]['npts'] > 0:
                npts = groups[x]['npts']
                groups[x]['wave'] = np.fromfile(fid, dtype=np.float32, count=npts)
            else:
                groups[x]['wave'] = []
    
    data['RecHead'] = RecHead
    data['groups'] = groups

    return data


def load_data_dp(path, print_skipped=False):
    """Load DP data from .awf files and find power of F1, F2, and 2*F1-F2.
    
    The .awf files are created with BioSigRZ (Tucker-Davis Technologies) during
    experiments. The files are fully loaded, including power spectrum. Then,
    addditional information on primary components and distortion products are
    autodetected.
    
    Parameters
    ----------
    path : string
        A string indicating the parent directory from which to import all DP
        .awf files from all subdirectories.
    print_skipped : {'True', 'False'}, optional
        If set to True, filenames that are skipped for unclear NoiseType or ABR
        are displayed. The default is False.
    
    Returns
    -------
    data : array of dict
        Each element of the array is a dictionary of a single .awf file and its
        associated data.
    """
    
    # Initialize reference arrays
    reference = reference_values()['load_data_dp']
    exp_ref = reference['experimenter']
    supplier = reference['supplier']
    NoiseSPL = reference['NoiseSPL']
    ABRtimes = reference['ABRtimes']
    NoiseTypes = reference['NoiseTypes']
    

    # Loop over all files in specified directory tree and load all DP .awf files
    all_data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.awf') and 'DP' in file:
                full_file = os.path.join(root, file)
                
                # Information on skipped files when requested
                if 'femaleeffect' in file.lower() or 'evi_effect' in root.lower():
                    special = 'female_effect'
                elif 'sham' in root.lower():
                    special = 'sham'
                elif '2hour' in root.lower():
                    special = '2hour'
                else:
                    special = 'none'
                temp_NoiseType = NoiseTypes[np.array([x.lower() in file.lower() for x in NoiseTypes])]
                if temp_NoiseType.size == 0:
                    if print_skipped:
                        print(file + ' has no readable NoiseType; skipping file.')
                    continue
                temp_ABRtime = ABRtimes[np.array([x.lower() in file.lower() for x in ABRtimes])]
                if temp_ABRtime.size == 0:
                    if print_skipped:
                        print(file + ' has no readable ABRtime; skipping file.')
                    continue
                
                # Experiment information and filename
                temp_supplier = supplier[np.array([x.lower() in root.lower() for x in supplier])]
                temp_NoiseSPL = NoiseSPL[np.array([x.lower() in root.lower() for x in NoiseSPL])]
                file_info = {'file_name': file,
                             'supplier': temp_supplier[0],
                             'special': special,
                             'NoiseSPL': temp_NoiseSPL[0],
                             'NoiseType': temp_NoiseType[0],
                             'ABRtime': temp_ABRtime[0]}

                first_char = file.split('_')[1][0].upper()
                if first_char != 'C' and first_char != 'E' and first_char != 'I' and first_char != 'B' and first_char != 'X':
                    tmpID = 'C' + file.split('_')[1] + '_' + file.split('_')[2].split(' ')[0]
                else:
                    tmpID = file.split('_')[1] + '_' + file.split('_')[2].split(' ')[0]
                file_info['ID'] = tmpID

                experimenter = next((key for key, value in exp_ref.items() if tmpID in value), None)
                if experimenter == None:
                    raise LookupError('Undefined experimenter for animal ' + tmpID + '. Add values for file ' + file + ' in function reference_values().')
                file_info['experimenter_ID'] = experimenter
                
                # Actual loading of data
                awfdata = awfread(full_file)
                awfdata['fileInfo'] = file_info
                
                # Detect peaks in spectrum
                awfdata = get_peaks(awfdata)

                # Add to output argument
                all_data.append(awfdata)

    return all_data


def get_peaks(datadict):
    """Get peaks of power spectrum.

    Given the dict-form of a TDT .awf file several parameters characterizing
    F1, F2, and 2*F1-F2 are added to the input dictionary. These data can be
    found in, e.g.,
    
        datadict['groups'][i]['peaks']
    
    IMPORTANT: The data is COPIED from the cursors in the .awf file! The noise
    floor is therefore not calculated.

    Parameters
    ----------
    datadict : dictionary
        A dictionary of an imported TDT .awf file to which the 'peaks'
        information will be added.
    
    Returns
    -------
    datadict : dictionary
        The input dictionary with 'peaks' information added.

    See also detect_peaks.
    """

    noisefloor_threshold = -85. # components need to be 5 dB above noise floor, and 80 dB correction for TDT reasons
    thresholdDP = np.inf
    idx_threshold = np.nan
    # Loop over maximum nr. of traces
    for x in range(0,30):
        group = datadict['groups'][x]
        
        # Only continue if record is not empty, i.e., it has a power spectrum stored in 'wave'
        if any(group['wave']):
            
            # Get cursors
            primary1 = group['cursors'][0]
            primary2 = group['cursors'][1]
            distprod = group['cursors'][2]
            
            # Construct dictionary with useful values
            peaks = {**datadict['fileInfo'],
                     'recn': group['recn'],
                     group['VarName2'][0]: group['vars'][1],
                     group['VarName3'][0]: group['vars'][2],
                     group['VarName4'][0]: group['vars'][3],
                     group['VarName5'][0]: group['vars'][4],
                     group['VarName6'][0]: group['vars'][5],
                     'Intensity1': primary1['val'],
                     'Intensity2': primary2['val'],
                     'IntensityDP': distprod['val'],
                     'NF1_mean': np.nan,
                     'NF2_mean': np.nan,
                     'NFDP_mean': np.nan,
                     'NF1_std': np.nan,
                     'NF2_std': np.nan,
                     'NFDP_std': np.nan,
                     'iPrimary1': int(group['npts'] * group['cursors'][0]['tmar'] / group['dur']),
                     'iPrimary2': int(group['npts'] * group['cursors'][1]['tmar'] / group['dur']),
                     'iDP': int(group['npts'] * group['cursors'][2]['tmar'] / group['dur'])}
            
            # Add entries to dictionary that are calculated from initial values
            peaks['anf1'] = peaks['Intensity1'] > noisefloor_threshold
            peaks['anf2'] = peaks['Intensity2'] > noisefloor_threshold
            peaks['anfDP'] = peaks['IntensityDP'] > noisefloor_threshold
            peaks['thr_dB_anf'] = noisefloor_threshold
            if peaks['anfDP'] and group['vars'][2] < thresholdDP:
                thresholdDP = group['vars'][2]
                idx_threshold = x
        
        else:
            peaks = {}
        
        # Insert peaks into datadict
        datadict['groups'][x]['peaks'] = peaks
    
    # Insert threshold
    for  x in range(0,30):
        if any(datadict['groups'][x]['peaks']):
            if x == idx_threshold:
                datadict['groups'][x]['peaks']['is_threshold'] = True
            else:
                datadict['groups'][x]['peaks']['is_threshold'] = False

    return datadict


def detect_peaks(datadict):
    """Detect peaks of power spectrum.
    
    Given the dict-form of a TDT .awf file several parameters characterizing
    F1, F2, and 2*F1-F2 are added to the input dictionary. These data can be
    found in, e.g.,
    
        datadict['groups'][i]['peaks']
    
    IMPORTANT: The data is CALCULATED from the waveform in the .awf file! This
    includes a noise floor estimate.

    Parameters
    ----------
    datadict : dictionary
        A dictionary of an imported TDT .awf file to which the 'peaks'
        information will be added.
    
    Returns
    -------
    datadict : dictionary
        The input dictionary with 'peaks' information added.

    See also get_peaks.
    """
    
    # Initialize parameters
    reference = reference_values()['detect_peaks']
    nf_neighbors = reference['nf_neighbors']
    comp_neighbors = reference['comp_neighbors']
    thr_dB_anf = reference['thr_dB_anf']
    iNoiseFloor = np.concatenate([np.arange(-(nf_neighbors+1), -1), np.arange(2, nf_neighbors+2)]) # indices for noisefloor estimate
    
    thresholdDP = np.inf
    idx_threshold = np.nan
    # Loop over maximum nr. of traces
    for x in range(0,30):
        group = datadict['groups'][x]
        
        # Only continue if record is not empty, i.e., it has a power spectrum stored in 'wave'
        if any(group['wave']):
            
            # Find indices of primary components and distortion product 2*F1-F2
            iPrimary1 = int(round(group['npts'] * group['vars'][4] / group['dur'])) + np.arange(-comp_neighbors, comp_neighbors+1)
            iPrimary1 = iPrimary1[np.argmax(group['wave'][iPrimary1])]
            iPrimary2 = int(round(group['npts'] * group['vars'][5] / group['dur'])) + np.arange(-comp_neighbors, comp_neighbors+1)
            iPrimary2 = iPrimary2[np.argmax(group['wave'][iPrimary2])]
            iDP = int(round(group['npts'] * (2*group['vars'][4] - group['vars'][5]) / group['dur'])) + np.arange(-comp_neighbors, comp_neighbors+1)
            iDP = iDP[np.argmax(group['wave'][iDP])]
            
            # Noise floor range hard coded, make sure there is no overlap with primaries and DPs
            if iPrimary2 - iPrimary1 < 10:
                print(datadict['fileInfo'])
                raise ValueError('Primary components are too closely spaced for proper noise floor detection.\n F1_index: '
                                 + str(iPrimary1) + '\n F2_index ' + str(iPrimary2))
            
            # Construct dictionary with useful values
            peaks = {**datadict['fileInfo'],
                     'recn': group['recn'],
                     group['VarName2'][0]: group['vars'][1],
                     group['VarName3'][0]: group['vars'][2],
                     group['VarName4'][0]: group['vars'][3],
                     group['VarName5'][0]: group['vars'][4],
                     group['VarName6'][0]: group['vars'][5],
                     'Intensity1': group['wave'][iPrimary1],
                     'Intensity2': group['wave'][iPrimary2],
                     'IntensityDP': group['wave'][iDP],
                     'NF1_mean': np.mean(group['wave'][iNoiseFloor+iPrimary1]),
                     'NF2_mean': np.mean(group['wave'][iNoiseFloor+iPrimary2]),
                     'NFDP_mean': np.mean(group['wave'][iNoiseFloor+iDP]),
                     'NF1_std': np.std(group['wave'][iNoiseFloor+iPrimary1]),
                     'NF2_std': np.std(group['wave'][iNoiseFloor+iPrimary2]),
                     'NFDP_std': np.std(group['wave'][iNoiseFloor+iDP]),
                     'iPrimary1': iPrimary1,
                     'iPrimary2': iPrimary2,
                     'iDP': iDP}
            
            # Add entries to dictionary that are calculated from initial values
            peaks['anf1'] = peaks['Intensity1'] > peaks['NF1_mean'] + thr_dB_anf
            peaks['anf2'] = peaks['Intensity2'] > peaks['NF2_mean'] + thr_dB_anf
            peaks['anfDP'] = peaks['IntensityDP'] > peaks['NFDP_mean'] + thr_dB_anf
            peaks['thr_dB_anf'] = thr_dB_anf
            if peaks['anfDP'] and group['vars'][2] < thresholdDP:
                thresholdDP = group['vars'][2]
                idx_threshold = x
        
        else:
            peaks = {}
        
        # Insert peaks into datadict
        datadict['groups'][x]['peaks'] = peaks
    
    # Insert threshold
    for  x in range(0,30):
        if any(datadict['groups'][x]['peaks']):
            if x == idx_threshold:
                datadict['groups'][x]['peaks']['is_threshold'] = True
            else:
                datadict['groups'][x]['peaks']['is_threshold'] = False

    return datadict


def clean_data(all_data):
    """Check data for errors and clean up known errors."""

    # Check for repetition of filenames
    check_filenames(all_data)
    
    # Clean up data
    all_data = remove_empty_groups(all_data)

    # Remove outliers
    all_data = remove_outliers(all_data)

    # Replace DNT/ZT3 and NNT/ZT15
    all_data = replace_nnt_dnt(all_data)

    return all_data


def check_filenames(all_data):
    """Check loaded data for filename conflicts and print filename if conflict found."""
    
    n_filenames = np.unique([d['fileInfo']['file_name'] for d in all_data]).size
    if n_filenames != np.size(all_data):
        fn_list = [d['fileInfo']['file_name'] for d in all_data]
        [print(fn) for fn in fn_list if fn_list.count(fn)>1]
        raise ValueError('The number of unique filenames and loaded .awf files is not the same.')
    return


def remove_empty_groups(all_data):
    """Removes first entries of 'groups' if empty.
    
    If these groups are left in, looking at the first element of the 'peaks' field fails in later steps.
    There is at least one file ('160524_I-1655_3_24hourpost_8kHz_DP_NNT.awf') with this characteristic.
    """
    
    count = [count for count, d in enumerate(all_data) if not d['groups'][0].get('peaks')]
    for ct in count:
        all_data[ct]['groups'] = [grp for grp in all_data[ct]['groups'] if grp.get('peaks')]
    return all_data


def replace_nnt_dnt(all_data):
    for d in all_data:
        for g in d['groups']:
            nt = g['peaks'].get('NoiseType')
            if  nt == 'DNT':
                g['peaks'].update({'NoiseType': 'ZT3'})
            elif nt == 'NNT':
                g['peaks'].update({'NoiseType': 'ZT15'})
            elif nt == '5AM':
                g['peaks'].update({'NoiseType': 'ZT23'})
            elif nt == '7AM':
                g['peaks'].update({'NoiseType': 'ZT1'})
            elif nt == '5PM':
                g['peaks'].update({'NoiseType': 'ZT11'})
            elif nt == '7PM':
                g['peaks'].update({'NoiseType': 'ZT13'})
    return all_data


def remove_outliers(all_data):
    """Removes DPOAE outliers based on intensity ratio of F1:F2 and plots the result"""
    
    # Define cutoff and print filenames that have IntensityF1-IntensityF2 beyond that cutoff
    spl_difference_primaries = 10
    spl_difference_allowed = 20
    FN_remove = [d['fileInfo']['file_name'] for d in all_data if (d['groups'][0]['peaks']['Intensity1'] - d['groups'][0]['peaks']['Intensity2']) < spl_difference_primaries - spl_difference_allowed or (d['groups'][0]['peaks']['Intensity1'] - d['groups'][0]['peaks']['Intensity2']) > spl_difference_primaries + spl_difference_allowed]
    FN_keep = [d['fileInfo']['file_name'] for d in all_data if (d['groups'][0]['peaks']['Intensity1'] - d['groups'][0]['peaks']['Intensity2']) >= spl_difference_primaries - spl_difference_allowed and (d['groups'][0]['peaks']['Intensity1'] - d['groups'][0]['peaks']['Intensity2']) <= spl_difference_primaries + spl_difference_allowed]
    print('Removing data for the following files:')
    [print(' ' + fn) for fn in FN_remove]
    
    # Plot before removing outliers
    plt.figure()
    plt.subplot(2, 1, 1)
    n, _, _ = plt.hist([int(d['groups'][0]['peaks']['Intensity1'] - d['groups'][0]['peaks']['Intensity2']) for d in all_data], bins=32)
    ylim = [0, max(n)]
    plt.plot(np.ones(2)*(spl_difference_primaries - spl_difference_allowed), ylim, 'r-')
    plt.plot(np.ones(2)*(spl_difference_primaries + spl_difference_allowed), ylim, 'r-')
    plt.title('Before (top) and after (bottom) outlier removal')
    plt.ylabel('Count')
    
    # Remove outliers
    all_data = select_data_dp(all_data, file_name=FN_keep)
    
    # Plot after removing outliers
    plt.subplot(2, 1, 2)
    n, _, _ = plt.hist([int(d['groups'][0]['peaks']['Intensity1'] - d['groups'][0]['peaks']['Intensity2']) for d in all_data], bins=32)
    ylim = [0, max(n)]
    plt.plot(np.ones(2)*(spl_difference_primaries - spl_difference_allowed), ylim, 'r-')
    plt.plot(np.ones(2)*(spl_difference_primaries + spl_difference_allowed), ylim, 'r-')
    plt.xlabel('F1-F2 (dB)')
    plt.ylabel('Count')
    plt.show()
    
    return all_data


def select_data_dp(datadict, **kwargs):
    """Make sub-selection of DP array of dictionaries.
    
    Parameters
    ----------
    datadict : array of dictionaries
        An array of dictionaries with DP .awf file data that needs
        to be narrowed down.
    **kwargs : 
        Any number of key=value to use for selecting. value can be
        an interable. The order of arguments matters, as the first
        key:value pair will evaluate first, each time narrowing
        down the input dataframe further. key must be a key of the
        dictionary
        
            datadict[i]['groups'][i]['peaks']
    
    Returns
    -------
    datadict : array of dictionaries
        Selection from original array of dictionaries based on parsed
        values.
    """
    
    if kwargs is not None:
        for key, value in kwargs.items():
            if value == str:
                datadict = [d for d in datadict if d['groups'][0]['peaks'][key][0].lower() == value.lower()]
            elif type(value) != list:
                datadict = [d for d in datadict if d['groups'][0]['peaks'][key] == value]
            else:
                if value[0] == str:
                    datadict = [d for d in datadict if d['groups'][0]['peaks'][key][0].lower() in [el.lower() for el in value]]
                else:
                    datadict = [d for d in datadict if d['groups'][0]['peaks'][key] in value]
    
    return datadict


def plot_spectrum_dp(datadict, ylim=[]):
    """Plot DPOAE  spectrum.
    
    Parameters
    ----------
    datadict : dict
        A dictionary created with awfread and containing DPOAE data. For every
        trace found in the original TDT .awf file a new subplot shows the power
        spectrum. Primaries and 2*F1-F2 are labeled.
    ylim : [numeric, numeric], optional
        Use to set ylim for all subplots. The default is [], meaning that
        matplotlib.pyplot will use it's own default.
    """
    
    # Determine number of subplots and start plotting
    groups = datadict['groups']
    Nwave = sum([any(g['wave']) for g in groups])
    fig = plt.figure(figsize=[9, 2*Nwave])
    for i in range(0, Nwave):
        grp = groups[i]
        
        peaks = grp['peaks']
        X = np.linspace(0, grp['dur'], grp['npts']) / 1e3
        
        plt.subplot(Nwave, 1, i+1)
        plt.plot(X, grp['wave'] + 80)
        
        # Create labels
        if any(ylim):
            plt.ylim(ylim)
        if peaks['anf1']:
            label_F1 = 'F1: ' + str(int(peaks['Intensity1'] + 80)) + ' dB SPL'
        else:
            label_F1 = 'F1 in noise floor'
        if peaks['anf2']:
            label_F2 = 'F2: ' + str(int(peaks['Intensity2'] + 80)) + ' dB SPL'
        else:
            label_F2 = 'F2 in noise floor'
        if peaks['anfDP']:
            label_DP = '2*F1-F2: ' + str(int(peaks['IntensityDP'] + 80)) + ' dB SPL'
        else:
            label_DP = '2*F1-F2 in noise floor'
        
        # Plotting
        plt.plot(X[peaks['iPrimary1']], peaks['Intensity1'] + 80, 'g*', label=label_F1)
        plt.plot(X[peaks['iPrimary2']], peaks['Intensity2'] + 80, 'c*', label=label_F2)
        if peaks['anfDP']:
            plt.plot(X[peaks['iDP']], peaks['IntensityDP'] + 80, 'r*', label=label_DP)
        else:
            plt.plot(X[peaks['iDP']], peaks['IntensityDP'] + 80, 'r*', label=label_DP, markerfacecolor='w', markeredgewidth=0.4)
        
        # Set labels, titles
        plt.legend()
        if i == 0:
            plt.title('\n\n\n' + 'F1 = ' + str(grp['vars'][2]) + ' dB SPL')
        elif i == np.floor(Nwave/2):
            plt.ylabel('Power (dB SPL)', fontsize=16)
        elif i == Nwave - 1:
            plt.xlabel('Frequency (kHz)', fontsize=16)
            plt.title('F1 = ' + str(grp['vars'][2]) + ' dB SPL')
        else:
            plt.title('F1 = ' + str(grp['vars'][2]) + ' dB SPL')
    
    # Set figure layout, figure title
    fig.tight_layout()
    fig.suptitle(datadict['fileInfo']['supplier'] + ', '
                 + datadict['fileInfo']['NoiseSPL'] + ', '
                 + datadict['fileInfo']['NoiseType'] + ', '
                 + datadict['fileInfo']['ABRtime'] + ', '
                 + datadict['fileInfo']['ID']
                 + '\nF1 = ' + str(groups[0]['vars'][4]/1e3) + ' kHz, '
                 + 'F2 = ' + str(groups[0]['vars'][5]/1e3) + ' kHz',
                 fontsize=16, verticalalignment='bottom')
    
    return


def dataframe_subset(df, **kwargs):
    """Make sub-selection of DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Any pandas.Dataframe to make sub-selection from.
    **kwargs : 
        Any number of 'fieldname'=value to use for selecting. value can be
        an interable. The order of arguments matters, as the first
        keyword:value pair will evaluate first, each time narrowing
        down the input dataframe further.
    
    Returns
    -------
    df : pandas.DataFrame
        Selection from original dataframe based on parsed values.
    """

    if kwargs is not None:
        for key, value in kwargs.items():
            if np.size(value) == 1:
                df = df[df[key] == value]
            else:
                df = df.iloc[[v in value for v in df[key]]]
    return df


def peaks_df(datadict):
    """Get peaks of DPOAE data from dictionary
    
    The function detect_peaks adds DPOAE peak information to each .awf file dictionary
    loaded with load_data_dp. The function peaks_df retrieves this information and returns
    it in a pandas dataframe for further use.
    
    Parameters
    ----------
    datadict : dictionary or list of dictionaries
        A (list of) dictionaries of (an) imported TDT .awf file(s) from which to retrieve
        'peaks' information will be added.
    
    Returns
    -------
    df : pandas.DataFrame
        The input dictionary with 'peaks' information added.
    """
    
    if type(datadict) == dict:
        datadict = [datadict]
    
    df_array = [pd.concat([pd.Series(grp['peaks']) for grp in d['groups'] if grp.get('peaks')], axis=1) for d in datadict]
    df = pd.concat([pks for pks in df_array], axis=1).transpose().reset_index(drop=True)
    
    df = df.astype({'file_name': object, 'supplier': 'category', 'special': 'category', 'NoiseSPL': 'category',
                    'NoiseType': 'category', 'ABRtime': 'category', 'ID': 'category', 'experimenter_ID': 'category',
                    'recn': int, 'AudFreq': float, 'L1': float, 'L2': float, 'F1': float, 'F2': float,
                    'Intensity1': float, 'Intensity2': float, 'IntensityDP': float, 'NF1_mean': float,
                    'NF2_mean': float, 'NFDP_mean': float, 'NF1_std': float, 'NF2_std': float, 'NFDP_std': float,
                    'iPrimary1': int, 'iPrimary2': int, 'iDP': int, 'anf1': bool, 'anf2': bool, 'anfDP': bool,
                    'thr_dB_anf': float, 'is_threshold': bool})
    
    return df


def plot_io_dp(datadict, xlim=[], ylim=[]):
    """Plot the input-output relation for DPOAE data
    
    Parameters
    ----------
    datadict : dict
        A dictionary created with awfread and containing DPOAE data. Every 'group'
        in the dictionary will correspond to a single data point.
    xlim : [numeric, numeric], optional
        Use to set xlim for all subplots. The default is [], meaning autodetection
        by matplotlib.pyplot.
    ylim : [numeric, numeric], optional
        Use to set ylim for all subplots. The default is [], meaning autodetection
        by matplotlib.pyplot.
    """

    labelcolor = reference_values()['colormap_alternate']

    reference = reference_values()['plot_io_dp']
    jitter_factor = reference['jitter_factor']
    noisefloor = reference['noisefloor']
    
    # Create dataframe with all DP information
    df = peaks_df(datadict)
    
    # Adjust power spectrum intensities by 80 dB (TDT specified)
    df[['Intensity1', 'Intensity2', 'IntensityDP', 'NF1_mean', 'NF2_mean', 'NFDP_mean']] += 80
    
    # Select rows only where DP is above noise floor
    df = df[df.apply(lambda d: d['IntensityDP'] >= noisefloor.get(d['AudFreq']), axis=1)]
    
    # Set default values
    if not xlim:
        xlim = [10*np.floor(min(df['L1'])/10) - 2,
                10*np.ceil(max(df['L1'])/10) + 2]
    if not ylim:
        ylim = [10*np.floor(min(df['IntensityDP'])/10),
                10*np.ceil(max(df['IntensityDP'])/10)]
    
    # Set figure shape
    uF2s = np.sort(df['F2'].unique())
    uABRtimes = df['ABRtime'].unique()
    nRows = len(uF2s)
    nCols = sum([abrt != 'baseline' for abrt in uABRtimes])
    fig = plt.figure(figsize=[4.8*nCols, 4.*nRows])
    
    # Figure title
    u_supp = df['supplier'].unique()
    u_nois = df['NoiseSPL'].unique()
    u_expe = df['experimenter_ID'].unique()
    if len(u_supp) > 1:
        supp_ID = '+'.join(u_supp)
    else:
        supp_ID = u_supp[0]
    if len(u_nois) > 1:
        nois_ID = '+'.join(u_nois)
    else:
        nois_ID = u_nois[0]
    if len(u_expe) > 1:
        expe_ID = '+'.join(u_expe)
    else:
        expe_ID = u_expe[0]
    fig.suptitle(supp_ID + ', ' + df['special'].iloc[0] + ', ' + str(nois_ID)
                 + ' SPL, experimenter ' + expe_ID, fontsize=16)
    
    # Construct {F2: F1} reference dict
    F2F1ref = df[['F1', 'F2']]
    F2F1ref = F2F1ref[~F2F1ref.duplicated()].to_dict('records')
    F2F1ref = {d['F2']: d['F1'] for d in F2F1ref}
    
    # Loop over different F2s and ABRtimes (baseline, 24h, 2w) and plot 24h and 2w only
    for count_F2, F2 in enumerate(uF2s):
        for count_ABRt, ABRt in enumerate(filter(lambda s: s!='baseline', uABRtimes)):
            # Narrow down dataframe to have single frequency, single ABRtime
            df_small = dataframe_subset(df, AudFreq=F2, ABRtime=['baseline', ABRt])
            
            # With a 3rd order polynomial model, we need data for 3 or more L1 values. Otherwise
            # fit_lme_dp will throw an error.
            df_keep_noisetype = df_small[['L1', 'NoiseType', 'ID']].groupby(['L1', 'NoiseType']).agg('count') > 0
            df_keep_noisetype = df_keep_noisetype.groupby('NoiseType').agg(sum) >= 4
            df_small_lme = df_small.merge(df_keep_noisetype.reset_index().astype({'NoiseType': 'category'}), on='NoiseType', suffixes=['', '_keep'])
            df_small_lme = dataframe_subset(df_small_lme, ID_keep=True)
            
            # Plot data seperately for each noise type, e.g., 'baseline', 'NNT', and 'DNT'
            ax = plt.subplot(nRows, nCols, 1+(count_F2*nCols) + count_ABRt)
            for NT in df['NoiseType'].unique()[::-1]:
                # Narrow down data for current NoiseType
                df_plot = dataframe_subset(df_small, NoiseType=NT)
                
                # Plot data
                jittered_level = np.add(df_plot['L1'], np.random.uniform(-jitter_factor, jitter_factor, len(df_plot['L1'])))
                plt.plot(jittered_level, df_plot['IntensityDP'].values, marker='o', linestyle='',
                         markersize=3, markeredgewidth=.3, color=labelcolor[NT], label=NT)
                
            plt.xlim(xlim)
            plt.ylim(ylim)
            
            if count_ABRt % nCols == 0:
                plt.ylabel('DP intensity (dB SPL)', fontsize=16)
            if count_F2 == 0:
                handles, labels = ax.get_legend_handles_labels()
                sort_index = np.argsort(labels)[::-1]
                handles = np.array(handles)[sort_index]
                labels = np.array(labels)[sort_index]
                plt.legend(handles, labels, loc='center left')
                plt.title('{first}\nF1={second:.1f} kHz, F2={third:.1f} kHz, noisefloor<{fourth} dB SPL'.format(first=ABRt, second=F2F1ref[F2]/1e3, third=F2/1e3, fourth=noisefloor.get(F2)))
            else:
                plt.title('F1={first:.1f} kHz, F2={second:.1f} kHz, noisefloor<{fourth} dB SPL'.format(first=F2F1ref[F2]/1e3, second=F2/1e3, fourth=noisefloor.get(F2)))
            if count_F2 == len(uF2s) - 1:
                plt.xlabel('Stimulus intensity F2 (dB SPL)', fontsize=16)
    
    plt.show()
    
    return


def plot_area_under_curve_dp(datadict):
    """Create strip plot showing distribution of area under curve
    
    When parsing a list of dictionaries from .awf files this function does the
    following:
    
        1) Convert DP intensity and stimulus intensity from SPL to power in Pa
        2) Find the mean of two subsequent DP powers and the spacing between two
        neighboring SPL powers, multiply them, and sum them, giving the area under
        a DP input-output curve on linear scale.
        3) Take the square root of this area, normalize it to 20e-6 Pa, and take
        the 10*log10 of this number, converting back to dB SPL.
        4) Plot a strip plot for these numbers, excluding the values that
        are zero before log-conversion and -inf after conversion. Make a separate plot
        for each Frequency, and accept groups of noisetypes within each frequency.
        5) Print information if there were zero values, meaning that the bar-and-whisker
        plots are underestimating the variation of the data.
        6) Do a Kruskal-Wallis test (nonparametric) to find differences within frequency
        groups and if a difference is found, do a post hoc analysis between noisetypes
        within a frequency group.
    
    Parameters
    ----------
    datadict : list of dictionaries
        A (list of) dictionaries created with load_data_dp.
    """
    
    colormap = reference_values()['colormap_alternate']

    reference = reference_values()['plot_area_under_curve_dp']
    noisefloor = reference['noisefloor']
    alpha = reference['alpha']

    # Create dataframe with all DP information
    df = peaks_df(datadict)
    
    # Adjust power spectrum intensities by 80 dB (TDT specified)
    df[['Intensity1', 'Intensity2', 'IntensityDP', 'NF1_mean', 'NF2_mean', 'NFDP_mean']] += 80
    df['AudFreq'] /= 1e3

    # Create power column
    df['PowerDP'] = 10**(df['IntensityDP'] / 10) * 20e-6
    df['PowerL1'] = 10**(df['L1'] / 10) * 20e-6
    
    # Set/remove noise floor values
    _below_nf = df.apply(lambda d: d['IntensityDP'] < noisefloor.get(d['AudFreq']*1e3), axis=1)
    df.loc[_below_nf, 'PowerDP'] = 0

    # Create axes
    fig = plt.figure()
    plt.axes()

    # Figure title
    u_supp = df['supplier'].unique()
    u_nois = df['NoiseSPL'].unique()
    u_expe = df['experimenter_ID'].unique()
    if len(u_supp) > 1:
        supp_ID = '+'.join(u_supp)
    else:
        supp_ID = u_supp[0]
    if len(u_nois) > 1:
        nois_ID = '+'.join(u_nois)
    else:
        nois_ID = u_nois[0]
    if len(u_expe) > 1:
        expe_ID = '+'.join(u_expe)
    else:
        expe_ID = u_expe[0]
    fig.suptitle(supp_ID + ', ' + df['special'].iloc[0] + ', ' + str(nois_ID)
                 + ' SPL, experimenter ' + expe_ID, fontsize=14)
    
    df.set_index(['NoiseType', 'AudFreq', 'ID', 'file_name'], inplace=True)
    
    def _area_under_curve(dataframe):
        lvl = 3
        file_names = dataframe.index.get_level_values(lvl).unique()
        df_area = pd.DataFrame()
        for file_name in file_names:
            _dd = dataframe.xs(file_name, level=lvl)
            _delta_spl = abs(np.diff(_dd['PowerL1']))
            _intensity_power = _dd['PowerDP'].values
            _mean_intensity_power = [np.mean(p) for p in zip(_intensity_power[:-1], _intensity_power[1:])]
            _area_parts = np.multiply(_mean_intensity_power, _delta_spl)
            _area = sum(_area_parts)
            df_area = df_area.append(pd.Series({'area': _area, 'NoiseType': _dd.iloc[0].name[0],
                                                'AudFreq': _dd.iloc[0].name[1]}),
                                     ignore_index=True)
        return df_area
    
    areas_df = _area_under_curve(df)
    areas_df = areas_df.sort_values(by=['NoiseType'], ascending=False)
    areas_df['area'] = 10 * np.log10(np.sqrt(areas_df['area'])/20e-6)
    
    for index, row in areas_df.groupby(['NoiseType', 'AudFreq']).agg('min').iterrows():
        if row['area'] == -np.inf:
            print('Area == 0 found for NoiseType {nt} and frequency {f}'.format(nt=index[0], f=index[1]))
    
    areas_df_nonzero = areas_df.query('area > 0')
    ax = sns.stripplot(x='AudFreq', y='area', hue='NoiseType', dodge=True, size=3, data=areas_df_nonzero, palette=colormap)
    
    sns.pointplot(x='AudFreq', y='area', hue='NoiseType', dodge=True, markers='', join=True, ci=None, data=areas_df_nonzero, palette=colormap)
    handles, labels = ax.get_legend_handles_labels()
    half_length = int(len(handles)/2)
    plt.legend(handles[:half_length], labels[:half_length])

    plt.xlabel('Frequency F2 (kHz)')
    plt.ylabel('sqrt(Area) (dB SPL)')
    plt.show()
    
    print('alpha = ' + str(alpha) + '\n')
    for AudFreq in np.sort(areas_df.AudFreq.unique()):
        _small_df = areas_df.query('AudFreq == {f}'.format(f=AudFreq))
        
        print('Frequency: ' + str(AudFreq))
        
        kw = stats.kruskal(*(_small_df['area'][_small_df['NoiseType'] == NT]
                             for NT in _small_df.NoiseType.unique()))
        print('  Overall Kruskal-Wallis:')
        if kw.pvalue < alpha:
            print('   *Significant difference (p={:.4f})'.format(kw.pvalue))
            print('  Post hoc Mann-Whitney U:')
            for nt_comb in itertools.combinations(_small_df.NoiseType.unique(), 2):
                sqrt_areas = (_small_df['area'][_small_df['NoiseType'] == NT] for NT in nt_comb)
                if len(np.unique([item for sublist in sqrt_areas for item in sublist])) < 2:
                    print('    All values are equal. No significant effect for groups {grp0}-{grp1})'.format(grp0=nt_comb[0], grp1=nt_comb[1]))
                else:
                    mwu = stats.mannwhitneyu(*(_small_df['area'][_small_df['NoiseType'] == NT]
                                               for NT in nt_comb), alternative='two-sided')
                    if mwu.pvalue < alpha:
                        print('   *Significant effect for groups {grp0}-{grp1} (p={pval:.4f})'.format(grp0=nt_comb[0], grp1=nt_comb[1], pval=mwu.pvalue))
                    else:
                        print('    No significant effect for groups {grp0}-{grp1} (p={pval:.4f})'.format(grp0=nt_comb[0], grp1=nt_comb[1], pval=mwu.pvalue))
                
        else:
            print('    No significant difference (p={:.4f})'.format(kw.pvalue))
    return


def plot_threshold_dp(datadict, ylim, shift=False):
    """Create strip plot showing distribution DPOAE threshold
    
    When parsing a list of dictionaries from .awf files this function does the
    following:
    
        1) Find the first stimulus intensity that evokes a DP of -5 dB SPL or more and if
        none are found, use 85 dB SPL as threshold.
        2) Plot a strip plot for these thresholds. Separate each Frequency, and
        accept groups of noisetypes within each frequency.
        3) Do a Kruskal-Wallis test (nonparametric) to find differences within frequency
        groups and if a difference is found, do a post hoc analysis between noisetypes
        within a frequency group.
    
    If input argument "shift=True" the threshold shift between different recording times
    (baseline, 24h, 2w) is plotted and tested.
    
    Parameters
    ----------
    datadict : list of dictionaries
        A (list of) dictionaries created with load_data_dp.
    """

    colormap = reference_values()['colormap_alternate']

    reference = reference_values()['plot_threshold_dp']
    noisefloor = reference['noisefloor']
    alpha = reference['alpha']
    
    # Create dataframe with all DP information
    df = peaks_df(datadict)
    
    # Adjust power spectrum intensities by 80 dB (TDT specified)
    df[['Intensity1', 'Intensity2', 'IntensityDP', 'NF1_mean', 'NF2_mean', 'NFDP_mean']] += 80
    df['AudFreq'] /= 1e3

    # Create axes
    fig = plt.figure()

    # Figure title
    u_supp = df['supplier'].unique()
    u_nois = df['NoiseSPL'].unique()
    u_expe = df['experimenter_ID'].unique()
    if len(u_supp) > 1:
        supp_ID = '+'.join(u_supp)
    else:
        supp_ID = u_supp[0]
    if len(u_nois) > 1:
        nois_ID = '+'.join(u_nois)
    else:
        nois_ID = u_nois[0]
    if len(u_expe) > 1:
        expe_ID = '+'.join(u_expe)
    else:
        expe_ID = u_expe[0]
    fig.suptitle(supp_ID + ', ' + df['special'].iloc[0] + ', ' + str(nois_ID)
                 + ' SPL, experimenter ' + expe_ID, fontsize=14)
    
    def _threshold(dataframe, noisefl):
        """Calculate threshold"""
        dataframe = dataframe.set_index('file_name')
        df_threshold = pd.DataFrame()
        for file_name in dataframe.index.unique():
            _dd = dataframe.xs(file_name)
            _nfl = noisefl.get(_dd['AudFreq'].unique()[0]*1e3)
            _dd_anf = _dd.query('IntensityDP > {}'.format(_nfl))
            if len(_dd_anf) == 0:
                _noisetype = _dd['NoiseType'].iloc[0]
                _audfreq = _dd['AudFreq'].iloc[0]
                _threshold = 85
            elif type(_dd_anf['NoiseType']) != np.str_:
                _noisetype = _dd_anf['NoiseType'].iloc[0]
                _audfreq = _dd_anf['AudFreq'].iloc[0]
                _threshold = _dd_anf['L1'].min()
            else:
                _noisetype = _dd_anf['NoiseType']
                _audfreq = _dd_anf['AudFreq']
                _threshold = _dd_anf['L1']
            df_threshold = df_threshold.append(pd.Series({'threshold': _threshold,
                                                          'NoiseType': _noisetype,
                                                          'AudFreq': _audfreq}).append(_dd[['supplier', 'ID', 'ABRtime',
                                                                                            'special', 'NoiseSPL',
                                                                                            'experimenter_ID']].iloc[0]),
                                               ignore_index=True)
        return df_threshold
        
    df = _threshold(df, noisefloor)
    
    # Experiments 'C42_3', 'C46095_1', and 'I-1658_16' have a freq in their
    # filename that does not match the freq used according to the .awf file.
    # These situations are resolved by dropping later occurrences.
    df = df.drop_duplicates(subset=['ABRtime', 'AudFreq', 'ID'])

    # Calculate shift when queried
    uABRtimes = sorted(df['ABRtime'].unique()) # baseline is now last, helpful for shift=True
    colnames_join = ['supplier', 'ID', 'special', 'NoiseSPL', 'experimenter_ID', 'AudFreq']
    if shift:
        if np.size(uABRtimes) == 1:
            raise ValueError('Single ABRtime in dataframe. Increase number of ABRtimes to two or try using plot_threshold with \'shift=False\'.')
        if np.size(uABRtimes) > 2:
            raise ValueError('More than two ABRtimes in dataframe. Decrease number of ABRtimes to two.')
        df_reference = df[df['ABRtime'] == uABRtimes[1]].dropna() # usually 'baseline'
        df_other = df[df['ABRtime'] == uABRtimes[0]].dropna()
        df = df_other.set_index(colnames_join).join(df_reference.set_index(colnames_join),
                                how='outer', lsuffix='_other', rsuffix='_reference').dropna()
        df['threshold'] = df['threshold_other'] - df['threshold_reference']
        df['NoiseType'] = df.apply(lambda row: row['NoiseType_other'] + '-' + row['NoiseType_reference'], axis=1)
        
        new_colormap = {}
        for NTs in df['NoiseType'].unique():
            for NTs_other in df['NoiseType_other'].unique():
                if NTs_other in NTs:
                    new_colormap[NTs] = colormap[NTs_other]
        colormap = new_colormap
    
    df = df.dropna().reset_index()

    # Plotting
    jitter_factor = 1
    df['threshold'] = np.add(df['threshold'], np.random.uniform(-jitter_factor, jitter_factor, len(df['threshold'])))

    ax = plt.axes()
    df['ID'] = df['ID'].astype(str)
    df['AudFreq_cat'] = df['AudFreq'].astype('category')
    df['AudFreq_cat'] = df['AudFreq_cat'].cat.rename_categories({8.: '8', 12.: '12', 16.: '16', 24.: '24', 32.: '32'})
    pale_colormap = {k:([.7, .7, .7] if k=='baseline' else v) for k,v in colormap.items()}
    pale_colormap = {k:([1, .6, .6] if v=='r' else v) for k,v in pale_colormap.items()}
    pale_colormap = {k:([.6, .6, 1] if v=='b' else v) for k,v in pale_colormap.items()}
    
    sns.lineplot(ax=ax, x='AudFreq_cat', y='threshold', hue='NoiseType', units='ID', data=df, estimator=None, linewidth=.5, mew=.5, mfc='w', marker='o', ms=4, palette=pale_colormap, legend=False)

    clrs = [plt.getp(line, 'color') + [1.] for line in ax.lines]
    for c, v in enumerate(ax.lines):
        plt.setp(v, markeredgecolor=clrs[c])
    nlineslight = len(ax.lines)
    
    mean_sem = df[['NoiseType', 'AudFreq', 'threshold']].groupby(by=['NoiseType', 'AudFreq']).agg(['mean', 'sem']).reset_index()
    for NT in sorted(mean_sem['NoiseType'].unique(), reverse=True):
        small_df = select_data_df(mean_sem, NoiseType=NT)
        ax.errorbar(np.arange(0, len(small_df)), small_df['threshold']['mean'], yerr=small_df['threshold']['sem'], color=colormap[NT], linewidth=1.5, capsize=4, marker='o', markersize=6, zorder=nlineslight+1, label=NT)

    plt.ylim(ylim)
    
    ax.legend()
    
    plt.xlabel('Stimulus frequency F2 (kHz)', fontsize=14)
    if shift:
        plt.ylabel('Shift of L1 at threshold (dB)', fontsize=14)
    else:
        plt.ylabel('L1 at threshold (dB SPL)', fontsize=14)
    plt.show()
    
    # Statistical tests
    print('alpha = ' + str(alpha) + '\n')
    for AudFreq in np.sort(df.AudFreq.unique()):
        _small_df = df.query('AudFreq == {f}'.format(f=AudFreq))
        print('Frequency: ' + str(AudFreq))
        kw = stats.kruskal(*(_small_df['threshold'][_small_df['NoiseType'] == NT]
                             for NT in _small_df.NoiseType.unique()))
        print('  Overall Kruskal-Wallis:')
        if kw.pvalue < alpha:
            print('   *Significant difference (p={:.4f})'.format(kw.pvalue))
            if shift:
                continue
            print('  Post hoc Mann-Whitney U:')
            for nt_comb in itertools.combinations(_small_df.NoiseType.unique(), 2):
                mwu = stats.mannwhitneyu(*(_small_df['threshold'][_small_df['NoiseType'] == NT]
                                           for NT in nt_comb), alternative='two-sided')
                if mwu.pvalue < alpha:
                    print('   *Significant effect for groups {grp0}-{grp1} (p={pval:.4f})'.format(grp0=nt_comb[0], grp1=nt_comb[1], pval=mwu.pvalue))
                else:
                    print('    No significant effect for groups {grp0}-{grp1} (p={pval:.4f})'.format(grp0=nt_comb[0], grp1=nt_comb[1], pval=mwu.pvalue))
                
        else:
            print('    No significant difference (p={:.4f})'.format(kw.pvalue))
    
    return


def select_data_df(df, **kwargs):
    """Make sub-selection of DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Any pandas.Dataframe to make sub-selection from.
    **kwargs : 
        Any number of 'fieldname'=value to use for selecting. value can be
        an interable. The order of arguments matters, as the first
        keyword:value pair will evaluate first, each time narrowing
        down the input dataframe further.
    
    Returns
    -------
    df : pandas.DataFrame
        Selection from original dataframe based on parsed values.
    """

    if kwargs is not None:
        for key, value in kwargs.items():
            if np.size(value) == 1:
                df = df[df[key] == value]
            else:
                df = df.iloc[[v in value for v in df[key]]]
    return df