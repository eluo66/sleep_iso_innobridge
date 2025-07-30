#TODO load the pickle file
#TODO plot (freq_iso, spec_iso)

import pickle
import matplotlib.pyplot as plt
import numpy as np


sid = '111189359'

with open(f'result/ISO_result_{sid}.pickle', 'rb') as f:
    data = pickle.load(f)

freq_iso = np.array(data[1])
spec_iso = np.array(data[0]['sigma_oof'][1])

#plotting: 
plt.plot(freq_iso, spec_iso)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral Power')
plt.title('Infraslow Oscillation Spectrum')
plt.grid(True)
#plt.show()
plt.savefig(f'figure_iso_spectrum/iso_{sid}.png', bbox_inches='tight')