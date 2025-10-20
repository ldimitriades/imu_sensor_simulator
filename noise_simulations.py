import numpy as np   
import matplotlib.pyplot as plt


#MODELING ALLAN VARIANCE AND NOISES
fs = 100.0
dt = 1/fs
T_total = 200000.0
Npts = int(T_total*fs)
rng = np.random.default_rng(42)


N_arw = 2.0e-3                 # ARW coefficient 
B_bi  = 0.5e-3                 # Bias-instability 
tau_BR = 2.0e3                 # BI-RRW crossover time 

sigma0 = 0.664 * B_bi
K_rrw  = sigma0 / np.sqrt(tau_BR)   # RRW coefficient  sigma = Ksqrt(T)

def overlapping_adev(x, fs, taus):
    x = np.asarray(x, float)
    N = len(x)
    sigmas = np.zeros_like(taus, float)
    for i, tau in enumerate(taus):
        m = int(round(tau*fs))
        if m < 2 or 2*m >= N:
            sigmas[i] = np.nan; continue
        csum = np.cumsum(np.insert(x,0,0.0))
        avgs = (csum[m:] - csum[:-m]) / m         # overlapping cluster averages
        diffs = avgs[m:] - avgs[:-m]
        sigmas[i] = np.sqrt(0.5*np.mean(diffs**2))
    return sigmas

taus = np.logspace(-1, 4.2, 90)

##### White rate noise: scale to N/sqrt(T) at small T
white_unit = rng.standard_normal(Npts)
adev_white = overlapping_adev(white_unit, fs, taus)
band_w = (taus >= 0.2) & (taus <= 1.0) & np.isfinite(adev_white)
scale_white = np.nanmedian( (N_arw/np.sqrt(taus[band_w])) / adev_white[band_w] )
white = scale_white * white_unit

#### Bias-instability, scale to sigma0 in mid T
taus_ou = np.logspace(-1, 4, 6)             
bias_flick = np.zeros(Npts)
for tau_i in taus_ou:
    phi = np.exp(-dt/tau_i); q = np.sqrt(1-phi**2)
    x = np.zeros(Npts); eps = rng.standard_normal(Npts)
    for k in range(1, Npts):
        x[k] = phi*x[k-1] + q*eps[k]
    bias_flick += x
# normalize and scale to model flat part in the middle
adev_bf = overlapping_adev(bias_flick, fs, taus)
band_b = (taus >= 10.0) & (taus <= 300.0) & np.isfinite(adev_bf)
scale_bf = np.nanmedian( (sigma0) / adev_bf[band_b] )
bias_flick *= scale_bf

#####RRW drift: random walk of bias, scale to Ksqrt(T) in long T
eps = rng.standard_normal(Npts)
bias_rw_unit = np.cumsum(np.sqrt(dt) * eps)   # unit random walk
adev_rw = overlapping_adev(bias_rw_unit, fs, taus)
band_r = (taus >= tau_BR/2) & (taus <= 2*tau_BR) & np.isfinite(adev_rw)
scale_rw = np.nanmedian((K_rrw*np.sqrt(taus[band_r])) / adev_rw[band_r])
bias_rw = scale_rw * bias_rw_unit

#Putting all together
rate = white + bias_flick + bias_rw
adev = overlapping_adev(rate, fs, taus)
# Plotting, make sure to add some lines separating the sections 
plt.figure(figsize=(7,5))
plt.loglog(taus, adev, lw=2)
plt.xlabel(r'$\tau$ [s]')
plt.ylabel(r'$\sigma(\tau)$')
plt.grid(True, which='both')
plt.axvline(84, color='gray', linestyle='--')
plt.axvline(935, color='gray', linestyle='--')
plt.text(12, 0.0012, "A", fontsize=16, ha='center')
plt.text(300, 0.001, "B", fontsize=16, ha='center')
plt.text(3500, 0.001, "C", fontsize=16, ha='center')
plt.title('Allan deviation(white + bias-instability + rate random walk)')
plt.savefig('allan_variance_plot_labeled.png')
plt.show()


