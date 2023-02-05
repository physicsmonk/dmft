import matplotlib.pyplot as plt
import numpy as np
plt.style.use("publication")

real_freqs = np.loadtxt("real_freqs.txt")
model_spec = np.loadtxt("model_spec_func.txt")
mats_func = np.loadtxt("model_mats_func.txt")
def_model = np.loadtxt("default_model.txt")
spec = np.loadtxt("spec_func.txt")
misfit = np.loadtxt("log10chi2_log10alpha.txt")
retard_func = np.loadtxt("retarded_func.txt")

fig, ax = plt.subplots(3, figsize=(8.5/2.54, 18.5/2.54))
ax[0].plot(real_freqs, model_spec[:,0])
ax[0].plot(real_freqs, model_spec[:,3])
ax[0].plot(real_freqs, def_model[:,0], linestyle="--")
ax[0].plot(real_freqs, def_model[:,3], linestyle="--")
ax[0].plot(real_freqs, spec[:,0], linestyle='', marker='o')
ax[0].plot(real_freqs, spec[:,3], linestyle='', marker='^')
ax[0].plot(real_freqs, retard_func[:,0], linestyle='', marker='d')
#ax[0].set_xlim((-10, 10))
#ax[0].set_ylim((-1, 1))
ax[1].plot(mats_func[:,1])
ax[1].plot(mats_func[:,2])
ax[2].plot(misfit[:,0], misfit[:,1], marker='o')
plt.show();