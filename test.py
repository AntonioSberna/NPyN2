
import numpy as np
import n2


### Read outputs from the analysis
# Base shear [kN]
f = -np.sum(np.loadtxt('R.out'), axis=1)/1000
# Top displacement
d = np.loadtxt('D.out', usecols=3)
# First eigenvector
ev = np.loadtxt('modal.out')



# Modal masses
mass1 = 115.95310907237513*4;
mass2 = 101.93679918450561*4;
masses = np.array([0, mass1, mass1, mass1, mass2])



# Eigenvector normalization
eigenVector = ev/ev[-1]

# Modal mass value
modalmass = np.dot(masses.conj(), eigenVector)

# First partecipation factor coefficient
modalcoeff = modalmass/np.einsum('i,i,i->', masses.conj(), eigenVector, eigenVector)



# Spettro
spectrum = n2.Spectrum({"S": 1, "eta": 1, "C_c": 1, "ag": 0.271, "F0": 2.432, "Tcstar": 0.372})


N2 = n2.N2(f, d, modalcoeff, np.sum(masses)/1000, spectrum)
N2.run_analysis()

# Ductility check
mu_c, mu_d = N2.ductility_check()

# print(mu_c, mu_d)






# # Plot
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.gca()


# _, ax = spectrum.plot_spectrum(fig, ax, figkwargs = {"label":'Spectral acceleration'})
# ax.set_xlabel(r'Period $\left[\mathrm{s}\right]$')
# # ax.set_ylabel(r'Spectral acceleration $\left[\mathrm{m}/\mathrm{s}^2\right]$')
# ax.set_ylabel(r'Spectral acceleration $\left[g\right]$')
# ax.grid(alpha = 0.2)
# ax.legend()
# plt.show()


# _, ax = spectrum.plot_elastic_adrs(fig, ax, d_fact = 1000)
# _, ax = spectrum.plot_const_duct_adrs(fig, ax, mu_c, d_fact = 1000)
# ax.set_ylabel(r'$S_a \quad\left[\mathrm{g}\right]$')
# ax.set_xlabel(r'$S_d \quad\left[\mathrm{mm}\right]$')
# ax.grid(alpha = 0.2)
# ax.legend()
# plt.show()



