import numpy as np
import matplotlib.pyplot as plt

fs = 1
T=1/fs

cross = 0.35
dpoles = np.linspace(0.1, 1, dtype=complex)
dpoles = np.append(dpoles, cross+1j*np.linspace(-0.9, 0.9, 10000))
dpoles = np.append(dpoles, 0.7+1j*np.linspace(-0.7, 0.7, 10000))
dpoles = np.append(dpoles, np.exp(-0.5*np.pi)+1j*np.linspace(-0.95, 0.95, 10000))

cpoles = np.log(dpoles)/T

circ = np.exp(1j*np.linspace(np.pi/2, 3*np.pi/2, 360))

r = np.linspace(np.exp(-np.pi), 1, 500)
phi = -np.log(r)
locus = r*np.exp(1j*phi)

plt.figure(figsize=(6, 12))
plt.subplot(2, 1, 2)
plt.scatter(dpoles.real, dpoles.imag, color="C2", label="test poles")
plt.plot(locus.real, locus.imag, label="Butterworth damping")
plt.plot(-circ.real, -circ.imag, 'k', label='stability boundary')
plt.plot((cross**-circ).real, (cross**-circ).imag, color="C1",
         label = "constant natural frequencies")
plt.plot(((0.7)**-circ).real, ((0.7)**-circ).imag, color="C1")
plt.plot((np.exp(-0.5*np.pi)**-circ).real, (np.exp(-0.5*np.pi)**-circ).imag, color="C1")
plt.xlim(-1, 1.01)
plt.legend()
plt.title("discrete time complex plane")

plt.subplot(2, 1, 1)
plt.scatter(cpoles.real, cpoles.imag, color="C2",  label="test poles")
plt.plot(np.log(locus).real, np.log(locus).imag, label="Butterworth damping")
plt.plot(-np.log(cross)*circ.real, -np.log(cross)*circ.imag, color="C1",
         label = "constant natural frequencies")
plt.plot(-np.log(0.7)*circ.real, -np.log(0.7)*circ.imag, color="C1")
plt.plot(-np.log(np.exp(-0.5*np.pi))*circ.real, -np.log(np.exp(-0.5*np.pi))*circ.imag, color="C1")
plt.ylim(-1.1, 1.1)
plt.xlim(-2, 0.01)
plt.grid()
plt.legend()
plt.title("continuous time complex plane")

# %% find

# zeta = 1/ sqrt(1 + (imag/real)**2)
# disc time r * exp(i theta)
# cont time real + i imag = ln(r) + i theta
# zeta = 1/ sqrt(1 + (theta/ln(r))**2)
# butterworth: zeta = 1/sqrt2 -> theta = -ln(r)
#
# disc time I + iQ = rho + i (1 -rho ) * sqrt(phi-1)
# define alpha = (1/rho - 1)*sqrt(phi-1)
# invertible: phi = 1 + (alpha*rho/(1-rho))**2
# disc time I + iQ = rho * (1 + i alpha)
# theta = -ln(r)
# theta = arctan(Q/I) = arctan(alpha)
#       = -ln(sqrt(rho**2 + alpha**2*rho**2)) = -ln(rho) - ln(1+alpha)
#
# 0 = arctan(alpha) + ln(1+alpha) + ln(rho)

import numpy as np
from scipy.optimize import fsolve, curve_fit

def butterwoth_phi(rho):
    def equation(a, p):
        return np.arctan(a) + 0.5 * np.log(1 + a) + np.log(p)
    return fsolve(equation, 0, args=(rho,))[0]

def power_law(x, a, b, c, d):
    return a + b * np.exp(c/x) + d*x
def straight(x, a, b):
    return a + b * x

def alpha2phi(alpha, rho):
    return 1 + (alpha*rho/(1-rho))**2

sol = np.zeros_like(r)
for i, p in enumerate(r):
    sol[i] = butterwoth_phi(p)

phi = alpha2phi(sol, r)

mask = np.where(r>0)[0][1:-1]
poptp, pcov = curve_fit(power_law, r[mask], phi[mask])
mask = np.where(r>0.4)[0][:-1]
popts, pcov = curve_fit(straight, r[mask], phi[mask])

plt.ylim(0, 1)
plt.xlim(0.3, 1)

plt.figure()
plt.subplot(2,1,1)
plt.plot(r, phi, label=r"$\varphi$")
plt.plot(r, power_law(r, *poptp), label=rf'${poptp[1]:.2f} \exp({poptp[2]:.2f}/x) + {poptp[3]:.2f}\rho + {poptp[0]:.2f}$')
plt.plot(r, straight(r, *popts), label=rf'${popts[1]:.4f}\rho + {popts[0]:.4f}$')
plt.legend()
plt.ylim(1.2, 1.5)
plt.subplot(2,1,2)
plt.plot(0,0)
plt.plot(r, np.abs(phi-power_law(r, *poptp)), label=rf'${poptp[1]:.2f} \exp({poptp[2]:.2f}/x) + {poptp[3]:.2f}\rho + {poptp[0]:.2f}$')
plt.plot(r, np.abs(phi-straight(r, *popts)), label=rf'${popts[1]:.4f}\rho + {popts[0]:.4f}$')
# plt.plot(r, np.abs(phi-straight(r, 10/9, 1/3)))
plt.plot(r, np.abs(phi-straight(r, 1.11, 0.33)), label=rf'$0.33\rho + 1.11$')
plt.ylim(-0.001, 0.011)
plt.plot([0.25, 1], [0.01, 0.01], "k--")
plt.plot([0.35, 0.35], [-1, 0.01], "k--")
plt.plot([0.4, 0.4], [-1, 0.005], "k--")
plt.plot([0.3, 1], [0.005, 0.005], "k--")
plt.xlabel(r'$\rho$')
plt.legend()
