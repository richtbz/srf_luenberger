# Open loop recursive least squares based SRF cavity parameter identification
# Copyright (C) 2023 V. Ziemann
# Source: https://github.com/volkziem/SysidRFcavity
#
# The content of this file was originally part of the SysidRFcavity project
# and is redistributed here under the same license terms.
#
# Modifications:
# - 2026 - Bozo Richter - convert Matlab code to python
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from rls import rls


Niter=10000     # number of iterations
Nforget=100;    # forgetting horizon
alpha=1-1/Nforget
Npulse=-40000   # negative values turn off pulses
R=1             # shunt impedance, ensure current and voltage is normalized
sigp=0.0001     # process noise level
sigm=0.001      # measurement noise level
dt=1e-7         # sample time at 10 MHz

for periodic_pertubation in [0, 1]:
    # spoke parameters
    omega0=2*np.pi*352e6     # resonant frequency
    QE=1.8e5                 # external-Q
    QL=QE                    # loaded-Q, if sc same as QE
    omega12=omega0/(2*QL)    # bandwidth
    omegaE=omega0/QE
    domega=-omega12/2        # set detuning to half the bandwidth

    q0=np.array([[omega12*dt], [domega*dt]])   # bandwidth and detuning
    bandwidth=q0[0, 0]
    F0=np.array([[-q0[0,0], -q0[1,0]], [q0[1, 0],-q0[0, 0]]])  # eq. 4, F0=Abar
    Areal=np.eye(2)+F0            # eq. 6
    Breal=R*omega12*dt*np.eye(2)  # eq. 1, for generator current, used to drive cavity
    BrealE=R*omegaE*dt*np.eye(2)  # eq. 3, for forward current used in sysid
    pp=1                          # initial value of pT
    qhat=np.zeros((2,1))          # initial parameter estimate
    xp=sigp*np.random.randn(2,1)  # initialize cavity voltage reading, process noise
    x=sigm*np.random.randn(2,1)   # measured voltage inside cavity
    data=np.zeros((Niter,7))      # storage for later plotting
    uset=np.array([[0],[0]])      # generator is off at start

    # %%

    for iter in range(Niter): # main iteration loop
        if iter==100:
            uset=np.array([[1], [0]]) # first pulse
        if Npulse>0:                    # negative Npulse turns off later pulses
            if iter % Npulse == 0:       # pulsed operation
                if uset[0,0]==0:
                    uset=np.array([[1], [0]])
                else:
                    uset=np.array([[0], [0]])

        if 1:   # bandwidth change, Figure 5
            factor=2                    # magnitude of step in bandwidth
            if iter==Niter//2:            # start half-way of Niter
                q0[0, 0]=factor*q0[0,0] # increase bandwidth by factor
                omega12=omega12*factor;
                Areal=np.eye(2)+np.array([[-q0[0,0],-q0[1,0]],[q0[1,0],-q0[0,0]]])
                Breal=Breal*factor;
                QL=QL/factor;
            if 0:  #iter==Niter/2     # TURNED OFF, stop at half of Niter
                q0[0, 0]=q0[0,0]/factor   # reduce bandwidth by factor
                omega12=omega12/factor;
                Areal=np.eye(2)+np.array([[-q0[0,0],-q0[1,0]], [q0[1,0],-q0[0,0]]])
                Breal=Breal/factor
                QL=QL*factor
        if periodic_pertubation: # periodic perturbation (1=1 kHz, 20=20 kHz), Figure 3 and 4
            q0[1,0]=0.5*bandwidth*np.sin(20*6.2832e-4*iter);
            Areal=np.eye(2)+np.array([[-q0[0,0],-q0[1,0]],[q0[1,0],-q0[0,0]]])

        #.........................................cavity dynamics
        u=uset
        xpnew=Areal@xp+Breal@u+sigp*np.random.randn(2, 1) # eq. 4, xp=V
        xnew=xpnew+sigm*np.random.randn(2, 1)             # eq. 5, x=V', add measurement noise

        #...................................system identification
        #up=u*omega12/omegaE;             # forward current I^+
        up=u*QE/(2*QL)                   # eq.2
        y=xnew-x-BrealE*up               # eq.11, right
        vv2=(x.T@x)[0,0]
        tmp=alpha/(alpha+pp*vv2)         # bracket in eq.20
        qhat=tmp*(qhat+np.array([[-x[0,0]*y[0,0]-x[1,0]*y[1,0]],[-x[1,0]*y[0,0]+x[0,0]*y[1,0]]])*pp/alpha) # eq.21
        pp=tmp*pp/alpha                  # eq. 20
        xp=xpnew       # update process voltage in the cavity
        x=xnew         # remember measured voltage

        #.save for later plotting
        data[iter, 0]=x[0,0];       # normalized voltages, measured
        data[iter, 1]=x[1, 0];
        data[iter, 2]=up[0,0];       # generator currents
        data[iter, 3]=up[1, 0];
        data[iter, 4]=qhat[0,0];    # bandwidth
        data[iter, 5]=qhat[1, 0];    # detuning
        data[iter, 6]=pp;         # p_T

    bw, det, pt = rls(dt,
                      Vf=data[:, 2]+1j*data[:, 3],
                      Vcav=data[:, 0]+1j*data[:, 1],
                      f12=bandwidth/dt/2/np.pi,
                      ffilt=1e5/2/np.pi,
                      fdet0=0,
                      amin=0,
                      noise=None)


    # %%

    mm=list(range(Niter))         # xaxis for plots
    mm2=list(range(Niter//2, Niter))  # just the second half of the data

    if 1:    # plot of voltages and currents
        fig, axs = plt.subplots(nrows=2)
        axs[0].set_title("RF simulation data")
        axs[0].plot(mm, data[:, 0],'k')
        axs[0].plot(mm, data[:, 1],'r')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('v_r, v_i')
        axs[0].legend(['v_r','v_i'])

        axs[1].plot(mm, data[:,2],'k')
        axs[1].plot(mm, data[:,3],'r')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('v_r, v_i')
        axs[1].legend(['v_r','v_i'])

    if 1:   # evolution of fit parameters
        plt.figure()
        fac=1/(2*np.pi*dt);   # convert q-units to Hz
        plt.title("estimates - main")
        plt.plot(mm,fac*data[:, 4],'k')
        plt.plot(mm,fac*data[:, 5],'r')
        plt.xlabel('Iterations')
        plt.ylabel(r'$f_{12},\Delta f     [Hz]$')
        plt.legend([r'$f_{12}$',r'$\Delta f$'])
        plt.ylim([-1100,2500])
        #xlim([4950,5350])

        plt.figure()
        plt.title("estimates - function")
        plt.plot(mm,bw,'k')
        plt.plot(mm,det,'r')
        plt.xlabel('Iterations')
        plt.ylabel(r'$f_{12},\Delta f     [Hz]$')
        plt.legend([r'$f_{12}$',r'$\Delta f$'])
        plt.ylim([-1100,2500])
        #xlim([4950,5350])

        rms_second_half=np.std(data[mm2, 4:6], axis=0)

    if 1:   # PT
        plt.figure() #   plot PT
        plt.title("error bars")
        plt.plot(mm, data[:, 6], 'k', linewidth=5)
        plt.plot(mm, pt, 'r')
        plt.gca().set_yscale('log')
        plt.xlim([1,Niter])
        plt.xlabel('Iterations')
        plt.legend(['p_T main', 'p_T function'])

        asymp=1/(Nforget*(data[-1, :2].T@data[-1, :2]));
        plt.plot([1,Niter],[asymp,asymp],'r--')
        error_bar_q=np.sqrt(asymp)*np.sqrt(sigm**2+2*sigp**2)


    # if 0:   # estimation error
        # figure(4); clf                  # Plot estimation error
        # eb1=np.sqrt(sigm^2+2*sigp^2)*np.sqrt(data[:,6]) # empirical error bar of q(1)
        # data[:,0]=np.abs(data[:,4]-q0[0, 0]) # estimation error of q(1)
        # loglog(mm,data(:,1),'k',mm,eb1,'b-.','LineWidth',2);
        # xlim([1,Niter]); ylim([7e-7,2*max(data(:,5))])
        # xlabel('Iterations'); ylabel('Estimation error |a_T(1)|')
        # legend('Simulation','Errorbars');

plt.show()
