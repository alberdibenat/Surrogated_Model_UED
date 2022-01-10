import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp2d
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

from scipy.optimize import curve_fit
from scipy.optimize import minimize
import scipy.optimize as skopt
from sklearn.base import clone
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


class ToF_Jitter:
    
    c = 3e8 #m/s
    c_light = 3e8
    f = 1.3e9 #Hz
    w = 2*np.pi*f #rad/s
    w_deg = w*180/np.pi #deg/s
        

    def __init__(self,filename):
        #READ THE GUN SIMULATION OUTCOMES
        self.filename = filename
        X_mesh = []
        Y_mesh = []
        TOF_carrier = []
        energy_carrier = []
        with open(filename,'r') as f:
            count = 0
            for line in f.readlines():
                if count == 0:
                    dimX,dimY = [int(x) for x in line.strip('\n').split(',')]
                    count += 1
                else:
                    X_Y_ToF_E = [float(x) for x in line.split(' ')]
                    X_mesh.append(X_Y_ToF_E[0])
                    Y_mesh.append(X_Y_ToF_E[1])
                    TOF_carrier.append(X_Y_ToF_E[2])
                    energy_carrier.append(X_Y_ToF_E[3])
            f.close()
        X_mesh = np.asarray(X_mesh).reshape(dimX,dimY)
        Y_mesh = np.asarray(Y_mesh).reshape(dimX,dimY)
        TOF_carrier = np.asarray(TOF_carrier).reshape(dimX,dimY)
        energy_carrier = np.asarray(energy_carrier).reshape(dimX,dimY)
        
        #TOF_carrier = np.transpose(TOF_carrier)
        #energy_carrier = np.transpose(energy_carrier)
        total_energy = energy_carrier + 0.511
        gamma_carrier = (energy_carrier+0.511)/0.511
        
        
        Gun_amplitudes = [sub[0] for sub in Y_mesh]
        Gun_phases = X_mesh[0]
        
        
        
        
        #Now we have a grid in 2d with the TOF saved in TOF_carrier, evaluated at the points Gun_amplitudes,Gun_phases. We will interpolate it.
        self.TOF_2d = interp2d(Gun_phases,Gun_amplitudes,TOF_carrier,kind='cubic')
        self.Energy_2d = interp2d(Gun_phases,Gun_amplitudes,energy_carrier,kind='cubic') #Kinetic energy
        self.Gamma_2d = interp2d(Gun_phases,Gun_amplitudes,gamma_carrier,kind='cubic')
        self.Total_energy_2d = interp2d(Gun_phases,Gun_amplitudes,total_energy,kind='cubic') #Total energy



    def Jitter_calculator(self,gun_phase,gun_amplitude,b1_phase,b1_amplitude,b2_phase,b2_amplitude,b3_phase,b3_amplitude):

        ##---------- GLOBAL CONSTANTS ------------
        c_light = 3.0e8
        e = 1.6e-19
        me_SI = 9.1e-31
        me_nat = me_SI*c_light**2 / 1.6e-19
        rest_energy = me_SI*c_light**2
        
        
        ##---------- COMMON PARAMETERS FOR ALL CAVITIES ------------
        frequency = 1.3e9
        wavelength = c_light/frequency
        w = 2*np.pi*frequency
        k = 2*np.pi/wavelength
        length_booster = 1.0*wavelength
        
        
        ##---------- AMPLITUDES AND PHASES FOR CAVITIES ------------
        Amplitude_b1 = b1_amplitude*1.0e6
        V0_b1 = e*Amplitude_b1*length_booster/2.0
        Amplitude_b2 = b2_amplitude*1.0e6
        V0_b2 = e*Amplitude_b2*length_booster/2.0
        Amplitude_b3 = b3_amplitude*1.0e6
        V0_b3 = e*Amplitude_b3*length_booster/2.0
        Amplitude_gun = gun_amplitude #MV/m
        Phase_gun = gun_phase #ToF and energy maps are built with Astra phase for the gun
        Phase_b1 = b1_phase #Relative to on-crest phase
        Phase_b2 = b2_phase #Relative to on-crest phase
        Phase_b3 = b3_phase #Relative to on-crest phase
        
        
        ##---------- AMPLITUDE AND PHASE DEVIATIONS ------------
        sigma_RL = 300e-15
        sigma_phase_gun = 0.05
        sigma_phase_gun = 180*np.sqrt((sigma_phase_gun*np.pi/180)**2+(sigma_RL*w)**2)/np.pi
        sigma_amplitude_gun = Amplitude_gun*1e-4
        sigma_phase_b1 = 0.05*np.pi/180
        sigma_amplitude_Vb1 = V0_b1*1e-4
        sigma_phase_b2 = 0.05*np.pi/180
        sigma_amplitude_Vb2 = V0_b2*1e-4
        sigma_phase_b3 = 0.05*np.pi/180
        sigma_amplitude_Vb3 = V0_b3*1e-4
        
        
        ##---------- DRIFT LENGTHS ------------
        Lgun = 0.323
        L1 = 3.20 - Lgun
        L2 = 4.02 - (L1+Lgun)
        L3 = 4.88 - (L2+L1+Lgun)
        L4 = 7.64 - (L3+L2+L1+Lgun)
        
        
        
        
        ##---------- TIME TO EXIT THE GUN, ENERGY AT L1 AND T1 ------------
        t_gun = self.TOF_2d(Phase_gun,Amplitude_gun)
        gamma_gun = self.Gamma_2d(Phase_gun,Amplitude_gun)
        E_gun = self.Total_energy_2d(Phase_gun,Amplitude_gun)*e*1e6 #in J
        t1 = L1*E_gun/(c_light*np.sqrt(E_gun**2-rest_energy**2)) #time of flight needed to cross L1
        
        dtgun_dphasegun = self.TOF_2d(Phase_gun,Amplitude_gun,dx=1,dy=0)
        dtgun_damplitudegun = self.TOF_2d(Phase_gun,Amplitude_gun,dx=0,dy=1)
        degun_dphasegun = self.Total_energy_2d(Phase_gun,Amplitude_gun,dx=1,dy=0)*e*1e6 #From MeV/deg to J/deg
        degun_damplitudegun = self.Total_energy_2d(Phase_gun,Amplitude_gun,dx=0,dy=1)*e*1e6 #From MeV*m/MV to J*m/MV
        
        L1_factor = L1/(me_SI*c_light**3*(gamma_gun**2-1)**(3.0/2.0))
        dt1_dphasegun = - L1_factor*degun_dphasegun
        dt1_damplitudegun = - L1_factor*degun_damplitudegun
        
        
        
        
        ##---------- PARAMETERS AT L2 ------------
        #Phase scan to find Maximum energy gain phase for booster1:
        phase_scans = np.linspace(0.0,2*np.pi,10000)
        maximum = 0.0
        index = 0
        for k in range(len(phase_scans)):
            E_1 = E_gun + V0_b1*np.cos(phase_scans[k] + w*(t_gun + t1)) #in J
            if E_1>maximum:
                index = k
                maximum = E_1
        maximum_energy_gain_phase = phase_scans[index]
        #print(maximum_energy_gain_phase*180/np.pi)
        Phase_b1_0 = maximum_energy_gain_phase + Phase_b1*np.pi/180 #rad, phase of booster 1 at t=0.0
        
        E_1 = E_gun + V0_b1*np.cos(Phase_b1_0 + w*(t_gun + t1)) #in J
        gamma_1 = E_1/(0.511*1e6*e) 
        t2 = L2/(c_light*np.sqrt(1.0-1.0/(gamma_1**2))) #time of flight needed to cross L2
        
        
        L2_factor = L2/(me_SI*c_light**3*(gamma_1**2-1)**(3.0/2.0))
        dE1_dphasegun = degun_dphasegun - V0_b1*w*np.sin(Phase_b1_0+w*(t_gun+t1))*(dtgun_dphasegun+dt1_dphasegun)
        dE1_damplitudegun = degun_damplitudegun - V0_b1*w*np.sin(Phase_b1_0+w*(t_gun+t1))*(dtgun_damplitudegun+dt1_damplitudegun)
        dE1_dphaseb1 = - V0_b1*np.sin(Phase_b1_0 + w*(t_gun + t1))
        dE1_damplitudeb1 = np.cos(Phase_b1_0 + w*(t_gun + t1))
        
        dt2_dphasegun = - L2_factor*dE1_dphasegun
        dt2_damplitudegun = - L2_factor*dE1_damplitudegun
        dt2_dphaseb1 = - L2_factor*dE1_dphaseb1
        dt2_damplitudeb1 = - L2_factor*dE1_damplitudeb1
        
        
        
        ##---------- PARAMETERS AT L3 ------------
        #Phase scan to find Maximum energy gain phase for booster1:
        maximum = 0.0
        index = 0
        for k in range(len(phase_scans)):
            E_2 = E_1 + V0_b2*np.cos(phase_scans[k] + w*(t_gun + t1 + t2)) #in J
            if E_2 > maximum:
                index = k
                maximum = E_2
        maximum_energy_gain_phase = phase_scans[index]
        #print(maximum_energy_gain_phase*180/np.pi)
        Phase_b2_0 = maximum_energy_gain_phase + Phase_b2*np.pi/180 #rad, phase of booster 1 at t=0.0
        
        E_2 = E_1 + V0_b2*np.cos(Phase_b2_0 + w*(t_gun + t1 + t2)) #in J
        gamma_2 = E_2/(0.511*1e6*e) 
        t3 = L3/(c_light*np.sqrt(1.0-1.0/(gamma_2**2))) #time of flight needed to cross L3
        
        L3_factor = L3/(me_SI*c_light**3*(gamma_2**2-1)**(3.0/2.0))
        dE2_dphasegun = dE1_dphasegun - V0_b2*w*np.sin(Phase_b2_0+w*(t_gun+t1+t2))*(dtgun_dphasegun+dt1_dphasegun+dt2_dphasegun)
        dE2_damplitudegun = dE1_damplitudegun - V0_b2*w*np.sin(Phase_b2_0+w*(t_gun + t1 + t2))*(dtgun_damplitudegun+dt1_damplitudegun+dt2_damplitudegun)
        dE2_dphaseb1 = dE1_dphaseb1 - V0_b2*w*np.sin(Phase_b2_0 + w*(t_gun + t1 + t2))*dt2_dphaseb1
        dE2_damplitudeb1 = dE1_damplitudeb1 - V0_b2*w*np.sin(Phase_b2_0 + w*(t_gun + t1 + t2))*dt2_damplitudeb1
        dE2_dphaseb2 = - V0_b2*np.sin(Phase_b2_0 + w*(t_gun + t1 + t2))
        dE2_damplitudeb2 = np.cos(Phase_b2_0 + w*(t_gun + t1 + t2))
        
        dt3_dphasegun = - L3_factor*dE2_dphasegun
        dt3_damplitudegun = - L3_factor*dE2_damplitudegun
        dt3_dphaseb1 = - L3_factor*dE2_dphaseb1
        dt3_damplitudeb1 = - L3_factor*dE2_damplitudeb1
        dt3_dphaseb2 = - L3_factor*dE2_dphaseb2
        dt3_damplitudeb2 = - L3_factor*dE2_damplitudeb2
        
        
        ##---------- PARAMETERS AT L4 ------------
        #Phase scan to find Maximum energy gain phase for booster1:
        maximum = 0.0
        index = 0
        for k in range(len(phase_scans)):
            E_3 = E_2 + V0_b3*np.cos(phase_scans[k] + w*(t_gun + t1 + t2 + t3)) #in J
            if E_3 > maximum:
                index = k
                maximum = E_3
        maximum_energy_gain_phase = phase_scans[index]
        #print(maximum_energy_gain_phase*180/np.pi)
        Phase_b3_0 = maximum_energy_gain_phase + Phase_b3*np.pi/180 #rad, phase of booster 1 at t=0.0
        
        E_3 = E_2 + V0_b3*np.cos(Phase_b3_0 + w*(t_gun + t1 + t2 + t3)) #in J
        gamma_3 = E_3/(0.511*1e6*e) 
        t4 = L4/(c_light*np.sqrt(1.0-1.0/(gamma_3**2))) #time of flight needed to cross L3
        
        L4_factor = L4/(me_SI*c_light**3*(gamma_3**2-1)**(3.0/2.0))
        dE3_dphasegun = dE2_dphasegun - V0_b3*w*np.sin(Phase_b3_0+w*(t_gun+t1+t2+t3))*(dtgun_dphasegun+dt1_dphasegun+dt2_dphasegun+dt3_dphasegun)
        dE3_damplitudegun = dE2_damplitudegun - V0_b3*w*np.sin(Phase_b3_0+w*(t_gun + t1 + t2+t3))*(dtgun_damplitudegun+dt1_damplitudegun+dt2_damplitudegun+dt3_damplitudegun)
        dE3_dphaseb1 = dE2_dphaseb1 - V0_b3*w*np.sin(Phase_b3_0 + w*(t_gun + t1 + t2 + t3))*(dt2_dphaseb1 + dt3_dphaseb1)
        dE3_damplitudeb1 = dE2_damplitudeb1 - V0_b3*w*np.sin(Phase_b3_0 + w*(t_gun + t1 + t2 + t3))*(dt2_damplitudeb1 + dt3_damplitudeb1)
        dE3_dphaseb2 = dE2_dphaseb2 - V0_b3*w*np.sin(Phase_b3_0 + w*(t_gun + t1 + t2 + t3))*dt3_dphaseb2
        dE3_damplitudeb2 = dE2_damplitudeb2 - V0_b3*w*np.sin(Phase_b3_0 + w*(t_gun + t1 + t2 + t3))*dt3_damplitudeb2
        dE3_dphaseb3 = - V0_b3*np.sin(Phase_b3_0 + w*(t_gun + t1 + t2 + t3))
        dE3_damplitudeb3 = np.cos(Phase_b3_0 + w*(t_gun + t1 + t2 + t3))
        
        dt4_dphasegun = - L4_factor*dE3_dphasegun
        dt4_damplitudegun = - L4_factor*dE3_damplitudegun
        dt4_dphaseb1 = - L4_factor*dE3_dphaseb1
        dt4_damplitudeb1 = - L4_factor*dE3_damplitudeb1
        dt4_dphaseb2 = - L4_factor*dE3_dphaseb2
        dt4_damplitudeb2 = - L4_factor*dE3_damplitudeb2
        dt4_dphaseb3 = - L4_factor*dE3_dphaseb3
        dt4_damplitudeb3 = - L4_factor*dE3_damplitudeb3
        
        
        
        ##---------- COLLECT AND CALCULATE ------------
        Time_of_flight = t_gun + t1 + t2 + t3 + t4
        
        
        dToF_dphasegun = dtgun_dphasegun + dt1_dphasegun + dt2_dphasegun + dt3_dphasegun + dt4_dphasegun
        dToF_damplitudegun = dtgun_damplitudegun + dt1_damplitudegun + dt2_damplitudegun + dt3_damplitudegun + dt4_damplitudegun 
        dToF_dphaseb1 = dt2_dphaseb1 + dt3_dphaseb1 + dt4_dphaseb1
        dToF_damplitudeb1 = dt2_damplitudeb1 + dt3_damplitudeb1 + dt4_damplitudeb1
        dToF_dphaseb2 = dt3_dphaseb2 + dt4_dphaseb2
        dToF_damplitudeb2 = dt3_damplitudeb2 + dt4_damplitudeb2
        dToF_dphaseb3 = dt4_dphaseb3
        dToF_damplitudeb3 = dt4_damplitudeb3
        
        self.ToF_Jitter = np.sqrt((dToF_dphasegun*sigma_phase_gun)**2+(dToF_damplitudegun*sigma_amplitude_gun)**2 + 
                            (dToF_dphaseb1*sigma_phase_b1)**2+(dToF_damplitudeb1*sigma_amplitude_Vb1)**2 + 
                            (dToF_dphaseb2*sigma_phase_b2)**2+(dToF_damplitudeb2*sigma_amplitude_Vb2)**2 +
                            (dToF_dphaseb3*sigma_phase_b3)**2+(dToF_damplitudeb3*sigma_amplitude_Vb3)**2)
        
        self.Energy_deviation = np.sqrt((dE3_dphasegun*sigma_phase_gun)**2 + (dE3_damplitudegun*sigma_amplitude_gun)**2 + 
                                  (dE3_dphaseb1*sigma_phase_b1)**2 + (dE3_damplitudeb1*sigma_amplitude_Vb1)**2 +
                                  (dE3_dphaseb2*sigma_phase_b2)**2 + (dE3_damplitudeb2*sigma_amplitude_Vb2)**2 +
                                  (dE3_dphaseb3*sigma_phase_b3)**2 + (dE3_damplitudeb3*sigma_amplitude_Vb3)**2)
        
        
        return self.ToF_Jitter, (self.Energy_deviation/E_3)*100
