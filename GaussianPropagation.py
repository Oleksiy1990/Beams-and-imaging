import numpy as np
import matplotlib.pyplot as plt 
from lmfit import Parameters, minimize, fit_report, Minimizer

def __w_Gauss_freespace_residual(pars,meas_points,meas_beamwidths=None,lam=None):
    parvals = pars.valuesdict()
    w0 = parvals["omega_zero"]
    z0 = parvals["z0"]

    zR = (np.pi*w0**2)/lam # Rayleigh range 
    model_beamwidth = w0*np.sqrt(1+((meas_points-z0)/zR)**2)
    #That is the equation for Gaussian beam rad propagation
    
    if meas_beamwidths is None:
        return model_beamwidth
    return (model_beamwidth - meas_beamwidths)

def Gaussian_beam_propagation(meas_points,widths,lambda_beam,plot=False):
    
    params_BeamPropagation = Parameters()
    params_BeamPropagation.add('omega_zero',value=widths[0])
    params_BeamPropagation.add('z0',value=0)
    
    fit = Minimizer(__w_Gauss_freespace_residual,params_BeamPropagation,fcn_args=(meas_points,),\
        fcn_kws={"meas_beamwidths":widths,"lam":lambda_beam})
    fit_res = fit.minimize(maxfev=10**8)
    print(fit_report(fit_res))

    if plot is True:
        print("Let's plot it")
        fitted_w0 = fit_res.params.valuesdict()["omega_zero"]
        fitted_z0 = fit_res.params.valuesdict()["z0"]
        if fitted_z0<meas_points[0]:
            plotpoints = np.linspace(fitted_w0,meas_points[-1],100)
        elif meas_points[0]<fitted_z0<meas_points[-1]:
            plotpoints = np.linspace(meas_points[0],meas_points[-1],100)
        else:
            plotpoints = np.linspace(meas_points[0],fitted_z0,100)
        model_beamwidth = __w_Gauss_freespace_residual(fit_res.params,plotpoints,meas_beamwidths=None,lam=lambda_beam)
        plt.plot(plotpoints,model_beamwidth,'b')
        plt.scatter(meas_points,widths,c="r")
        plt.show()




meas = np.array([5,10,15,20])*1e1
results_wide98 = np.array([2.908,2.862,2.944,3.646])
results_wide74 = np.array([2.986,3.101,3.275,3.560])
results_short98 = np.array([0.493,0.493,0.503,0.535])
results_short74 = np.array([0.465,0.472,0.490,0.515])

meas_13pts = np.arange(0,325,25)
shortaxis_13pts = np.array([0.416,0.423,0.428,0.436,0.437,0.442,0.448,0.454,0.462,0.467,0.473,0.480,0.489])
longaxis_13pts = np.array([2.523,2.489,2.546,2.546,2.578,2.589,2.477,2.435,2.523,2.507,2.642,2.471,2.479])
#print(meas_13pts)

#Gaussian_beam_propagation(meas,results_short74,698e-6,plot=True)
Gaussian_beam_propagation(meas_13pts,longaxis_13pts,698e-6,plot=True)