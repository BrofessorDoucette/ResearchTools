import datetime
import numpy as np
import enum

class DIFFUSION_COEFFICIENT_TYPE_1D(enum.Enum):
    
    RBSP = 0
    RBSP_SCALED = 1
    OZEKE = 2
    BA = 3

def calculate_diffusion_coefficients(L, Kp, type : DIFFUSION_COEFFICIENT_TYPE_1D):
    
    '''Provide Kp(t) and L to obtain arrays of the necessary diffusion coefficients for 1-D simulations. Returns D_ll and dD_ll_dL in inverse days.'''
    
    match type:
        
        case DIFFUSION_COEFFICIENT_TYPE_1D.RBSP:
            
            #These parameters and expressions for Dll and dDll_dL are from this paper: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016JA023002
            
            a1 = -16.253
            b1 = 0.224
            a2 = -16.951
            b2 = 0.181
            c2 = 1.982

            D_ll_B = np.exp(a1 + b1 * np.outer(Kp, L) + L)
            D_ll_E = np.exp(a2 + b2 * np.outer(Kp, L) + c2 * L)

            D_ll = (D_ll_B + D_ll_E)
            dD_ll_dL = ((1 + b1 * Kp) * D_ll_B + (c2 + b2 * Kp) * D_ll_E)
            
            return D_ll, dD_ll_dL
        
        case DIFFUSION_COEFFICIENT_TYPE_1D.RBSP_SCALED:
            
            a1 = -16.253
            b1 = 0.224
            a2 = -16.951
            b2 = 0.181
            c2 = 1.982

            D_ll_B = np.exp(a1 + b1 * np.outer(Kp, L) + L) * 3.1
            D_ll_E = np.exp(a2 + b2 * np.outer(Kp, L) + c2 * L) * 3.8

            D_ll = (D_ll_B + D_ll_E)
            dD_ll_dL = ((1 + b1 * Kp) * D_ll_B + (c2 + b2 * Kp) * D_ll_E)
            
            return D_ll, dD_ll_dL
        
        case DIFFUSION_COEFFICIENT_TYPE_1D.OZEKE:
            
            #These parameters and expressions are from this paper: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JA019204
            
            a = 6.62e-13
            b = -0.0327
            c = 0.625
            d = -0.0108
            e = 0.499
            
            exp_factor_B = 10.0**(b*(L**2) + c*L + d*(Kp**2) + e*Kp)
            D_ll_B = a * (L**8) * exp_factor_B
            dD_ll_B_dL = 8 * a * (L**7) * exp_factor_B + a * (L**8) * np.log(10) * (2*b*L + c) * exp_factor_B
            
            f = 2.16e-8
            g = 0.217
            h = 0.461

            exp_factor_E = 10.0 ** (g * L + h * Kp)
            D_ll_E = f * (L**6) * 10.0**(g * L + h * Kp)
            dD_ll_E_dL = f * (L**5) * (6 + g * L * np.log(10)) * exp_factor_E
            
            D_ll = D_ll_B + D_ll_E
            
            dD_ll_dL = dD_ll_B_dL + dD_ll_E_dL

            return D_ll, dD_ll_dL
        
        case DIFFUSION_COEFFICIENT_TYPE_1D.BA:
            
            #These parameters and expressions can be found here: https://apps.dtic.mil/sti/tr/pdf/ADA507626.pdf.
            #Here we kept only the electromagnetic part. This is because thats what Scott did here: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JA028707
            
            a = 0.506
            b = -9.325
            
            D_ll_EM = (L**10) * 10 ** (a * Kp + b)
            dD_ll_EM_dL = (L ** 9) * 10 ** (1 + a * Kp + b)
            
            return D_ll_EM, dD_ll_EM_dL
            
        
    return None, None
    
    
def fokker_planck_1D_simulation(iterations_between_saves, dt, dL, L, f, min_BCs, max_BCs, D_ll, dD_ll_dL, numerical_stability_scaling_factor = 1e7):
    
    '''Simulates the 1-D fokker-planck equation. See the notebook simulating_radial_diffusion.ipynb for an example of how to use this.'''

    assert(len(min_BCs) == len(max_BCs))
    
    #We scale these for numerical stability reasons.. Otherwise the numbers are all really small.
    f = np.copy(f) * numerical_stability_scaling_factor
    min_BCs = min_BCs * numerical_stability_scaling_factor
    max_BCs = max_BCs * numerical_stability_scaling_factor
    
    L = np.copy(L) 
    f_S = np.zeros(shape=(0, len(L)))
    T_S = []
    for T in range(len(min_BCs) - 1):
        
        df_dL = (f[2:] - f[0:-2]) / (2 * dL) #Central Differation Approx
        d2f_dL2 = (f[2:] - 2 * f[1:-1] + f[0:-2]) / (dL**2)
        
        df_dt = (dD_ll_dL[T, 1:-1] - (2.0 / L[1:-1]) * D_ll[T, 1:-1]) * df_dL + D_ll[T, 1:-1] * d2f_dL2
        f[1:-1] += df_dt * dt
        f[0] = min_BCs[T + 1]
        f[-1] = max_BCs[T + 1]
        
        if (T % iterations_between_saves == 0) or (T == len(min_BCs) - 2):
            f_S = np.vstack((f_S, f / numerical_stability_scaling_factor))
            T_S.append(T)
    
    return L, f_S, np.asarray(T_S)