import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

'''
import plotly.tools
import plotly.graph_objs as go
import plotly.offline
'''

##### Template classes
class Model():
    def __init__(self):
        return
    def Run(self):
        raise NotImplementedError
    def Plot(self):
        raise NotImplementedError

##### Helper classes
class Fluid():
    def __init__(self, fluid_csv_name):
        curr_directory = os.getcwd()
        fluid_csv_path = os.path.join(curr_directory, "petronautics", "files", fluid_csv_name)
        input_df = pd.read_csv(fluid_csv_path)
        #print(input_df.to_string())
        #print(input_df.loc[input_df['Parameter'] == 'Water viscosity (cp)', 'Value'].values)
        self.mu_w = input_df.loc[input_df['Parameter'] == 'Water viscosity (cp)', 'Value'].values[0]
        #print(self.mu_w, type(self.mu_w))
        self.fvf_w = input_df.loc[input_df['Parameter'] == 'Water formation volume factor', 'Value'].values[0]
        self.c_w = input_df.loc[input_df['Parameter'] == 'Water compressibility (1/psi)', 'Value'].values[0]
        self.rho_w = input_df.loc[input_df['Parameter'] == 'Water density (kg/m3)', 'Value'].values[0]

        self.mu_o = input_df.loc[input_df['Parameter'] == 'Oil viscosity (cp)', 'Value'].values[0]
        self.fvf_o = input_df.loc[input_df['Parameter'] == 'Oil formation volume factor', 'Value'].values[0]
        self.rho_o = input_df.loc[input_df['Parameter'] == 'Oil density (kg/m3)', 'Value'].values[0]
        self.c_o = input_df.loc[input_df['Parameter'] == 'Oil compressibility (1/psi)', 'Value'].values[0]
        return
class Relperm():
    def __init__(self, relperm_csv_name):
        curr_directory = os.getcwd()
        relperm_csv_path = os.path.join(curr_directory, "petronautics", "files", relperm_csv_name)
        input_df = pd.read_csv(relperm_csv_path)
        #print(input_df.to_string())

        self.s_w_r = input_df.loc[input_df['Parameter'] == 'Swr', 'Value'].values[0]
        self.kr_w_endpoint = input_df.loc[input_df['Parameter'] == 'Krw0', 'Value'].values[0]
        self.n_w = input_df.loc[input_df['Parameter'] == 'Nw', 'Value'].values[0]

        self.s_o_r = input_df.loc[input_df['Parameter'] == 'Sor', 'Value'].values[0]
        self.kr_o_endpoint = input_df.loc[input_df['Parameter'] == 'Kro0', 'Value'].values[0]
        self.n_o = input_df.loc[input_df['Parameter'] == 'No', 'Value'].values[0]

        #raise ValueError('Relperm  not found!')
        return
    def CalcRelPerm(self, s_w):
        s_o = 1 - s_w
        
        s_w_normalized = (s_w - self.s_w_r)/(1 - self.s_w_r - self.s_o_r)
        s_w_clipped = np.clip(s_w_normalized, 0, 1)
        s_o_normalized = (s_o - self.s_o_r)/(1 - self.s_w_r - self.s_o_r)
        s_o_clipped = np.clip(s_o_normalized, 0, 1)

        kr_w = self.kr_w_endpoint*s_w_clipped**self.n_w
        kr_o = self.kr_o_endpoint*s_o_clipped**self.n_o
        return kr_w, kr_o
class GasDensity():
    def __init__(self):
        # Methane attributes
        self.acentricFactorCH4 = .008
        self.mwCH4 = 16.043 # g/mol
        self.tcCH4 = 190.6 # K
        self.pcCH4 = 4.6 * 1e6 # Pa
        self.gasConstantR = 8.314 # in Pa m^3 / mol K (Gas constant)
    def CalcGasDensity(self, pressure, temperature):
        # pressure in Pa
        # temperature in K
        # density in kg/m^3

        # Assumes the correct z compressibility factor is z1
        z = self._CalcPR78( pressure , temperature )
        #z = 1
        gasDensity = (pressure*self.mwCH4)/(z*self.gasConstantR*temperature)/1000
        return gasDensity
    def _CalcPR78(self, P, T):
        w = self.acentricFactorCH4
        Tc = self.tcCH4
        Pc = self.pcCH4
        R = self.gasConstantR
        
        if w <= 0.49:
            K = 0.37464 + 1.54226*w - 0.26992*w**2
        else:
            K = 0.37964 + w*(1.48503 + w*(-0.164423 + 0.01666*w))
        
        alpha = (1 + K*(1 - (T/Tc)**(1/2)))**2
        a = 0.457236*R**2*Tc**2*alpha/Pc
        b = 0.0778*R*Tc/Pc
        #da_dt = -0.45724*R**2*Tc**2/Pc*K*(alpha/(T*Tc))**(1/2)
        
        A = a*P/(R*T)**2
        B = b*P/R/T
        a0 = -A*B + B**2 + B**3
        a1 = A - 3*B**2 - 2*B
        a2 = -1 + B
        
        Q = (3*a1 - a2**2)/9
        R = (9*a2*a1 - 27*a0 - 2*a2**3)/54
        D = Q**3 + R**2
        S = (R + D**(1/2))**(1/3)
        T = (R - D**(1/2))**(1/3)
        
        if D < 0:
            theta = math.acos(R/((-(Q**3))**(1/2)))
            z1 = 2*((-Q)**(1/2))*math.cos(theta/3) - (1/3)*a2
            #z2 = 2*((-Q)**(1/2))*math.cos((theta + 2*pi)/3) - (1/3)*a2
            #z3 = 2*((-Q)**(1/2))*math.cos((theta + 4*pi)/3) - (1/3)*a2
        else:
            z1 = (-1/3)*a2 + (S + T)
            #z2 = (-1/3)*a2 - (1/2)*(S + T) + (1/2)*1i*3**(1/2)*(S - T)
            #z3 = (-1/3)*a2 - (1/2)*(S + T) - (1/2)*1i*3**(1/2)*(S - T)
        return z1
    
    def CalcFugacityCoefficient(self, Z, A, B):
        fugacityCoefficientPhi = math.exp( Z - 1 - math.log(Z - B) \
                                        - ( A / (math.sqrt(8) * B) ) \
                                            * math.log(  (Z + (1 + math.sqrt(2)) * B) \
                                                    /(Z + (1 - math.sqrt(2)) * B) \
                                                    ) \
        )
        return fugacityCoefficientPhi
class CapillaryPressure():
    def __init__(self, input_csv_name):
        curr_directory = os.getcwd()
        input_csv_path = os.path.join(curr_directory, "petronautics", "files", input_csv_name)
        self.df = pd.read_csv(input_csv_path)
        return
    def Calc(self, s_w):
        return
class HydrateFormation():
    def __init__(self):
        self.waterDensity = 1024        # kg H2O/m^3 H2O
        self.hydrateDensity = 928.5     # kg hydrate/m^3 hydrate
        self.mwCH4 = 16.043 / 1000      # kg CH4/mol CH4
        self.mwH2O = 18.01528 / 1000    # kg H2O/mol H2O
        self.methaneMassFractionInHydrate = 4*self.mwCH4/(4*self.mwCH4 + 23*self.mwH2O)
        return

##### Model classes
class BuckleyLeverett(Model):
    def __init__(self, input_csv_name):
        curr_directory = os.getcwd()
        input_csv_path = os.path.join(curr_directory, "petronautics", "files", input_csv_name)
        input_df = pd.read_csv(input_csv_path)
        #print(input_df.to_string())

        self.fluid = Fluid(input_df.loc[input_df['Parameter'] == 'Fluid', 'Value'].values[0])
        self.relperm = Relperm(input_df.loc[input_df['Parameter'] == 'Relperm', 'Value'].values[0])
        self.k = float(input_df.loc[input_df['Parameter'] == 'Permeability (micrometers^2)', 'Value'].values[0])
        self.delta_rho = self.fluid.rho_w - self.fluid.rho_o
        self.u = float(input_df.loc[input_df['Parameter'] == 'Velocity (cm/d)', 'Value'].values[0])
        self.alpha = float(input_df.loc[input_df['Parameter'] == 'Angle (deg)', 'Value'].values[0])
        self.krw = None
        self.kro = None
        self.t_d = float(input_df.loc[input_df['Parameter'] == 'tD (PV injected)', 'Value'].values[0])
        self.conversion_factor = 9.81/1e6**2/1000*100**4*1000*24*3600/1000
        self.result_dict = {}
        return
    def Run(self):
        ### Calc and set sw, krw, and kro
        self.s_w = np.linspace(
            start=self.relperm.s_w_r,
            stop=(1 - self.relperm.s_o_r),
            num=101,
        )
        self.kr_w, self.kr_o = self.relperm.CalcRelPerm(self.s_w)

        ### Calc fw
        self.f_w = self._CalcFractionalFlowWater(self.s_w)

        ### Get shock front saturation
        self.s_w_f, self.f_w_s_w_f, self.x_d = self._CalcShockFrontSaturation(self.s_w, self.f_w)
        #print(self.s_w_f, self.f_w_s_w_f, self.x_d)
        return
    def Plot(self):
        self._PlotRelperm()
        self._PlotWalshDiagram()
        plt.show()
        return
    
    ### Private methods
    def _CalcFractionalFlowWater(self, s_w):
        lambda_w = self.kr_w/self.fluid.mu_w
        lambda_o = self.kr_o/self.fluid.mu_o

        gravity_term = self.conversion_factor*self.k*self.delta_rho*np.sin(np.deg2rad(self.alpha))/self.u
        #print(type(gravity_term), gravity_term)
        f_w = lambda_w/(lambda_w + lambda_o)*(1 - gravity_term*lambda_o)
        return f_w
    def _CalcShockFrontSaturation(self, s_w, f_w):
        ################### haven't implemented a smarter swf calc for case 2
        case = 1
        if case == 1:
            secant = (f_w[1:] - f_w[0])/(s_w[1:] - s_w[0])
            secant = np.append(0, secant)

            tangent = (f_w[1:] - f_w[:-1])/(s_w[1:] - s_w[:-1])
            tangent = np.append(tangent, tangent[-1])

            min_difference_index = np.argmin(np.abs(secant - tangent))
            s_w_f = s_w[min_difference_index]
            f_w_s_w_f = f_w[min_difference_index]

            x_d = tangent*self.t_d
            shock_velocity = tangent[min_difference_index]
            x_d[s_w < s_w_f] = shock_velocity*self.t_d

            return s_w_f, f_w_s_w_f, x_d
        elif case == 2:



            s_w_f = (1 - self.relperm.s_o_r)*0.9
            eps = 1e-5
            tolerance = 1e-5

            for iteration in range(1, 20):
                tangent = (self._CalcFractionalFlowWater(s_w_f + eps) - self._CalcFractionalFlowWater(s_w_f))/eps
                secant = (self._CalcFractionalFlowWater(s_w_f) - self._CalcFractionalFlowWater(self.relperm.s_w_r)) \
                    /(s_w_f - self.relperm.s_w_r)
                f_x = tangent - secant
                
                if abs(f_x) < tolerance:
                    break


            else:
                print("Newton method failed to converge after " + str(iteration) + " iterations!")

            # f_x -> 0


            return s_w_f
    def _PlotRelperm(self):
        fig1, ax1 = plt.subplots(1, 1)
        x1 = self.s_w
        y1 = self.kr_w
        y2 = self.kr_o

        ax1.set_xlabel("Water saturation")
        ax1.set_ylabel("Relative permeability")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.plot(
            x1,
            y1,
            label='krw',
        )
        ax1.plot(
            x1,
            y2,
            label='kro',
        )
        ax1.legend()

        fig1.tight_layout()
        return
    def _PlotWalshDiagram(self):
        fig1, ax_array = plt.subplots(2, 2)
        #print(ax_array, type(ax_array))
        ax1, _, ax3, _ = ax_array.flatten()

        x1 = self.s_w
        y1 = self.f_w

        x2 = [self.s_w[0], self.s_w_f]
        y2 = [0, self.f_w_s_w_f]

        if self.x_d[0] < 1:
            x3 = [0, self.s_w[0]] + list(self.s_w)
            y3 = [1, 1] + list(self.x_d)
        else:
            x3 = [0]
            x3.extend(self.s_w)
            y3 = [1]
            y3.extend(self.x_d)

        ax1.set_xlabel("Water saturation")
        ax1.set_ylabel("Fractional flow")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.plot(
            x1,
            y1,
            #label='fw',
        )
        ax1.plot(
            x2,
            y2,
            #label='',
        )
        #ax1.legend()
        

        ax3.set_xlabel("Water saturation")
        ax3.set_ylabel("Dimensionless distance")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.fill_between(
            x3,
            y3,
            #label='',
        )

        fig1.tight_layout()
        return

class GasHydrateStability(Model):
    def __init__(self, parameter_list):
        return
    def Run(self):
        return
    def Plot(self):
        return



class ReservoirSimulation(Model):
    pass
class SeismicAnalysis(Model):
    pass
class Geomechanics(Model):
    pass
class ThermoFlash(Model):
    pass
class SoluteTransport(Model):
    pass
class LayerCake(Model):
    pass
class Koval(Model):
    pass
class CapacitanceResistance(Model):
    pass

