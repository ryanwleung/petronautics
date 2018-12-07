import tkinter as tk
import tkinter.font as font
import tkinter.messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.backends.backend_tkagg
import matplotlib.figure
import matplotlib.backends.backend_agg

import numpy as np
import models.petroleumengineering

class Controller(object):
    def __init__(self):
        #self.fonts = {}
        #self.fonts['title'] = tk.font.Font(family='Helvetica', size=18, weight='bold', slant='italic')
        self.view = View(self)
        self.model = None        
        self.current_results = None
        return
    def Start(self):
        self.view.Initialize()
        self.view.root.mainloop() 
        return
    ### Methods that View pages will call
    def ChangePage(self, page_name):
        self.view.GridPage(page_name)
        if page_name == 'BuckleyLeverett':
            #import models.petroleumengineering
            ### code when BL was in petroleum engineering module
            #self.model = models.petroleumengineering.BuckleyLeverett()
            ### code after migration to here for compatibility
            self.model = BuckleyLeverettModel()
        return
    def SendInputToModel(self, *, input_dict, notification_stringvar):
        result_dict = None
        data_is_correct, notification = self.model.ProcessInput(input_dict)
        notification_stringvar.set(notification)
        if data_is_correct:
            parameter_dict = self.model.FormatInputIntoParameters(input_dict)
            self.model.SetParameters(parameter_dict)
            result_dict = self.GetCurrentResults()
        return data_is_correct, result_dict
    def RunModel(self):
        self.model.Run()
        return
    def GetCurrentResults(self):
        return self.model.GetResults()
class Model(object):
    ### Abstract methods
    def __init__(self):
        self.result_dict = {}
        return
    def ProcessInput(self, input_dict):
        raise NotImplementedError
    def FormatInputIntoParameters(self, input_dict):
        raise NotImplementedError
    def SetParameters(self, parameter_dict):
        raise NotImplementedError
    def Run(self):
        raise NotImplementedError
    def GetResults(self):
        return self.result_dict.copy()
class View(object):
    def __init__(self, controller):
        self.root = tk.Tk()
        self.root.configure(background='black')
        self.root.title('Engineering Toolbox')
        
        self.controller = controller
        self.controller.font_dict = self.ReturnFonts()
        
        self.container = tk.Frame(master=self.root)
        self.container.grid(row=0, column=0, padx=5, pady=5)
        self.container.configure(background='blue')
        
        self.page_dict = {
            'BuckleyLeverett': View.BuckleyLeverett,
            'Template': View.Template
        }

        self.active_page = None
        
        return
    def ReturnFonts(self):
        font_dict = {}
        #font_dict['title'] = tk.font.Font(family='Helvetica', size=18, weight='bold', slant='italic')
        font_dict['title'] = tk.font.Font(font='TkHeadingFont')
        font_dict['menu'] = tk.font.Font(font='TkMenuFont')

        #for value in font_dict.values():
        #    print(value.name)
        return font_dict    
    def GridPage(self, page_name):
        self.active_page.DeactivateGrid()
        self.active_page = self.page_dict[page_name](parent=self.container, controller=self.controller)
        self.active_page.ActivateGrid()
        return
    def Initialize(self):
        self.active_page = View.Home(parent=self.container, controller=self.controller)
        self.active_page.ActivateGrid()
    class Page(object): 
        def __init__(self, *, parent, controller):
            self.controller = controller
            self.frame = tk.Frame(master=parent)
            return
        def ActivateGrid(self):
            self.frame.grid(row=0, column=0, padx=5, pady=5)
            return
        def DeactivateGrid(self):
            self.frame.grid_remove()
            return
    class Home(Page):
        def __init__(self, parent, controller):
            View.Page.__init__(self, parent=parent, controller=controller)

            self.label = tk.Label(master=self.frame, text="Home")
            self.label.grid(row=0)

            self.nav_bar = self.NavBar(parent=self.frame, controller=self.controller)
            self.nav_bar.frame.grid(row=1)
            return
        class NavBar(object):
            def __init__(self, *, parent, controller):
                self.controller = controller
                self.frame = tk.Frame(master=parent)

                self.nav_button_dict = {
                    'BuckleyLeverett': 'Buckley-Leverett',
                    'Template': 'WIP'
                }
                self.nav_button_list = []
                for key, value in self.nav_button_dict.items():
                    nav_button = self.NavButton(parent=self.frame, controller=self.controller, name=key, text=value)
                    self.nav_button_list.append(nav_button)
                for nav_button_index in range(len(self.nav_button_list)):
                    # start row_index at 1 because the header is at 0
                    row_index = nav_button_index + 1
                    self.nav_button_list[nav_button_index].button.grid(row=row_index)
                return
            class NavButton(object):
                def __init__(self, *, parent, controller, name, text):
                    self.name = name
                    self.controller = controller
                    self.text = text
                    self.button = tk.Button(master=parent, command=self.CallBack, text=self.text)
                    return
                def CallBack(self):
                    self.controller.ChangePage(self.name)

                    return
    class BuckleyLeverett(Page):
        def __init__(self, parent, controller):
            View.Page.__init__(self, parent=parent, controller=controller)

            # Member variables
            self.is_ready = False
            self.result_dict = {}

            # Base frame
            self.label = tk.Label(master=self.frame, text="Buckley Leverett Theory")
            self.label.grid(row=0)
            
            # Entry table
            self.entry_table = View.FrameObject(parent=self.frame, controller=controller)
            self.entry_table.label = tk.Label(master=self.entry_table.frame, text="Input Table")
            self.entry_table.label.grid(row=0, column=0, columnspan=2)
            
            self.entry_table.input_dict = {}
            self.entry_table.insert_default_button = tk.Button(
                master=self.entry_table.frame, text="Insert default values", command=self.CommandInsertDefault
            )
            self.entry_table.save_input_button = tk.Button(
                master=self.entry_table.frame, text="Save inputs", command=self.CommandSaveInput
            )
            self.entry_table.run_button = tk.Button(master=self.entry_table.frame, text="Run model", command=self.CommandRun)
            self.entry_table.notification_stringvar = tk.StringVar()
            self.entry_table.notification_label = tk.Label(
                master=self.entry_table.frame, textvariable=self.entry_table.notification_stringvar
            )
            self.LoadEntryTable()

            # Result box
            self.result_box = View.PlotObject(
                parent=self.frame,
                controller=controller,
                width=400,
                height=300,
            )            
            self.result_box.label.config(text="Results")
            self.result_box.label.grid(row=0, column=0, columnspan=2)
            self.result_box.canvas.grid(row=1, column=0, columnspan=2, padx=0, pady=0)
            
            self.result_box.box_dict = {}
            self.result_box.td_button = tk.Button(master=self.result_box.frame, text="Apply td", command=self)
            self.LoadResultBox()

            # Plot box
            self.plot_box = View.PlotObject(
                parent=self.frame,
                controller=controller,
                width=640,
                height=480,
            )            
            self.plot_box.label.config(text="Plots")
            self.plot_box.label.grid(row=0, column=0, columnspan=2)
            self.plot_box.canvas.grid()
            
            self.plot_box.figure.clear()
            self.plot_box.axes = self.plot_box.figure.subplots(nrows=2, ncols=2)
            return
        
        ### Load methods
        def LoadEntryTable(self):
            self.entry_table.input_dict = {
                's_w_r': {'text': "Residual water saturation", 'default': 0.2},
                's_o_r': {'text': "Residual oil saturation", 'default': 0.2},
                'kr_w_endpoint': {'text': "Endpoint krw", 'default': 1},
                'kr_o_endpoint': {'text': "Endpoint kro", 'default': 1},
                'n_w': {'text': "Nw (water exponent)", 'default': 2},
                'n_o': {'text': "No (oil exponent)", 'default': 2},
                'mu_w': {'text': "Water viscosity (cp)", 'default': 1},
                'mu_o': {'text': "Oil viscosity (cp)", 'default': 10},
                'k': {'text': "Permeability (micro m^2)", 'default': 0.5},
                'delta_rho': {'text': "Delta density (g/cm^3)", 'default': 0.2},
                'u': {'text': "Velocity (cm/d)", 'default': 0.6},
                'alpha': {'text': "Angle (degrees)", 'default': 0},
                't_d': {'text': "td (pore volumes injected)", 'default': 0.5},
            }
            entry_index = 1
            for key in self.entry_table.input_dict.keys():
                self.entry_table.input_dict[key]['label'] = tk.Label(
                    master=self.entry_table.frame, text=self.entry_table.input_dict[key]['text']
                )
                self.entry_table.input_dict[key]['label'].grid(row=entry_index, column=0)
                self.entry_table.input_dict[key]['entry'] = tk.Entry(master=self.entry_table.frame)
                self.entry_table.input_dict[key]['entry'].grid(row=entry_index, column=1)
                self.entry_table.input_dict[key]['input'] = tk.DoubleVar()
                entry_index += 1
            self.entry_table.insert_default_button.grid(row=entry_index, column=0, columnspan=2)
            entry_index += 1
            self.entry_table.save_input_button.grid(row=entry_index, column=0, columnspan=2)
            entry_index += 1
            self.entry_table.run_button.grid(row=entry_index, column=0, columnspan=2)
            entry_index += 1
            self.entry_table.notification_label.grid(row=entry_index, column=0, columnspan=2)
            return
        def LoadResultBox(self):
            self.result_box.box_dict = {
                's_w_f': {'text': "Shock front saturation"},
            }
            box_index = 2
            for key in self.result_box.box_dict.keys():
                self.result_box.box_dict[key]['label'] = tk.Label(
                    master=self.result_box.frame, text=self.result_box.box_dict[key]['text']
                )
                self.result_box.box_dict[key]['label'].grid(row=box_index, column=0)
                self.result_box.box_dict[key]['entry'] = tk.Entry(master=self.result_box.frame)
                self.result_box.box_dict[key]['entry'].grid(row=box_index, column=1)
                box_index += 1
            ### NEED TO CHANGE COMMAND OF BUTTON for td calc
            #self.result_box.td_button.grid(row=box_index, column=0, columnspan=2)
            #box_index += 1
            return

        ### Override methods
        def ActivateGrid(self):
            View.Page.ActivateGrid(self)
            self.entry_table.frame.grid(row=0, column=0)
            return

        ### Plot methods
        def PlotRelperm(self):
            self.result_box.axes.clear()
            self.result_box.axes.plot(self.result_dict['s_w'], self.result_dict['kr_w'])
            self.result_box.axes.plot(self.result_dict['s_w'], self.result_dict['kr_o'])
            self.result_box.axes.set_xlabel('Water saturation')
            self.result_box.axes.set_ylabel('Relative Permeability')
            self.result_box.axes.set_xlim(left=0, right=1)
            self.result_box.axes.set_ylim(bottom=0, top=1)
            self.result_box.figure.set_tight_layout(True)
            self.result_box.BlitDraw()
            return
        def PlotWalshDiagrams(self):
            # Sw vs Fw in top left plot
            self.plot_box.axes[0, 0].clear()
            self.plot_box.axes[0, 0].plot(self.result_dict['s_w'], self.result_dict['f_w'])
            # Shock front line
            x_1 = self.result_dict['s_w'][0]
            y_1 = self.result_dict['f_w'][0]
            x_2 = self.result_dict['s_w_f']
            y_2 = self.result_dict['f_w_s_w_f']
            slope = (y_2 - y_1)/(x_2 - x_1)
            x_3 = 1/slope + x_1
            y_3 = 1
            x_shock = [x_1, x_3]
            y_shock = [y_1, y_3]
            self.plot_box.axes[0, 0].plot(x_shock, y_shock)
            self.plot_box.axes[0, 0].set_xlabel('Water saturation')
            self.plot_box.axes[0, 0].set_ylabel('Water fractional flow')
            self.plot_box.axes[0, 0].set_xlim(left=0, right=1)
            self.plot_box.axes[0, 0].set_ylim(bottom=0, top=1)


            # x_d vs Sw in bottom left plot
            self.plot_box.axes[1, 0].clear()
            x_plot_1_0 = np.insert(self.result_dict['s_w'], 0, self.result_dict['s_w'][0])
            y_plot_1_0 = np.insert(self.result_dict['x_d'], 0, 1)
            
            self.plot_box.axes[1, 0].plot(x_plot_1_0, y_plot_1_0)
            self.plot_box.axes[1, 0].set_ylabel('Dimensionless distance')
            self.plot_box.axes[1, 0].set_xlabel('Water saturation')
            self.plot_box.axes[1, 0].set_xlim(left=0, right=1)
            self.plot_box.axes[1, 0].set_ylim(bottom=0, top=1)


            self.plot_box.figure.set_tight_layout(True)
            self.plot_box.BlitDraw()
            return

        ### Entry table methods
        def CommandInsertDefault(self):
            self.entry_table.notification_stringvar.set("")
            for key in self.entry_table.input_dict.keys():
                self.entry_table.input_dict[key]['entry'].delete(0, 'end')
                self.entry_table.input_dict[key]['entry'].insert(0, self.entry_table.input_dict[key]['default'])
            return
        def CommandSaveInput(self):
            self.is_ready, self.result_dict = self.controller.SendInputToModel(
                input_dict=self.entry_table.input_dict,
                notification_stringvar=self.entry_table.notification_stringvar
            )
            if self.is_ready:
                self.PlotRelperm()
                self.result_box.frame.grid(row=1, column=0)
            return
        def CommandRun(self):
            if self.is_ready:
                self.plot_box.frame.grid(row=0, column=1, rowspan=2, columnspan=2)

                self.controller.RunModel()
                self.result_dict = self.controller.GetCurrentResults()

                self.PlotWalshDiagrams()
            else:
                tk.messagebox.showerror(title="Input Error", message="Input values must be corrected and saved before running the model.")
    class Template(Page):
        def __init__(self, parent, controller):
            View.Page.__init__(self, parent=parent, controller=controller)
            
            self.label = tk.Label(master=self.frame, text="Work in progress")
            self.label.grid(row=0)
            return


    class FrameObject(object):
        def __init__(self, *, parent, controller):
            self.controller = controller
            self.frame = tk.Frame(master=parent)
            self.label = tk.Label(master=self.frame)

    class PlotObject(object):
        def __init__(self, *, parent, controller, width, height):
            self.controller = controller
            self.frame = tk.Frame(master=parent)
            self.label = tk.Label(master=self.frame)
            self.figure = matplotlib.figure.Figure()
            width_in = width / self.figure.get_dpi()
            height_in = height / self.figure.get_dpi()
            self.figure.set_size_inches(width_in, height_in)
            
            self.axes = self.figure.add_subplot(111)
            self.canvas_agg = matplotlib.backends.backend_agg.FigureCanvasAgg(
                figure=self.figure,
            )
            
            self.renderer = self.canvas_agg.get_renderer()._renderer
            self.canvas = tk.Canvas(master=self.frame, width=width, height=height)
            self.photo = tk.PhotoImage(master=self.canvas)
            self.canvas.create_image(width / 2, height / 2, image=self.photo)
            return
        def BlitDraw(self):
            self.canvas_agg.draw()
            matplotlib.backends.tkagg.blit(self.photo, self.renderer, colormode=2)
            return


class BuckleyLeverettModel(Model):
    def __init__(self):
        self.fluid = None
        self.relperm = None
        self.k = None
        self.delta_rho = None
        self.u = None
        self.alpha = None
        self.krw = None
        self.kro = None
        self.t_d = None
        self.conversion_factor = 9.81/1e6**2/1000*100**4*1000*24*3600
        self.result_dict = {}
        return
    
    ### Override methods
    def ProcessInput(self, input_dict):
        input_is_correct = True
        for key in input_dict.keys():
            myEntry = input_dict[key]['entry']
            
            uncheckedInput = myEntry.get()
            checkedInput = None
            error_message = None

            try:
                checkedInput = float(uncheckedInput)
            except ValueError:
                error_message = "Error: the input for " + input_dict[key]['text'] + " should be a number."
                tk.messagebox.showerror(title="Input Error", message=error_message)
                input_is_correct = False
                break
            else:
                verifiedInput, verify_message = self.VerifyValidInput(key=key, value=checkedInput)
                if verifiedInput is not None:
                    input_dict[key]['input'].set(verifiedInput)
                else:
                    error_message = "Error: the input for " + input_dict[key]['text'] + " should be " + verify_message + "."
                    tk.messagebox.showerror(title="Input Error", message=error_message)
                    input_is_correct = False
                    break
        if input_is_correct:
            notification = "User input values saved and model ready to run!"
        else:
            notification = "Please input all values correctly and save again!"
        return input_is_correct, notification
    def FormatInputIntoParameters(self, input_dict):
        fluid_dict = {
            'mu_w': input_dict['mu_w']['input'].get(),
            'mu_o': input_dict['mu_o']['input'].get(),
        }
        relperm_dict = {
                'model': 'BrooksCorey',
                's_w_r': input_dict['s_w_r']['input'].get(),
                's_o_r': input_dict['s_o_r']['input'].get(),
                'kr_w_endpoint': input_dict['kr_w_endpoint']['input'].get(),
                'kr_o_endpoint': input_dict['kr_o_endpoint']['input'].get(),
                'n_w': input_dict['n_w']['input'].get(),
                'n_o': input_dict['n_o']['input'].get(),
        }
        buckley_leverett_dict = {
            'fluid': Fluid(fluid_dict),
            'relperm': Relperm(relperm_dict),
            'k': input_dict['k']['input'].get(),
            'delta_rho': input_dict['delta_rho']['input'].get(),
            'u': input_dict['u']['input'].get(),
            'alpha': input_dict['alpha']['input'].get(),
            't_d': input_dict['t_d']['input'].get()
        }
        return buckley_leverett_dict
    def SetParameters(self, parameter_dict):
        # Set parameters from dict
        self.fluid = parameter_dict['fluid']
        self.relperm = parameter_dict['relperm']
        self.k = parameter_dict['k']
        self.delta_rho = parameter_dict['delta_rho']
        self.u = parameter_dict['u']
        self.alpha = parameter_dict['alpha']
        self.t_d = parameter_dict['t_d']
        # Calc and set sw, krw, and kro
        self.s_w = np.linspace(
            start=self.relperm.s_w_r,
            stop=(1 - self.relperm.s_o_r),
            num=101,
        )
        self.kr_w, self.kr_o = self.relperm.CalcRelPerm(self.s_w)
        self.result_dict['s_w'] = self.s_w.copy()
        self.result_dict['kr_w'] = self.kr_w.copy()
        self.result_dict['kr_o'] = self.kr_o.copy()
        return
    def Run(self):
        s_w = self.s_w.copy()
        f_w = self.CalcFractionalFlowWater(s_w)

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


        self.result_dict['f_w'] = f_w
        self.result_dict['s_w_f'] = s_w_f
        self.result_dict['f_w_s_w_f'] = f_w_s_w_f
        self.result_dict['x_d'] = x_d
        return
    
    def VerifyValidInput(self, *, key, value):
        verify_message = '',
        keys_between_0_and_1 = [
            's_w_r',
            's_o_r',
            'kr_w_endpoint',
            'kr_o_endpoint'
        ]
        keys_greater_than_0 = [
            'n_w',
            'n_o',
            'mu_w',
            'mu_o',
            'k',
            'delta_rho',
            'u',
            't_d',
        ]
        # 1st block identifies what the value limits are for a given key
        #   if given key doesn't have specified limits, return value and empty verify_message
        # 2nd block checks value against limits
        #   if value is within limits, fast return value and empty verify_message
        #   if value is not in limits, set appropiate verify_message for user
        #       exit 1st and 2nd block and return None, verify_message       
        if key in keys_between_0_and_1:
            if value >= 0 and value <= 1:
                return value, verify_message
            else:
                verify_message = "between 0 and 1"
        elif key in keys_greater_than_0:
            if value > 0:
                return value, verify_message
            else:
                verify_message = "greater than 0"
        else:
            return value, verify_message
        return None, verify_message
    
    def CalcFractionalFlowWater(self, s_w):
        lambda_w = self.kr_w/self.fluid.mu_w
        lambda_o = self.kr_o/self.fluid.mu_o
        f_w = lambda_w/(lambda_w + lambda_o) \
            *(1 - self.conversion_factor*self.k*lambda_o*self.delta_rho*np.sin(np.deg2rad(self.alpha))/self.u)
        return f_w
    def CalcShockFrontSaturation(self):
        # NOT DONE, currently using simple implementation in Run function
        s_w_f = (1 - self.relperm.s_o_r)*0.9
        eps = 1e-5
        tolerance = 1e-5

        for iteration in range(1, 20):
            tangent = (self.CalcFractionalFlowWater(s_w_f + eps) - self.CalcFractionalFlowWater(s_w_f))/eps
            secant = (self.CalcFractionalFlowWater(s_w_f) - self.CalcFractionalFlowWater(self.relperm.s_w_r)) \
                /(s_w_f - self.relperm.s_w_r)
            f_x = tangent - secant
            
            if abs(f_x) < tolerance:
                break


        else:
            print("Newton method failed to converge after " + str(iteration) + " iterations!")

        # f_x -> 0


        return s_w_f
class Fluid(object):
    def __init__(self, fluid_dict):
        fluid_keys = fluid_dict.keys()
        if 'mu_w' in fluid_keys:
            self.mu_w = fluid_dict['mu_w']
        if 'fvf_w' in fluid_keys:
            self.fvf_w = fluid_dict['fvf_w']
        if 'rho_w' in fluid_keys:
            self.rho_w = fluid_dict['rho_w']
        if 'c_w' in fluid_keys:
            self.c_w = fluid_dict['c_w']

        if 'mu_o' in fluid_keys:
            self.mu_o = fluid_dict['mu_o']
        if 'fvf_o' in fluid_keys:
            self.fvf_o = fluid_dict['fvf_o']
        if 'rho_o' in fluid_keys:
            self.rho_o = fluid_dict['rho_o']
        if 'c_o' in fluid_keys:
            self.c_o = fluid_dict['c_o']
        return
class Relperm(object):
    def __init__(self, relperm_dict):
        if relperm_dict['model'] == 'BrooksCorey':
            self.s_w_r = relperm_dict['s_w_r']
            self.kr_w_endpoint = relperm_dict['kr_w_endpoint']
            self.n_w = relperm_dict['n_w']

            self.s_o_r = relperm_dict['s_o_r']
            self.kr_o_endpoint = relperm_dict['kr_o_endpoint']
            self.n_o = relperm_dict['n_o']
        else:
            raise ValueError('Relperm model not found!')
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
