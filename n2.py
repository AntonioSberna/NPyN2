
import numpy as np






class Spectrum:
    def __init__(self, spectr_par):
        self.S = spectr_par["S"]
        self.eta = spectr_par["eta"]
        self.C_c = spectr_par["C_c"]
        self.ag = spectr_par["ag"]
        self.F0 = spectr_par["F0"]
        self.Tcstar = spectr_par["Tcstar"]

        # Periodi notevoli
        self.T_c = self.C_c*self.Tcstar
        self.T_b = self.T_c/3
        self.T_d = 4*self.ag+1.6


    def resp_spectrum(self, T):

        conditions = [
            T < self.T_b,
            (self.T_b <= T) & (T < self.T_c),
            (self.T_c <= T) & (T < self.T_d),
            T >= self.T_d
        ]
        

        Se_T_values = [
            self.ag * self.S * self.eta * self.F0 * ((T / self.T_b) + ((1 / self.F0) * (1 - T / self.T_b))),
            self.ag * self.S * self.eta * self.F0,
            self.ag * self.S * self.eta * self.F0 * (self.T_c / T),
            self.ag * self.S * self.eta * self.F0 * (self.T_c * self.T_d / T**2)
        ]
        

        return np.select(conditions, Se_T_values)
    

    def add_edge_periods(self, T):
        T_not = [self.T_b, self.T_c, self.T_d]
        pos = np.searchsorted(T, T_not)
        for pos, val in zip(pos, T_not):
            T = np.insert(T, pos, val)
        
        return T
    
    def constant_ductility_spectrum(self, T, mu):
        # Elastic spectrum
        Se_T = self.resp_spectrum(T)

        Se_i = np.zeros_like(Se_T)
        Se_i[T < self.T_c] = Se_T[T < self.T_c] / ((mu - 1) * (T[T < self.T_c] / self.T_c) + 1)
        Se_i[T >= self.T_c] = Se_T[T >= self.T_c] / mu

        return Se_i




    ###
    ### Plot
    ###
    def plot_spectrum(self, fig, ax, **kwargs):
        figkwargs = kwargs.get('figkwargs', dict())
        T = kwargs.get('T', np.linspace(0, self.T_d*1.5, 1000))

        # Add edge periods
        T = self.add_edge_periods(T)

        ax.plot(T, self.resp_spectrum(T), **figkwargs)
        ax.set_xlim([0, T[-1]])
        ax.set_ylim(0)

        return fig, ax
    

    def plot_elastic_adrs(self, fig, ax, **kwargs):
        figkwargs = kwargs.get('figkwargs', dict())
        T = kwargs.get('T', np.linspace(0, self.T_d*2, 1000))
        d_fact = kwargs.get('d_fact', 1)

        # Add edge periods
        T = self.add_edge_periods(T)

        Sa_e = self.resp_spectrum(T)
        Sd_e = 9.8095 * d_fact * Sa_e *(T**2) / (2*np.pi)**2

        ax.plot(Sd_e, Sa_e, **figkwargs)
        ax.set_xlim(0)
        ax.set_ylim(0)

        return fig, ax


    def plot_const_duct_adrs(self, fig, ax, mu, **kwargs):
        figkwargs = kwargs.get('figkwargs', dict())
        T = kwargs.get('T', np.linspace(0, self.T_d*2, 1000))
        d_fact = kwargs.get('d_fact', 1)

        # Add edge periods
        T = self.add_edge_periods(T)

        # # Constant ductility spectrum
        Sa_i = self.constant_ductility_spectrum(T, mu)
        Sd_i = 9.8095 * d_fact * Sa_i *(T**2) / (2*np.pi)**2

        ax.plot(Sd_i, Sa_i, **figkwargs)
        ax.set_xlim(0)
        ax.set_ylim(0)

        return fig, ax





class N2():
    def __init__(self, F, D, modalcoeff, mass, spectr, **kwargs):
        self.F = F
        self.D = D
        self.modalcoeff = modalcoeff
        self.mass = mass
        self.spectr = spectr
        self.coll_fact = kwargs.get('coll_fact', 0.85)

        # Picco della curva di capacità MDOF
        self.Fmax = max(F)
        # Picco curva SDOF
        self.f_bu = self.Fmax/modalcoeff

    def lin_interp(x_targ, x, y):
        return (y[0] + (x_targ-x[0])*(y[1]-y[0])/(x[1]-x[0]))

    def cut_capacity_curve(self):
        idx_max = int(np.where(self.F == self.Fmax)[0])
        idx_cut = idx_max + int(np.where(self.F[idx_max+1:]<self.coll_fact*self.Fmax)[0][0])+1 # the step after (for the interpolation)

        # SDOF curve
        if idx_cut < len(self.F):
            self.f_cut = np.append(self.F[:idx_cut], self.coll_fact*self.Fmax)
            self.d_cut = np.append(self.D[:idx_cut], np.interp(self.coll_fact*self.Fmax, self.F[idx_cut-1:idx_cut+1], self.D[idx_cut-1:idx_cut+1]))
        else:
            self.f_cut = self.F
            self.d_cut = self.D
        pass

    def from_MDOF_to_SDOF(self):
        # SDOF curve peak force value
        self.f_sdof = self.f_cut/self.modalcoeff
        self.d_sdof = self.d_cut/self.modalcoeff
        pass


    def bilinearize(self):
        # Index position displacement at 0.6*Fbu
        index06= np.where(self.f_sdof>0.6*self.f_bu)[0][0]
        # Displacement value at 0.6*f_bu
        self.d06fbu =  np.interp(0.6*self.f_bu, self.f_sdof[index06-1:index06+1], self.d_sdof[index06-1:index06+1])

        # Equivalent stiffness
        self.k = self.f_bu * 0.6 / self.d06fbu #[N/mm]

        # Area under the curve
        area = np.trapz(self.f_sdof, self.d_sdof)
        # Area of bilinear up to 0.6*f_bu
        area_bilin = (self.d06fbu*self.f_bu*0.6/2)+ (self.d_sdof[-1] - self.d06fbu)*self.f_bu*0.6


        ### Looking for the f_y value in order to have same area under the curves
        # Initialization of d_y and f_y
        d_y = self.d06fbu
        f_y = 0.6*self.f_bu
        steps = 0.001

        while area_bilin < area and d_y<self.d_sdof[-1]: #bruto, in un altro codice l'avevo fatto più veloce, adesso lo cerco 
            d_y = d_y+steps
            f_y = self.k * d_y
            area_bilin = (d_y*f_y/2) + (self.d_sdof[-1]-d_y)*f_y

        self.d_y = d_y
        self.f_y = f_y
        pass


    def run_analysis(self):
        self.cut_capacity_curve()
        self.from_MDOF_to_SDOF()
        self.bilinearize()
        pass
    

    
    def ductility_check(self):
        # Ductility capacity
        mu_c = self.d_sdof[-1]/self.d_y    

        # Equivalent first modal period
        T = 2*np.pi*np.sqrt(self.mass/self.k) #[sec]
        
        # Elastic spectral acceleration
        Se_T = self.spectr.resp_spectrum(T)
        

        # Reduction factor
        q = Se_T*self.mass*1000/self.f_y

        # Ductility demand (Vidic et al. 1994)
        if T<self.spectr.T_c:
            mu_d = (q-1)*(self.spectr.T_c/T)+1 
        else:
            mu_d = q


        return mu_c, mu_d