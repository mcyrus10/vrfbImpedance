#!/Users/cyrus/miniconda3/bin/python3
#------------------------------------------------------------------------------
# Written by Michael C. Daugherty
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from numpy import array,pi,arctan,sinh,exp,sqrt,tanh,concatenate,real,imag,inf
from scipy.optimize import least_squares
#______________________________________________________________________________
#   Functions
#______________________________________________________________________________
def single_electrode(   frequency = 0,
                        fitted_params = 0,
                        fixed_params = 0):
    #{{{
    coth = lambda x: (exp(x)+exp(-x))/(exp(x)-exp(-x))
    R = 8.3144598
    F = 96485.33289
    # fixed params unpack
    T = fixed_params['T']               # Temperature (K)
    A = fixed_params['A']               # Geometric surface area (cm2)
    b = fixed_params['b']               # Electrode Thickness (cm)
    rho1 = fixed_params['rho1']         # ionic specific resistivity (ohm * cm)
    rho2 = fixed_params['rho2']         # electrode specification?? (ohm * cm)
    alpha = fixed_params['alpha']       # transfer coefficient
    C_R = fixed_params['C_R']           # Red concentration (mol cm^-3?)
    D_R = fixed_params['D_R']           # Red diffusivity (cm^2 s^-1)
    C_O = fixed_params['C_O']           # Ox concentration (mol cm^-3?)
    D_O = fixed_params['D_O']           # Ox diffusivity (cm^2 s^-1)
    C_dl = fixed_params['C_dl']         # specific double layer capacitance (F cm^-2)
    n = fixed_params['n']               # number of electrons
    #fitted params unpack
    At = fitted_params['At']            # total surface area (cm^2)
    i0 = fitted_params['i0']            # exchange current density (A cm^-2)
    P = fitted_params['P']              # CPE power factor (dimensionless)
    ASR_mem = fitted_params['ASR_mem']  # Membrane resistance (ohm)
    L = fitted_params['L']              # lead inductance (H)
    a = fitted_params['a']              # boundary layer thickness (cm)
    Af = fitted_params['Af']            # fitting parameter for diffusion (dimensionless)
     
    omega = 2*pi*frequency
    # Charge transfer resistance
    R_ct = R*T/(n*F*i0)

    # Warburg element
    gamma_R = R*T/(n**2*F**2*C_R)
    gamma_O = R*T/(n**2*F**2*C_O)
    term1 = (gamma_R*a/(Af*D_R))*tanh(a*sqrt(1j*omega/D_R))/(a*sqrt(1j*omega/D_R))
    term2 = (gamma_O*a/(Af*D_O))*tanh(a*sqrt(1j*omega/D_O))/(a*sqrt(1j*omega/D_O))

    Z_a_prime = R_ct + term1 + term2
    Z_a = 1/(1/Z_a_prime+(1j*omega)**P*C_dl)

    AZ_p = A*Z_a/At

    return(AZ_p)
    #}}}
def whole_cell(         fitted_params = 0,
                        fixed_params = 0,
                        frequency = 0):
    #{{{
    ASR_mem = fitted_params['ASR_mem']
    L = fitted_params['L']
    return(2*single_electrode(fitted_params = fitted_params,fixed_params =
        fixed_params,frequency = frequency)
        +ASR_mem+L*1j*pi*frequency)
    #}}}
def FIT(                fitted_params=0,
                        fixed_params=0,
                        ReZ = 0,
                        ImZ = 0,
                        frequency = 0, 
                        area= 5,
                        figno = 0,
                        algorithm = 'least_squares',
                        distributed = True,
                        circuit = 'Transmission',
                        fre_init = 0): 
    # {{{ 
    R = 8.3144598
    F = 96485.33289
    #--------------------------------------------------------------------------
    # algorithms =
    # ['leastsq','least_squares','nelder','lbfgsb','powell','cg','cobyla']
    #--------------------------------------------------------------------------
    T = fixed_params['T'];          
    b = fixed_params['b']
    A = fixed_params['A'];          
    rho1 = fixed_params['rho1']
    rho2 = fixed_params['rho2'];    
    alpha = fixed_params['alpha']
    C_R = fixed_params['C_R'];      
    C_O = fixed_params['C_O']
    D_R = fixed_params['D_R'];      
    D_O = fixed_params['D_O']
    C_dl = fixed_params['C_dl'];    
    n = fixed_params['n']
    # Real, Imaginary and Frequency are the values that define the fit!
    #--------------------------------------------------------------------------
    rez = ReZ if circuit == 'Transmission' else ReZ[fre_init:]
    imz = ImZ if circuit == 'Transmission' else ImZ[fre_init:]
    frequency = frequency if circuit == 'Transmission' else frequency[fre_init:]
    data = (rez-1j*imz)*area # ASR!!!!!!
    def residual(   x0,
                    frequency,
                    data):
        #----------------------------------------------------------------------
        # Residual is the function that is minimized in the 'minimize' function
        #----------------------------------------------------------------------
        # UNPACK THE ORDERED ARRAY X0!!
        #----------------------------------------------------------------------
        fp_temp =   {
                    'At':x0[0],
                    'i0':x0[1],
                    'P':x0[2],
                    'ASR_mem':x0[3],
                    'L':x0[4],
                    'a':x0[5],
                    'Af':x0[6]
                    }

        mod_l = whole_cell( fitted_params = fp_temp,
                            fixed_params = fixed_params,
                            frequency = frequency)
        Real = real(data)-real(mod_l)
        Imag = imag(data)-imag(mod_l)
        return(concatenate([Real,Imag]))
    # minimize is the function that minimizes the residual function and answer
    # is the output; fp_fitted is then populated with the output of the
    # minimize function
    bounds_lower = [0,0,0,0,0,0,0]
    bounds_upper = [inf,inf,1,inf,inf,inf,inf]
    keys = ['At','i0','P','ASR_mem','L','a','Af']
    x0 = [fitted_params[key] for key in keys]
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    out = least_squares(residual, x0 , bounds =(bounds_lower,bounds_upper),
            method = 'trf', args = (frequency, data))
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    fp_fitted = {}
    for i,key in enumerate(keys):
        fp_fitted[key] = out.x[i]

    temp = whole_cell(fitted_params = fp_fitted,fixed_params = fixed_params, frequency= frequency)
    # Calculations of resistances page 263-264 in Eludidating
    R_membrane = fp_fitted['ASR_mem']
    Rct = (A*R*T)/(fp_fitted['At']*F*fp_fitted['i0'])
    Rfd = (A*fp_fitted['a']*R*T)/(fp_fitted['At']*fp_fitted['Af']*F**2) *\
            (1/(fixed_params['D_R']*fixed_params['C_R']) +
            1/(fixed_params['C_O']*fixed_params['D_O']))
    R_distributed = b*(rho1+rho2)/3+b*rho1*rho2/(3*rho1+3*rho2) if distributed else 0
    Re = fp_fitted['ASR_mem']/2+R_distributed
    #--------------------------------------------------------------------------
    # NOTE THAT THE A IN THE DENOMINATOR FROM ALAN'S ELUCIDATING PAPER WAS
    # DELETED HERE
    #--------------------------------------------------------------------------
    Q = fp_fitted['At']*C_dl
    CPE_dl = Q**(1/fp_fitted['P'])*((Re*Rct)/(Re+Rct))**(1/fp_fitted['P']-1)
    fitting_output =    {
                        'Rct':Rct*2,
                        'Rfd':Rfd*2,
                        'R_distributed':R_distributed,
                        'R_ohmic':Re*2,
                        'CPE_dl': CPE_dl,
                        'Aeff':CPE_dl/C_dl
                        }
    out['fit'] = fp_fitted
    return(fp_fitted,fitting_output)
    # }}}
#______________________________________________________________________________
#   Procedure
#______________________________________________________________________________
# {{{ Raw Data (untreated GFD3 Electrode; .5 M V; 5 cm^2 Aspect Ratio
ReZ = array([ 0.08284266, 0.08796988, 0.09247773, 0.09680536, 0.1008338, 0.1046377,
 0.10762515, 0.11097036, 0.11456104, 0.11730141, 0.12102044, 0.12483983,
 0.12752137, 0.13284937, 0.13733491, 0.14415036, 0.15262745, 0.16596687,
 0.18005887, 0.20618707, 0.23595014, 0.28682289, 0.35581028, 0.4261204,
 0.56032306, 0.66784477, 0.80271685, 0.94683707, 1.0815266 , 1.1962512,
 1.3139805, 1.4123734, 1.4371094, 1.4075497, 1.451781 , 1.4999349,
 1.4951819, 1.520785, 1.5343723, 1.5442845, 1.5559914, 1.5715505,
 1.581386, 1.6037546, 1.6127067, 1.6335728, 1.6475986 , 1.6718234,
 1.6902045, 1.7113601, 1.7273785, 1.7500663, 1.7663705 , 1.7867819,
 1.8013573, 1.8191988, 1.83577, 1.8508064, 1.8635432 , 1.8764733 ])

ImZ = array([
 0.01176712, 0.01655927, 0.02027502, 0.02348918, 0.02688588, 0.02939095,
 0.03261841, 0.03675437, 0.04145328, 0.04685605, 0.05258239, 0.06055644,
 0.07127699, 0.08661526, 0.10406214, 0.12678802, 0.15203825, 0.18704301,
 0.2319441, 0.27572113, 0.33624849, 0.39666826, 0.47322401, 0.53811002,
 0.58483958, 0.63875258, 0.66788822, 0.64877927, 0.58104825, 0.54410547,
 0.46816054, 0.39988503, 0.36872765, 0.30746305, 0.26421374, 0.22142638,
 0.18307872, 0.15832621, 0.14938028, 0.14581558, 0.13111286, 0.12468486,
 0.12108799, 0.12532991, 0.12312792, 0.12196486, 0.12405467, 0.12505658,
 0.12438187, 0.12210829, 0.11554052, 0.11543264, 0.11017759, 0.10541131,
 0.0994625, 0.09146828, 0.0852076, 0.0776009, 0.06408801, 0.05852762])

frequency = array([
                5.0019516e+04, 3.9687492e+04, 3.1494135e+04, 2.5019525e+04, 1.9843742e+04,
                1.5751949e+04, 1.2519524e+04, 9.9218750e+03, 7.8710889e+03, 6.2695298e+03,
                5.0195298e+03, 4.0571001e+03, 3.2362456e+03, 2.4807932e+03, 1.9681099e+03,
                1.5624999e+03, 1.2403966e+03, 9.8405493e+02, 7.8124994e+02, 6.2019836e+02,
                4.9204675e+02, 3.9062488e+02, 3.0970264e+02, 2.4602338e+02, 1.9531244e+02,
                1.5485132e+02, 1.2299269e+02, 9.7656219e+01, 7.7504944e+01, 6.1515743e+01,
                4.8828110e+01, 3.8771706e+01, 3.0757868e+01, 2.4414059e+01, 1.9385853e+01,
                1.5358775e+01, 1.2187985e+01, 9.6689367e+00, 7.6894670e+00, 6.0939932e+00,
                4.8344669e+00, 3.8447335e+00, 3.0493746e+00, 2.4172332e+00, 1.9195327e+00,
                1.5246876e+00, 1.2086166e+00, 9.5976633e-01, 7.6234382e-01, 6.0430837e-01,
                4.7988322e-01, 3.8109747e-01, 3.0215418e-01, 2.3994161e-01, 1.9053015e-01,
                1.5107709e-01, 1.1996347e-01, 9.5265076e-02, 7.5513750e-02, 5.9981719e-02])
# }}}
# {{{ Curve fitting procedure
fixed_params =  {
                    'T':303.15,                 # temperature ( K )
                    'b':.25,                    # electrode thickness ( cm )
                    'A':5,                      # geometric surface area ( cm^2 )
                    'rho1':1.6,                 # electrolyte specific resistivity ( Ohm cm )
                    'rho2':.012,                # ??? ( Ohm cm )
                    'alpha':.5,                 # transfer coefficient ( dimensionless )
                    'C_R':.0005,              # Red concentration ( mol / cm^3 )
                    'C_O':.0005,              # Ox concentration (mol / cm^3 )
                    'D_R':5e-6,                 # Red diffusivity ( cm^2 / s ) 
                    'D_O':5e-6,                 # Ox diffusivity ( cm^2 / s )
                    'C_dl':20e-6,               # specific double layer capacity ( microfarad / cm^2 )
                    'n':1                       # n electrons
                    }
fitted_params = {
                    'At': 200,                    # Surface area actual ( cm^2 )
                    'i0': 1e-4,                 # exchange current density
                    'P': .95,                    # CPE power factor ( dimensionless )
                    'ASR_mem':5*.05,            # Membrane resistivity ( Ohm cm^2 )
                    'L': 2e-7,                  # inductance ( H )
                    'a': .001,                  # boundary layer thickness ( cm )
                    'Af': .01,                   # fudge factor
                    }

TEMP = FIT( fitted_params = fitted_params,
            fixed_params = fixed_params,
            ReZ = ReZ,
            ImZ = ImZ,
            frequency = frequency,
            fre_init = 12, circuit = 'Randles',
            distributed = False)

# }}}

# {{{ Plotting
plt.plot(ReZ*5,ImZ*5,'r.')
plt.figure(1)
plt.plot(ReZ*5,ImZ*5,'r.')
plt.xlabel('Re(Z) ( $\Omega $ cm$^2$ )')
plt.ylabel('-Im(Z) ( $\Omega $ cm$^2$ )')
plt.axis('equal')
plt.show()
# }}}
