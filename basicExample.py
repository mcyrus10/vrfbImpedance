#!/Users/cyrus/miniconda3/bin/python3
#------------------------------------------------------------------------------
# Written by Michael C. Daugherty
#------------------------------------------------------------------------------
from numpy import real,imag,pi,inf,array,concatenate,log
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
#______________________________________________________________________________
#   Functions
#______________________________________________________________________________
def Z(          Rohmic,
                Rct,
                Rfd,
                C1,
                C2,
                P,
                frequency):
    # {{{
    #--------------------------------------------------------------------------
    # Z returns the impedance at a frequency based on the previously described
    # circuit model
    #--------------------------------------------------------------------------
    # convert Hz to rad/s
    omega = frequency*2*pi  
    return(Rohmic+1/(1/Rct+(1j*omega)**P*C1)+1/(1/Rfd+(1j*omega)**P*C2))
    # }}}
def residual(   initialGuess,
                frequency,
                data):
    # {{{
    #--------------------------------------------------------------------------
    # This function is minimized by the least squares
    #--------------------------------------------------------------------------
    # unpack in inputs
    Rohmic,Rct,Rfd,C1,C2,P = [j for j in initialGuess]
    # generate a spectrum with the inputs
    Z_ = Z(Rohmic,Rct,Rfd,C1,C2,P,frequency)
    # Calculate the difference between the newly generated spectrum and the
    # experimental data
    Real = real(Z_)-real(data)
    Imag = imag(Z_)-imag(data)
    return(concatenate([Real,Imag]))
    # }}}
def ZFit(       initialGuess,
                ReZ,
                ImZ,
                frequency,
                area):
    # {{{
    # This function returns the fitted parameters in a dict
    # parameter order -> [Rohmic, Rct, Rfd, C1, C2, P] 
    data = (ReZ-1j*ImZ)*area
    # Set the bounds for the parameters
    bounds_lower = [0,0,0,0,0,0]
    bounds_upper = [inf,inf,inf,inf,inf,1]
    n = 1

    out = least_squares(residual,
                        initialGuess,
                        bounds = (bounds_lower,bounds_upper),
                        method = 'trf',
                        args = (frequency,data))

    # the fitted parameters are extracted from the 'out' variable
    Rohmic,Rct,Rfd,C1,C2,P = [out.x[j] for j in range(6)]
    return(Rohmic,Rct,Rfd,C1,C2,P)
    # }}}

#______________________________________________________________________________
#   Procedure
#______________________________________________________________________________
# {{{ Raw Data (untreated GFD3 Electrode; .5 M V;  5 cm^2 Aspect Ratio
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
# {{{   Curve Fitting procedure
#______________________________________________________________________________
# initial guesses for each parameter
Rohmic  = 1
Rct     = 2
Rfd     = 1
C1      = 20e-6
C2      = 1e-6
P       = 1

# storing the initial guesses in a list (this is the format that the 'FIT'
# function accepts, but it can be any type of data structure)
initialGuess = [Rohmic,Rct,Rfd,C1,C2,P]

# Call the fitting function with initial guesses and experimental data
Rohmic,Rct,Rfd,C1,C2,P = ZFit(  initialGuess,
                                ReZ,
                                ImZ,
                                frequency,
                                5)

# Generate a spectrum with the fitted parameters to inspect the fit
Z_ = Z(Rohmic,Rct,Rfd,C1,C2,P,frequency)
# }}}
#{{{ Plotting

fig,ax = plt.subplots(nrows = 1,ncols = 2, num = 1, figsize = (10,5))
ax[0].plot(real(Z_),-imag(Z_),'r.')
ax[0].plot(ReZ*5,ImZ*5,'kx')
ax[0].axis('equal')
ax[0].set_xlabel('Re(Z) / $\Omega$ cm$^2$')
ax[0].set_ylabel('-Im(Z) / $\Omega$ cm$^2$')

ax[1].plot(log(frequency*2*pi),ReZ*5,'kx')
ax[1].plot(log(frequency*2*pi),real(Z_),'r.')
ax[1].plot(log(frequency*2*pi),ImZ*5,'kx')
ax[1].plot(log(frequency*2*pi),-imag(Z_),'r.')
ax[1].set_xlabel('log($\omega$) / rad s$^{-1}$')
ax[1].set_ylabel('|Z| / $\Omega$ cm$^2$')

plt.show()
plt.savefig("spectrum.png")
# }}}

