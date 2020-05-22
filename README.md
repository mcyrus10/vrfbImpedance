Least Squares Regression for Impedance Analysis
===============================================

Written by Michael C. Daugherty for EESC Lab University of Tennessee

  * [Modules](#modules)
  * [Functions](#functions)
  * [Plotting](#plotting)
  * [Randles Circuit](#randles-circuit)

###Modules
These are the python modules needed for the following tutorials:
  * [matplotlib](https://matplotlib.org/) : matplotlib.pyplot
  * [numpy](https://numpy.org/) : real,imag,pi,inf,array,concatenate,log
  * [scipy](https://www.scipy.org/) : scipy.optimize.least\_squares

They can be installed with one shell command:

    $ pip install matplotlib numpy scipy

Curve fitting impedance spectra for VRFB
----------------------------------------
Equivalent circuit model: membrane serially connected to a parallel charge
transfer (Rct) and capacitance1 (C1) connected serially to a parallel diffusion
resistance (Rfd) and capacitance2 (C2). The capacitors are CPE elements with
the same exponent (P).
**NOTE**: Circuit elements should have justification in their physical meaning,
sufficiently complex equivalent circuits can fit many different spectra.
  This is the representation of the circuit:

                          C1                 C2
                          | |                | |
                    |-----| |-----|    |-----| |-----|
        Rohmic      |     | |     |    |     | |     |
    ---\/\/\/\/-----|             |----|             |-----
                    |             |    |             |
                    |---/\/\/\/---|    |---/\/\/\/---|
                          Rct                Rfd


This code can be easily modified to fit different types of spectra if a circuit
model is known or can be abstracted to fit 


###Functions
  - *Z* - Returns the real and imaginary impedance response for the above
    circuit
    - The ordered input arguments for this function are:
      1) Rohmic - ohmic resistance (membrane+electronic components)
      2) Rct - charge transfer resistance (ohm cm^2)
      3) Rfd - finite diffusion resistance (ohm cm^2)
      4) C1 - Capacitance 1 (connected in parallel with the charge transfer
      resistance) (F)
      5) C2 - Capacitance 2 (connected in parallel with the finite diffusion
      resistance (F)
      6) P - CPE exponent (-)
      7) frequency - frequency to evaluate (Hz)
    - Example usage:
```python
Rohmic  = 1
Rct     = 2
Rfd     = 1
C1      = 20e-6
C2      = 10e-6
P       = .93
fre     = 100
Z(Rohmic,Rct,Rfd,C1,C2,P,fre)
[Returns]
(3.9955342404129186-0.035679897432746334j)
```
  - *residual* - Returns the difference between a generated spectrum and a
    target spectrum (Experimental data). This function is an argument for 
    [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
which it minimizes with successive calls. The real and imaginary components are
calculated compared independently then concatenated to return a 1D array that
is 2x the length of the experimental data.
    - The ordered input arguments for this function are:
      1) initialGuess - an ordered list containing the initial guesses for
      each fitted parameter
      2) frequency - a numpy.array containing the frequencies at which each
      point on the spectrum is calculated
      3) data - target (experimental) data against which the generated
      spectrum is compared (numpy.array of complex numbers). **Note**: the
      function transforms the resistance into ASR in this instance, so it is
      not necessary to pre-multiply by the geometric surface area.
    - Example usage:
```python
# continued from previous
initialGuess = [Rohmic,Rct,Rfd,C1,C2,P]
Zexperimental = array([0.0824 - 0.0117j,
                                .       .
                                .       .
                                .       .
                                ])
frequency = array([5.00e4,
                            .
                            .
                            .
                            ])
residual(initialGuess, frequency, Zexperimental)
[Returns]
array([1.5084,
                .
                .
                .
                ])
```
  - *Zfit* - This function calls scipy.optimize.least\_squares which minimizes
    the residual and returns the optimized parameters
    - The ordered input arguments for this function are:
      1) initialGuess - an ordered list containing the initial guesses for
      each fitted parameter
      2) ReZ - Real component of impedance (experimental data) 
      3) ImZ - Imaginary component of impedance (experimental data) 
      4) frequency - experimental data
      5) area - geometric surface area (cm^2)
    - Example usage:
```python
# continued from previous
ReZ = real(Zexperimental)
ImZ = imag(Zexperimental)
area = 5
Zfit(initialGuess,ReZ,ImZ,frequency,area)
[Returns]
(0.5235613427786554,
 7.2867828048527326,
 1.4850401014703765,
 0.0003124126652879431,
 0.19887268939447986,
 0.9192495541928422)
```

###Plotting

After the values parameters are estimated, they are used to generate another
spectrum (*Z\_*), so that it can be visually compared to the experimental data
in the nyquist and bode plots.

![](spectrum.png)

Randles Circuit
---------------

Suggested reading for Randles Circuit understanding: *Electrochemical Impedance
Spectroscopy and its Applications* by Andrzej Lasia (Chapter 4).

  This is the representation of the circuit:

                                 CPE
                                |   |
                    |-----------|   |-------------|
        Rohmic      |           |   |             |
    ---\/\/\/\/-----|                             |-----
                    |                   |-----|   |
                    |---/\/\/\/---------|  W  |---|
                          Rct           |-----|
                                       

The code used for the previous circuit can be modified to accommodate an
equivalent circuit of this type; however, now, instead of using the lumped
parameters Rct and Rfd, the parameters that constitute the lumped parameters
can be fitted directly.

###Charge Transfer Resistance

The charge transfer resistance is not a function of frequency so the magnitude
of the charge transfer at all frequencies is:

$$R_{ct} = \frac{A}{A_t} \frac{RT}{nFi_0}$$

  * A = Geometric surface area
  * R = Universal Gas constant
  * T = Temperature
  * At = Wetted surface area
  * n = Number of electrons transferred
  * F = Faraday's constant

###Warburg Impedance Element

A Warburg element is used to model mass transfer for redox reactions. At
**open circuit**, the expression for the warburg impedance is:


$$W = \frac{A}{A_t} \frac{RT}{f F^2 [C_O]_b D_R}\frac{tanh(a\sqrt{\frac{j\omega}{D_R}})}{a\sqrt{\frac{j\omega}{D_R}}} + \frac{A}{A_t} \frac{RT}{f F^2 [C_R]_b D_O}\frac{tanh(a\sqrt{\frac{j\omega}{D_O}})}{a\sqrt{\frac{j\omega}{D_O}}}$$

  * f = Scale factor
  * a = Mean Nernstian diffusion layer thickness
  * j = $\sqrt{-1}$
  * $\omega$ = angular frequency
  * [C$_O$]$_b$ = Concentration of oxidized species in the bulk 
  * [C$_R$]$_b$ = Concentration of reduced species in the bulk
  * D$_O$ = Diffusion coefficient of oxidized species
  * D$_R$ = Diffusion coefficient of reduced species

As the frequency approaches zero:

$$\lim_{\omega \rightarrow 0} W = R_{fd} = \frac{A}{A_t} \frac{aRT}{fF^2}\bigg(\frac{1}{[C_R]D_R} + \frac{1}{[C_O]D_O}\bigg)$$

###Charge Transfer Impedance

To represent the Randles Circuit model we can combine the charge transfer
resistance and the Warburg element into the charge transfer impedance
$Z_{ct}$, which is simply the sum of these two elements (as they are
connected serially.

$$Z_{ct} = R_{ct} + W$$

###Modeling a whole cell impedance response

Now the impedance is:

$$Z = R_{ohmic} + 2\bigg(\frac{1}{\bigg)$$



###Randles Circuit Functions

  - single\_electrode - returns the real and imaginary impedance response for
    the randles circuit
    - The ordered input arguments for this function are:
      1) frequency - frequencies to evalute
      2) fitted_params - dictionary of all of the fitted (floating) parameters
      3) fixed_params - dictionary containing the static values for the model
    - Example usage:
```python
# continued from above
fixed_params =  {
                    'T':303.15,      # temperature ( K )
                    'b':.25,         # electrode thickness ( cm )
                    'A':5,           # geometric surface area ( cm^2 )
                    'rho1':1.6,      # electrolyte specific resistivity ( Ohm cm )
                    'rho2':.012,     # ??? ( Ohm cm )
                    'alpha':.5,      # transfer coefficient ( dimensionless )
                    'C_R':.0005,     # Red concentration ( mol / cm^3 )
                    'C_O':.0005,     # Ox concentration (mol / cm^3 )
                    'D_R':5e-6,      # Red diffusivity ( cm^2 / s ) 
                    'D_O':5e-6,      # Ox diffusivity ( cm^2 / s )
                    'C_dl':20e-6,    # specific double layer capacity (microfarad / cm^2)
                    'n':1            # n electrons
                    }
fitted_params = {
                    'At': 200,       # Surface area actual ( cm^2 )
                    'i0': 1e-4,      # exchange current density
                    'P': .95,        # CPE power factor ( dimensionless )
                    'ASR_mem':5*.05, # Membrane resistivity ( Ohm cm^2 )
                    'L': 2e-7,       # inductance ( H )
                    'a': .001,       # boundary layer thickness ( cm )
                    'Af': .01,       # Scale Factor
                    }
fre = 100
single_electrode(fre,fitted_params,fixed_params)
[returns]
(1.095422682208457-2.203550374218462j)
```
