import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import solve_ivp
import gsw
import warnings


"""
ICE SHELF/TIDEWATER GLACIER PLUME MODEL FOR ARBITRARY ICE-OCEAN BOUNDARY GEOMETRY

Natalie McGee 2025
Adapted from Matlab script by Donald Slater, part of Slater 2022, GRL

Useful Notes (Slater 2022)
1. Model assumes ocean surface at z=0 with z negative below surface. **NOTE: This is different z than is used in Slater (2016)
2. Model assumes water emerges at the minimum value in zi
3. Model assumes the ice-ocean boundary is oriented bottom left to top right
4. Model cannot cope with complex in-and-out geometries; i.e. the gradient of zi wrt xi cannot be negative anywhere
5. Model linearly interpolates the shape of the ice-ocean boundary and the ocean conditions between the supplied points

For reference:
    Slater, D. A., et al. (2016). Scalings for Submarine Melting at Tidewater Glaciers from Buoyant Plume Theory. https://doi.org/10.1175/JPO-D-15-0132.1
    Jenkins, A. (2011). Convection-Driven Melting near the Grounding Lines of Ice Shelves and Tidewater Glaciers. https://doi.org/10.1175/JPO-D-11-03.1
    Kaye, N. B. (2008). Turbulent plumes in stratified environments: A review of recent work. Atmosphere-Ocean, 46(4), 433–441. https://doi.org/10.3137/ao.460404

"""

#---------------------------------------------------
#  NONLINEAR EQUATION OF STATE (GSW)
#---------------------------------------------------

rho = lambda T, S, z: gsw.rho(S, T, gsw.p_from_z(z, 70))  # assumes latitude 70°

#---------------------------------------------------
#---------------------------------------------------
#  DEFINE CORE DIFFERENTIAL EQUATIONS
#---------------------------------------------------
#---------------------------------------------------

def equations_line(l, A, par, zi, li, sintheta, Ta, Sa, Na): 
    """
    Defines the differential equations governing the evolution of a buoyant subglacial discharge plume
    rising along an ice-ocean boundary, following buoyant plume theory.

    This function is called by the ODE solver to compute the rate of change of flux variables along
    the glacier face. It accounts for entrainment, meltwater input, and drag, and assumes ambient
    properties vary with depth.

    Parameters:
    ----------
    l : float
        Current arclength position along the ice-ocean interface (m), starting from grounding line.
    A : array_like
        Array of current state variables [Q, M, Q*T, Q*S, Q*N], where:
            Q : Volume flux of the plume (m^2/s)
            M : Momentum flux (m^3/s^2)
            Q*T : Temperature flux (°C·m^2/s)
            Q*S : Salinity flux (g/kg·m^2/s)
            Q*N : Nitrate flux (mmol/m^3·m^2/s)
    par : dict
        Dictionary of physical and model parameters (entrainment coefficient, melt/density parameters, etc.).
    zi : array_like
        Depth values (m), negative below surface.
    li : array_like
        Arclength values corresponding to zi (m).
    sintheta : array_like
        Sine of the slope angle (dz/dx) of the ice-ocean boundary at each li.
    Ta : array_like
        Ambient ocean temperature at depths zi (°C).
    Sa : array_like
        Ambient ocean salinity at depths zi (g/kg).
    Na : array_like
        Ambient ocean nitrate at depths zi (mmol/m^3).

    Returns:
    -------
    diff : list of floats
        Derivatives of the state variables with respect to arclength `l`, in the same order as input A:
            dQ/dl, dM/dl, d(QT)/dl, d(QS)/dl, d(QN)/dl

    """

    Q, M, T_flux, S_flux, N_flux = A 

    # Find instantaneous values of plume variables at current location l along glacier front 
    epsilon = 1e-10 # Add tiny value to denominators to keep from crashing if M, Q = 0
    b = Q**2 / (M + epsilon)        # Plume width (m)
    u = M / (Q + epsilon)           # Plume velocity (m/s)
    T = T_flux / (Q + epsilon)      # Plume conservative temperature (°C)
    S = S_flux / (Q + epsilon)      # Plume absolute salinity (g/kg)
    N = N_flux / (Q + epsilon)      # Plume nitrate (mmol/m^3)

    # Interpolate ambient conditions and slope at current location l
    Ta_l = np.interp(l, li, Ta)
    Sa_l = np.interp(l, li, Sa)
    Na_l = np.interp(l, li, Na)
    sintheta_l = np.interp(l, li, sintheta)
    z = np.interp(l, li, zi)  

    # Reduced gravity at given location in the plume
    if par['EoS'] == 0:     # Using linear EoS
        gp = par['g'] * (par['betaS'] * (Sa_l - S) - par['betaT'] * (Ta_l - T)) # Slater (2016) eqn. (6c)
    elif par['EoS'] == 1:   # Use nonlinear EoS
        rho_ambient = rho(Ta_l, Sa_l, 0)   # Potential density of ambient water at the current location
        rho_plume   = rho(T, S, 0)         # Potential density of plume at the current location 
        gp = (par['g'] / par['rho0']) * (rho_ambient - rho_plume) # Reduced gravity equation 
        

    if par['meltdragfeedback'] == 1:  # Incorporate melt and drag feeback
        # Calculate ice-ocean boundary temperature and salinity using quadratic formula:
        #   Jenkins (2011) eqns (7,8,9) are combined algebraically (Variables u and Cd are eliminated in this process):
        #   The result is a quadratic equation of Sb: quad1 * Sb^2 + quad2 * Sb + quad3 = 0
        quad1 = -par['l1'] * par['cw'] * par['GT'] + par['l1'] * par['ci'] * par['GS']                          # Coefficient on Sb^2 term
        quad2 = par['cw'] * par['GT'] * (T - par['l2'] - par['l3'] * z) + \
                par['GS'] * (par['ci'] * (par['l2'] + par['l3'] * z - par['l1'] * S - par['Ti']) + par['Lm'])   # Coefficient on Sb term
        quad3 = -par['GS'] * S * (par['ci'] * (par['l2'] + par['l3'] * z - par['Ti']) + par['Lm'])              # Constant term
        Sb = (-quad2 + np.sqrt(quad2**2 - 4 * quad1 * quad3)) / (2 * quad1)   # Quadratic formula
        
        Tb = par['l1'] * Sb + par['l2'] + par['l3'] * z   # Jenkins (2011) eqn. (9)
        
        # Calculate melt rate 
        mdot = par['cw'] * np.sqrt(par['Cd']) * par['GT'] * u * (T - Tb) / \
               (par['Lm'] + par['ci'] * (Tb - par['Ti']))   # Jenkins (2011) eqn. (8) (Si = 0)
               
        # Define melt feedback terms
        meltterm_vol = mdot                                                         # Last term in Jenkins (2011) eqn. (1)
        meltterm_temp = mdot * Tb - np.sqrt(par['Cd']) * par['GT'] * u * (T - Tb)   # Last 2 terms in Jenkins (2011) eqn. (3)
        meltterm_sal = mdot * Sb - np.sqrt(par['Cd']) * par['GS'] * u * (S - Sb)    # Last 2 terms in Jenkins (2011) eqn. (4)
        
        # Define drag feedback term
        dragterm = -par['Cd'] * u**2        # Last term in Jenkins (2011) eqn. (2)
        
    else:   # Do not incorporate melt or drag feeback 
        meltterm_vol = meltterm_temp = meltterm_sal = dragterm = 0

    # Entrainment
    E = par['alpha'] * u * sintheta_l    # Jenkins (2011) eqn. (6) (Note different variable names!)

    # Define equations
    dQdl = E + meltterm_vol                 # Jenkins (2011) eqn. (1)
    dMdl = b * gp * sintheta_l + dragterm   # Jenkins (2011) eqn. (2)
    dTdl = E * Ta_l + meltterm_temp         # Jenkins (2011) eqn. (3)
    dSdl = E * Sa_l + meltterm_sal          # Jenkins (2011) eqn. (4)
    dNdl = E * Na_l

    return [dQdl, dMdl, dTdl, dSdl, dNdl]

#---------------------------------------------------
#---------------------------------------------------
#  MAIN PLUME FUNCTION
#---------------------------------------------------
#---------------------------------------------------
    
def run_plume(zi, xi, Ta, Sa, Na, Q0, alpha):
    """
    Run a plume model for a buoyant glacial plume rising vertically from a line source 
    along a given ice-ocean boundary geometry. Function can be supplied with parameters 
    describing the ambient seawater conditions and the shape of the glacial front 
    (see Useful Notes for geometry constraints).
    
    - The function handles two modes of buoyancy calculation: linear or non-linear equation of state.
    - The model optionally includes feedback between melt rate and drag (if `meltdragfeedback` is enabled).
    - All interpolations (e.g., Ta(l)) assume monotonic glacier geometry and smooth input fields.
        
    Parameters:
    ----------
    zi : array_like
        Depths at which xi, Ta, Sa, and Na are defined (negative below surface).
    xi : array_like
        Horizontal position of ice-ocean boundary at each zi.
        Use 0 for vertical front (tidewater glacier).
    Ta : array_like
        Ocean temperature at depths zi.
    Sa : array_like
        Ocean salinity at depths zi.
    Na : array_like
        Ocean nitrate at depths zi.
    Q0 : float
        Subglacial discharge (m²/s).
    alpha : float
        Entrainment coefficient.

    Returns:
    -------
    sol : dict containing the following
        z : Depth along the ice face [m], negative downward
        b : Plume width [m]
        u : Plume velocity [m/s]
        T : Plume temperature [°C]
        S : Plume salinity [g/kg]
        N : Plume nitrate concentration [mmol/m^3]
        Sb : Boundary salinity (at ice face) [g/kg]
        Tb : Boundary temperature (at ice face, freezing point) [°C]
        mdot : Melt rate per unit width m^2/s, multiplied by 86400 for m^2/day [m^2/day]
        rho : Plume density [kg/m^3]
        Ta : Ambient temperature at plume depth [°C]
        Sa : Ambient salinity at plume depth [g/kg]
        Na : Ambient nitrate at plume depth [mmol/m^3]
        rhoa : Ambient density at plume depth [kg/m^3]
        zNB : Neutral buoyancy depth [m]
        TNB : Temperature at neutral buoyancy level [°C]
        SNB : Salinity at neutral buoyancy level [g/kg]
        NNB: Nitrate at neutral buoyancy level [mmol/m^3]
        QNB : Volume flux at neutral buoyancy level [m^2/s]
        HNB : Heat flux anomaly referenced to -2°C [W/m]
    """
    #---------------------------------------------------
    #  FIXED PHYSICAL PARAMETERS
    #---------------------------------------------------
    
    par = {
        'alpha': alpha,     # Entrainment coefficient [unitless]
        'g': 9.81,          # Gravitational acceleration [m/s^2]
        'rho0': 1020,       # Reference density of seawater [kg/m^3]
        'l1': -5.73e-2,     # Freezing point slope (w.r.t. salinity) [°C kg/g]
        'l2': 8.32e-2,      # Freezing point offset (at 0 depth, 0 sal) [°C]
        'l3': 7.61e-4,      # Freezing point slope (w.r.t. depth) [°C/m]
        'cw': 3974,         # Specific heat capacity of seawater [J/kg K]
        'ci': 2009,         # Specific heat capacity of ice [J/kg K]
        'Lm': 334000,       # Latent heat of fusion [J/kg]
        'Cd': 2.5e-3,       # Drag coefficient (plume-glacier front) [unitless]
        'GT': 1.1e-2,       # Heat transfer coefficient [unitless]
        'GS': 3.1e-4,       # Salt transfer coefficient [unitless]
        'Ti': -10,          # Temperature of ice [°C]
        'betaS': 7.86e-4,   # Thermal expansion coefficient (salinity) [kg/kg]
        'betaT': 3.87e-5,   # Thermal expansion coefficient (temperature) [1/°C]
        
        #---------------------------------------------------
        #  OPTIONAL PHYSICAL PARAMETERS
        #---------------------------------------------------
        
        'Gamma0': 1,            # Dimensionless paramter relating buoyancy to momentum
        'meltdragfeedback': 1,  # Turn on/off melt-drag feedback (0 = off, 1 = on)
        'EoS': 1              # Choose equation of state model (0 = linear, 1 = non-linear)
        }

    #---------------------------------------------------
    # INPUTS AND INITIAL CONDITIONS 
    #---------------------------------------------------
    
    # Ensure inputs are NumPy arrays
    zi = np.array(zi)
    xi = np.array(xi)
    Ta = np.array(Ta)
    Sa = np.array(Sa)
    Na = np.array(Na)
    
    # Basic input check: warn if any depth values are > 0 (should be ≤ 0)
    if np.any(zi > 0):
        warnings.warn("WARNING: z-input should all be <= 0")
    
    # Geometry check: no in-and-outs permitted in glacier geometry
    if np.any(np.gradient(zi, xi) < 0):
        warnings.warn("WARNING: gradient of zi wrt xi cannot be negative anywhere")
    
    # Sort all inputs by zi (depth), to start from the grounding line
    sort_ind = np.argsort(zi)
    zi = zi[sort_ind]
    xi = xi[sort_ind]
    Ta = Ta[sort_ind]
    Sa = Sa[sort_ind]
    Na = Na[sort_ind]

    # Initial conditions
    S0 = 0  # Salinity of subglacial input
    N0 = 0  # Nitrate of subglacial input
    T0 = par['l2'] + par['l3'] * np.min(zi)  # Temperature at glacier base. Jenkins (2011) eqn. (11) (S = 0)

    
    # Initial reduced gravity of plume at grounding line
    if par['EoS'] == 0:     # Using linear EoS
        g0p = par['g'] * (par['betaS'] * (Sa[0] - S0) - par['betaT'] * (Ta[0] - T0)) # Slater (2016) eqn. (6c)
    elif par['EoS'] == 1:   # Use nonlinear EoS
        rho_ambient = rho(Ta[0], Sa[0], 0)   # Potential density of ambient water at grounding line
        rho_plume   = rho(T0, S0, 0)         # Potential density of plume at grounding line
        g0p = (par['g'] / par['rho0']) * (rho_ambient - rho_plume) # Reduced gravity equation
    
    # Initial plume width and velocity
    b0 = (par['alpha'] * Q0**2 * par['Gamma0'] / g0p) ** (1/3) # Related to Slater (2016) eqn (5)
    u0 = Q0 / b0    # Flux through unit width of plume
    
    # Calculate initial fluxes
    QFLUX0 = u0 * b0        # Volume flux
    MFLUX0 = u0**2 * b0     # Momentum flux
    TFLUX0 = b0 * u0 * T0   # Temperature flux
    SFLUX0 = b0 * u0 * S0   # Salinity flux
    NFLUX0 = b0 * u0 * N0   # Nitrate flux
    
    # Compute the slope of the glacier front with z as the independent variable = dx/dz (for future integration with respect to z)
    dxdz = np.gradient(xi, zi) # For each z,x location calculate the slope of the glacier front dx/dz

    # Compute cumulative arc length (li) along the ice-ocean interface 
    li = cumulative_trapezoid(np.sqrt(1 + dxdz**2), zi, initial=0) # For each z location return the corresponding along-front distance

    # Compute sin(theta) where theta is the angle between the glacier front and the x-axis/fjord bottom
    sintheta = 1 / np.sqrt(1 + dxdz**2)

    
    #---------------------------------------------------
    # SOLVE THE DIFFERENTIAL EQUATIONS
    #---------------------------------------------------

    # Initial conditions
    y0 = [QFLUX0, MFLUX0, TFLUX0, SFLUX0, NFLUX0]
    
    # Define the full interval of the paramaterizing variable: along-front length li 
    t_span = (li[0], li[-1])
    
    # Define ODE solver tolerances
    tolerances = {
        'atol': 1e-10,      # Absolute tolerance
        'rtol': 1e-5,       # Relative tolerance
    }
    
    # Event function (stops integration when plume width exceeds threshold)
    def eventfc_line(t, y):
        Q, M, _, _, _ = y  # Unpack the solution vector: y = [Q, M, QT, QS, QN]
        b = Q**2 / M  # Plume width (b) from volume and momentum fluxes
        return float(b - 500)  # Event triggers when this equals 0
    
    eventfc_line.terminal = True # Stop ODE solver when event is triggered
    eventfc_line.direction = 0 # Detect zero-crossing from either side
    
    # Wrap equations_line into function with only two arguments (independent, dependent) to give it the correct signature for solve_ivp
    # l is the current value of li, the paramaterizing variable,
    # A is the array of plume properties at the current location l.
    def wrapped_equations_line(l, A): 
        return equations_line(l, A, par, zi, li, sintheta, Ta, Sa, Na)
    
    # Solve ODE
    sol = solve_ivp(
        fun=wrapped_equations_line, # Function to be solved
        t_span=t_span,              # Interval to evaluate over
        y0=y0,                      # Initial conditions
        events=eventfc_line,         # Event function (activates to shut off ODE after plume gets large)
        **tolerances,               # Define solver tolerances
        dense_output=True           # For smooth interpolation
    )
    
    #---------------------------------------------------
    # BACK OUT SOLUTION (Retrieve Physical Values)
    #--------------------------------------------------- 
    
    # Extract along-front positions
    l = sol.t
    
    # Array of fluxes at each l
    A = sol.y.T  # transpose to match MATLAB output shape

    
    # Prepare a dictionary to fill with the extracted values
    sol = {} 
    sol['z'] = []       # Depth along the ice face [m], negative downward
    sol['b'] = []       # Plume width [m]
    sol['u'] = []       # Plume velocity [m/s]
    sol['T'] = []       # Plume temperature [°C]
    sol['S'] = []       # Plume salinity [g/kg]
    sol['N'] = []       # Plume nitrate concentration [mmol/m^3]
    sol['Sb'] = []      # Boundary salinity (at ice face) [g/kg]
    sol['Tb'] = []      # Boundary temperature (at ice face, freezing point) [°C]
    sol['mdot'] = []    # Melt rate per unit width [m^2/s], multiplied by 86400 for m^2/day
    sol['rho'] = []     # Plume density [kg/m^3]
    sol['Ta'] = []      # Ambient temperature at plume depth [°C]
    sol['Sa'] = []      # Ambient salinity at plume depth [g/kg]
    sol['Na'] = []      # Ambient nitrate at plume depth [mmol/m^3]
    sol['rhoa'] = []    # Ambient density at plume depth [kg/m^3]

    
    # For each location along the front, extract the values of each plume property
    for i in range(len(l)):
        z = np.interp(l[i], li, zi)             
        Q, M, T_flux, S_flux, N_flux = A[i, :]   
        
        b = Q**2 / M
        u = M / Q
        T = T_flux / Q
        S = S_flux / Q
        N = N_flux / Q
    
        # Compute boundary salinity and temperature using quadratic formula (Jenkins (2011) eqns (7,8,9)). See equations_line()
        quad1 = -par['l1'] * par['cw'] * par['GT'] + par['l1'] * par['ci'] * par['GS']                          # Coefficient on Sb^2 term
        quad2 = par['cw'] * par['GT'] * (T - par['l2'] - par['l3'] * z) + \
                par['GS'] * (par['ci'] * (par['l2'] + par['l3'] * z - par['l1'] * S - par['Ti']) + par['Lm'])   # Coefficient on Sb term
        quad3 = -par['GS'] * S * (par['ci'] * (par['l2'] + par['l3'] * z - par['Ti']) + par['Lm'])              # Constant term
    
        Sb = (-quad2 + np.sqrt(quad2**2 - 4 * quad1 * quad3)) / (2 * quad1)     # Quadratic formula
        Tb = par['l1'] * Sb + par['l2'] + par['l3'] * z      # Jenkins (2011) eqn. (9)
    
        mdot = 86400 * par['cw'] * np.sqrt(par['Cd']) * par['GT'] * u * (T - Tb) / \
               (par['Lm'] + par['ci'] * (Tb - par['Ti']))   # Jenkins (2011) eqn. (8) (Si = 0). Multiply by 86400 to convert to m^2/day


        # Ambient conditions do not rely on the plume model
        Ta_i = np.interp(z, zi, Ta)
        Sa_i = np.interp(z, zi, Sa)
        Na_i = np.interp(z, zi, Na)
        
        rho_p = rho(T, S, z)            # Calculate plume density from local T, S
        rho_a = rho(Ta_i, Sa_i, z)      # Calculate ambient density from local Ta, Sa
    
        # Store
        sol['z'].append(z)
        sol['b'].append(b)
        sol['u'].append(u)
        sol['T'].append(T)
        sol['S'].append(S)
        sol['N'].append(N)
        sol['Sb'].append(Sb)
        sol['Tb'].append(Tb)
        sol['mdot'].append(mdot)
        sol['rho'].append(rho_p)
        sol['Ta'].append(Ta_i)
        sol['Sa'].append(Sa_i)
        sol['Na'].append(Na_i)
        sol['rhoa'].append(rho_a)
        
    #---------------------------------------------------
    # NEUTRAL BUOYANCY PROPERTIES
    #---------------------------------------------------
    
    # Find the last index where plume density < ambient density
    neutral_inds = np.where(np.array(sol['rho']) < np.array(sol['rhoa']))[0]
    if len(neutral_inds) > 0:
        id_nb = neutral_inds[-1]  # index of neutral buoyancy point

        sol['zNB'] = sol['z'][id_nb]    # Depth of neutral buoyancy
        sol['TNB'] = sol['T'][id_nb]    # Plume temperature at neutral buoyancy level
        sol['SNB'] = sol['S'][id_nb]    # Plume salinity at neutral buoyancy level
        sol['NNB'] = sol['N'][id_nb]    # Plume nitrate at neutral buoyancy level
        sol['QNB'] = sol['b'][id_nb] * sol['u'][id_nb]      # Plume volume flux at neutral buoyancy level
        sol['HNB'] = par['cw'] * sol['rho'][id_nb] * sol['QNB'] * (sol['TNB'] - (-2))  # Heat flux anomaly referenced to -2°C
       
    return sol

