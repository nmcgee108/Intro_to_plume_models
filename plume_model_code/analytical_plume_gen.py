import numpy as np

def analytical_plume(T_AW, S_AW, Q_SGD, h_gl, w, alpha):
    """
    Analytical plume model following Muilwijk et al. (2022), Section 3e.
    Code adapted from Matlab script by Aurora Roth (2023)

    Parameters
    ----------
    T_AW : float or array-like
        Conservative temperature of Atlantic Water at glacier front [°C].
    S_AW : float or array-like
        Absolute salinity of Atlantic Water at glacier front [g/kg].
   Q_SGD : float or array-like
        Volume flux of subglacial discharge [m³/s].
    h_gl : float or array-like
        Grounding line depth [m].
    w : float or array-like
        Channel width [m].
    alpha : float or array-like
        Entrainment coefficient

    Returns
    -------
    Tplume : ndarray
        Temperature of the resulting plume [°C].
    Splume : ndarray
        Salinity of the resulting plume [g/kg].
    AWp : ndarray
        Fraction of Atlantic Water in the plume.
    SGDp : ndarray
        Fraction of subglacial discharge in the plume.
    SMWp : ndarray
        Fraction of submarine meltwater in the plume.
    Q_AW : ndarray
        Volume flux of entrained Atlantic Water [m³/s].
    Q_SMW : ndarray
        Volume flux of submarine meltwater [m³/s].
    """
    
    # Ensure all inputs are numpy arrays of same size
    T_AW = np.atleast_1d(T_AW)
    S_AW = np.atleast_1d(S_AW)
    Q_SGD = np.atleast_1d(Q_SGD)
    h_gl = np.atleast_1d(h_gl)
    w = np.atleast_1d(w)

    # Broadcast all inputs to the same shape
    T_AW, S_AW, Q_SGD, h_gl, w = np.broadcast_arrays(T_AW, S_AW, Q_SGD, h_gl, w)

    # Constants from Muilwijk et al. (2022)
    a = alpha       # Entrainment coefficient
    g_0 = 0.26     # Reduced gravity [m^2/s]
    l2 = 8.32e-2   # Freezing point offset [°C]
    l3 = -7.53e-4  # Freezing point depth slope [°C/m]
    A1 = 1.56e-5   # Meltwater flux coefficient [s^(-2/3)]
    A2 = 0.84      # Meltwater flux coefficient [°C^-1]
    
    T_SMW = -90    # Effective temperature of ice [°C]
    S_SMW = 0      # Salinity of submarine meltwater
    S_SGD = 0      # Salinity of subglacial discharge

    # Pressure melting temperature of subglacial discharge
    T_SGD = l2 + l3 * h_gl  # See Muilwijk 2022 pg 7

    # Volume flux of entrained Atlantic Water
    Q_AW = a * h_gl * w * ((Q_SGD * g_0) / (a * w)) ** (1/3)    # Eqn 6 of Muilwijk 2022

    # Volume flux of submarine meltwater
    Q_SMW = w * h_gl * A1 * (1 + A2 * (T_AW - T_SGD)) * (Q_SGD / w) ** (1/3)    # Eqn 7 of Muilwijk 2022

    # Total volume flux in plume
    Q_total = Q_AW + Q_SGD + Q_SMW  

    # Conservative temperature of plume
    Tplume = (Q_AW * T_AW + Q_SGD * T_SGD + Q_SMW * T_SMW) / Q_total    # Eqn 4 of Muilwijk 2022

    # Absolute salinity of plume
    Splume = (Q_AW * S_AW + Q_SGD * S_SGD + Q_SMW * S_SMW) / Q_total    # Eqn 5 of Muilwijk 2022

    # Fractional contributions as defined by buoyant plume theory
    AWp = Q_AW / Q_total
    SGDp = Q_SGD / Q_total
    SMWp = Q_SMW / Q_total

    return Tplume, Splume, AWp, SGDp, SMWp, Q_AW, Q_SMW

