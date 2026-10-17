import math

# Launche Site Data
DEFAULT_MAX_V: float = 287.0 # m/s - Max predicted velocity
DEFAULT_A_MAX_V: float = 3_875.0 # m - Predicted altitude at predicted maximum rocket velocity
DEFAULT_LSA: float = 1_045.0 # m - Launch site altitude (ASL)
DEFAULT_TLS: float = 9.0 # deg C - Temperature at launch site

# Fin Geometry Data
DEFAULT_T: float = 1.27 # cm - Fin thickness
DEFAULT_M: float = 11.67 # cm - Fin sweep length
DEFAULT_TC: float = 9.0 # cm - Tip chord length
DEFAULT_RC: float = 25.0 # cm - Root chord length
DEFAULT_SSL: float = 8.5 # cm - Semi span length (height)
DEFAULT_GE: float = 2770400.76795#420_000 # kPa - Shear Modulus (doubled in calculation if tip-to-tip reinforcement)

# Other Constants
DEFAULT_P0: float = 101.325 # kPa - Air pressure at sea level
DEFAULT_K: float = 1.4 # no units - Ratio of specific heats for air

def get_flutter_velocity(aMaxV: float = DEFAULT_A_MAX_V, lsa: float = DEFAULT_LSA, tls: float = DEFAULT_TLS, t: float = DEFAULT_T, m: float = DEFAULT_M, tc: float = DEFAULT_TC, rc: float = DEFAULT_RC, ssl: float = DEFAULT_SSL, gE: float = DEFAULT_GE, p0: float = DEFAULT_P0, k: float = DEFAULT_K, useTipToTip: bool = True) -> float:
    """
    Calculates the flutter velocity of a rocket based on its fin geometry and launch conditions.

    Parameters:
        aMaxV (float): Predicted altitude at maximum rocket velocity in meters.
        lsa (float): Launch site altitude above sea level in meters.
        tls (float): Temperature at launch site in degrees Celsius.
        t (float): Fin thickness in centimeters.
        m (float): Fin sweep length in centimeters.
        tc (float): Tip chord length of the fin in centimeters.
        rc (float): Root chord length of the fin in centimeters.
        ssl (float): Semi span length (height) of the fin in centimeters.
        gE (float): Shear modulus in kPa. Doubled if tip-to-tip reinforcement is used.
        p0 (float): Air pressure at sea level in kPa.
        k (float): Ratio of specific heats for air (dimensionless).
        useTipToTip (bool, default=True): Whether to account for tip-to-tip reinforcement.
    
    Returns:
        float: The calculated flutter velocity in m/s.
    """

    # Gets the air temperature based on the given altitude above the launch site
    # Requires an altitude in meters (AGL) and returns a temp in deg C
    def getTemp(altitudeAGL: float) -> float:
        """
        Calculates the air temperature in degrees Celsius at a given altitude using the standard lapse rate.

        Parameters:
            altitudeAGL (float): Altitude above the launch site in meters (tls is measured at the launch site).

        Returns:
            float: Air temperature in degrees Celsius at the given altitude.
        """

        return tls - 0.0065 * altitudeAGL

    #! Update this to account for different fin shapes (assumes trapezoidal)
    cx = (2 * tc * m + tc ** 2 + m * rc + tc * rc + rc ** 2) / (3 * (tc + rc)) # cm - x coordinate of centroid of the trapezoidal fin (center of mass)
    thr = t / rc # no units - Thickness ratio
    tar = tc / rc # no units - Taper ratio
    #! Update this to accoutn for different fin shapes (assumes trapezoidal)
    fa = ssl * (tc + rc) / 2 # cm^2 - Fin area for the trapezoidal fin
    asr = ssl ** 2 / fa # no units - Aspect ratio
    eps = cx / rc - 0.25 # no units - The distance from the fin's center of mass to the fin quarter chord as a ratio of the full chord length
    DN = (24 * eps * k * p0) / math.pi # kPa - Denominator constant for flutter velocity calculation
    aMaxVAbsolute = lsa + aMaxV # m - Predicted altitude at predicted maximum rocket velocity as an absolute altitude (accounting for lsa)
    tempAtMaxV = getTemp(aMaxV) # deg C - Air temperature at predicted maximum rocket velocity
    a = 20.05 * math.sqrt(273.15 + tempAtMaxV) # m/s - Speed of sound at aMaxV
    p = p0 * (1 - 0.0065 * aMaxVAbsolute / 288.15) ** 5.256 # kPa - Air pressure at aMaxV (standard atmosphere, ASL altitude)
    term1 = (DN * (asr ** 3)) / ((thr ** 3) * (asr + 2)) # KPa - First section in the denominator
    term2 = (tar + 1) / 2 # no units - Second section in the denominator
    term3 = p / p0 # no units - Third section in the denominator
    flutter_velocity: float
    if useTipToTip:
        flutter_velocity = a * math.sqrt(gE * 2 / (term1 * term2 * term3))
    else:
        flutter_velocity = a * math.sqrt(gE / (term1 * term2 * term3))
    return flutter_velocity

def get_safety_margin(flutter_velocity: float, maxV: float) -> float:
    return flutter_velocity - maxV


# (Example usage) @gio pls write a unit test for this function
if __name__ == "__main__":
    flutter_velocity = get_flutter_velocity()
    safety_margin = get_safety_margin(flutter_velocity, DEFAULT_MAX_V)
    print(f"Flutter Velocity: {flutter_velocity:.2f} m/s")
    print(f"Safety Margin: {safety_margin:.2f} m/s")