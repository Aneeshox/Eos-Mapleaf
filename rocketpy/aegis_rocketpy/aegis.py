# Author: Aaron Arellano

import datetime
from rocketpy import Environment, Flight, Rocket
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank

INCLUDE_AIRBRAKES = False

# Currently ~150m off of mapleaf; these improvements would make this closer:
# - Grain geometry/masses; Adjust density to ensure proper mass from motor file
# - Airbrakes CDs

# plt.style.use("seaborn-v0_8-colorblind")
tomorrow = datetime.date(2023, 10, 13) + datetime.timedelta(days=1)

# ── Environment ──────────────────────────────────────────────────────────────
# elevation = 871.603032 m (Midland Texas average)
# Atmosphere: US Standard Atmosphere (matches MAPLEAF USStandardAtmosphere)
# Wind: constant (3, 0, 0) m/s (matches MAPLEAF ConstantMeanWind)
Env = Environment(
    latitude=49,
    longitude=81,
    elevation=871.603032,
)
Env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
Env.set_atmospheric_model(
    type="custom_atmosphere",
    pressure=None,        # falls back to standard atmosphere pressure
    temperature=None,     # falls back to standard atmosphere temperature
    wind_u=3,             # constant 3 m/s eastward — MAPLEAF ConstantMeanWind (3 0 0)
    wind_v=0,             # no north/south wind component
)
# Env.max_expected_height = 11000
Env.info()

# ── Oxidizer Tank ─────────────────────────────────────────────────────────────
fuel = Fluid(name="paraffin_s", density=900)
oxidizer_liq = Fluid(name="N2O_l", density=1220)
oxidizer_gas = Fluid(name="N2O_g", density=1.0)

tank_shape = CylindricalTank(0.167 / 2, 0.813)

pv = MassFlowRateBasedTank(
    name="oxidizer_tank",
    geometry=tank_shape,
    flux_time=5.476,
    liquid=oxidizer_liq,
    gas=oxidizer_gas,
    initial_liquid_mass=12.9398,
    initial_gas_mass=0.0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=1.449926953981008,
    gas_mass_flow_rate_in=0.0,
    gas_mass_flow_rate_out=0.0,
)

# ── Hybrid Motor ──────────────────────────────────────────────────────────────
hybrid_motor = HybridMotor(
    # wrote lil py script to convert Aegis.csv (or Aegis67 on github) to convert to a rocketpy friendly format
    thrust_source=r"C:\Users\giova\OneDrive\Documents\SOAR\Eos-Mapleaf\rocketpy\aegis_rocketpy\Aegis_rpy.csv",
    dry_mass=20.7388,
    
    # UpperPlumbing = 0.68
    # UpperPVCasing = 0.419
    # UpperPVBulkhead = 0.449
    # PressureVessel = 5.7007
    # LowerPVCasing = 0.419
    # LowerPVBulkhead = 0.4846
    # LowerPlumbingTube = 0.745
    # FinCan+CC+MoterLiner = 6.5
    # LowerPlumbing = 2.765
    # LowerInjectorBulkhead = 0.3585
    # Mixingplate = 0.263
    # Nozzle+Houseing = 1.955
    
    
    dry_inertia=(5.841, 5.841, 0.190), #sum of mois from above
    center_of_dry_mass_position=0.8011518001456213, # sum of component mass * component cgs / sum of masses
    
    # Currently adjusts height to ensure the proper mass is used. In the future, change the density instead
    grain_number=1,
    grain_separation=0,
    grain_outer_radius=0.167 / 2,
    grain_initial_inner_radius=0.10 / 2,
    grain_initial_height=0.09514129342981771,
    grain_density=900,
    nozzle_radius=0.167 / 2,
    throat_radius=0.05715,
    grains_center_of_mass_position=0.2516013400000001,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

hybrid_motor.add_tank(tank=pv, position=1.3233)
hybrid_motor.info()

# ── Rocket ────────────────────────────────────────────────────────────────────
aegis = Rocket(
    radius=0.167 / 2,
    mass=19.8910, # rocket without motor: 
    inertia=(12.393, 12.393, 0.030),
    center_of_mass_without_motor=-1.3149727671449398,
    
    # most consequential values; created a python script to  solve for off drag, but on is just an estimate
    power_off_drag=0.6, # check out cd_calculations.py
    power_on_drag=0.7,# idk, estimate?
    
    
    coordinate_system_orientation="tail_to_nose",
)

# Motor nozzle exit = tip of AegisPenis = −3.814845 − 0.03675634 = −3.8516 m from nose
aegis.add_motor(hybrid_motor, position=-3.8516)

# Nosecone: Von Karman, aspectRatio=3, baseDiameter=0.167 → length = 3 × 0.167 = 0.501 m
NoseCone = aegis.add_nose(length=0.501, kind="vonKarman", position=0)

# Fins: all geometry from MAPLEAF FinSet block
#   MAPLEAF position (0 0 −3.429) = leading-edge root chord, 3.429 m aft of nose tip
FinSet = aegis.add_trapezoidal_fins(
    n=3,
    span=0.14,
    root_chord=0.36,
    tip_chord=0.119,
    position=-3.429,      
    sweep_angle=40.7,
)

# Airbrakes — geometry unchanged; control logic to be added separately
# Define air brakes controller function
def airbrakes_controller(time, sampling_rate, state, state_history, observed_variables, air_brakes):
    '''
    From https://docs.rocketpy.org/en/latest/user/airbrakes.html:
        
    time (float): The current simulation time in seconds.

    sampling_rate (float): The rate at which the controller function is called, measured in Hertz (Hz).
    
    state (list): The state vector of the simulation. The state is a list containing the following values, in this order:
    
    state_history (list): A record of the rocket's state at each step throughout the simulation. The state_history is organized as a list of lists, with each sublist containing a state vector. The last item in the list always corresponds to the previous state vector, providing a chronological sequence of the rocket's evolving states.
    
    observed_variables (list): A list containing the variables that the controller function returns. The return of each controller function call is appended to the observed_variables list. The initial value in the first step of the simulation of this list is provided by the initial_observed_variables argument.

    air_brakes (AirBrakes): The AirBrakes instance being controlled.
    '''
    
    
    altitude_ASL = state[2]
    altitude_AGL = altitude_ASL - Env.elevation
    vz = state[5]  

    # Get previous vertical velocity from state history for acceleration calculation
    if len(state_history) > 0:
        previous_vz = state_history[-1][5]
    else:
        previous_vz = vz

    if time < hybrid_motor.burn_out_time:
        air_brakes.deployment_level = 0
        return (time, air_brakes.deployment_level, 0)

    # If below 500 meters above ground level, air brakes are not deployed
    if altitude_AGL < 500:
        air_brakes.deployment_level = 0
    else:
        
        new_deployment_level = (
            air_brakes.deployment_level + 0.1 * vz + 0.01 * previous_vz**2
        )

        max_change = 0.1 / sampling_rate
        lower_bound = air_brakes.deployment_level - max_change
        upper_bound = air_brakes.deployment_level + max_change
        new_deployment_level = min(max(new_deployment_level, lower_bound), upper_bound)

        air_brakes.deployment_level = new_deployment_level


    vx, vy, vz = state[3], state[4], state[5]
    wind_x = Env.wind_velocity_x(altitude_ASL)
    wind_y = Env.wind_velocity_y(altitude_ASL)
    free_stream_speed = ((wind_x - vx) ** 2 + (wind_y - vy) ** 2 + (vz) ** 2) ** 0.5
    mach_number = free_stream_speed / Env.speed_of_sound(altitude_ASL)


    return (
        time,
        air_brakes.deployment_level,
        air_brakes.drag_coefficient(air_brakes.deployment_level, mach_number),
    )

if INCLUDE_AIRBRAKES:
    air_brakes = aegis.add_air_brakes(
        drag_coefficient_curve=r"C:\Users\giova\OneDrive\Documents\SOAR\Eos-Mapleaf\rocketpy\aegis_rocketpy\airbrakes_cd.csv",
        controller_function=airbrakes_controller,
        sampling_rate=10,
        reference_area=None,
        clamp=True,
        initial_observed_variables=[0, 0, 0],
        override_rocket_drag=False,
        name="Ouro Air Brakes",
    )

test_flight = Flight(
    rocket=aegis,
    environment=Env,
    rail_length=12.192,    # m - MAPLEAF railLength = 12.192 m
    inclination=90,        # vertical launch
    heading=0,
)
aegis.draw()
aegis.info()
aegis.plots.static_margin()
test_flight.prints.apogee_conditions()
test_flight.plots.trajectory_3d()
test_flight.altitude()
test_flight.vz()