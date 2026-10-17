"""Testing building a rocket.
"""

import datetime
from rocketpy import Environment, Flight, Rocket, SolidMotor

ROCKETPY_PATH = "C:/Users/giova/RocketPy"


env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12)) # Hour given in UTC time
env.set_atmospheric_model(type="Forecast", file="GFS")
print("########## Environment Info ##########")
env.info()
print("######################################")
Pro75M1670 = SolidMotor(
	thrust_source=ROCKETPY_PATH + "/data/motors/cesaroni/Cesaroni_M1670.eng",
	dry_mass=1.815,
	dry_inertia=(0.125, 0.125, 0.002),
	nozzle_radius=33 / 1000,
	grain_number=5,
	grain_density=1815,
	grain_outer_radius=33 / 1000,
	grain_initial_inner_radius=15 / 1000,
	grain_initial_height=120 / 1000,
	grain_separation=5 / 1000,
	grains_center_of_mass_position=0.397,
	center_of_dry_mass_position=0.317,
	nozzle_position=0,
	burn_time=3.9,
	throat_radius=11 / 1000,
	coordinate_system_orientation="nozzle_to_combustion_chamber",
)
print("########## Motor Info ##########")
Pro75M1670.all_info()
print("################################")
calisto = Rocket(
	radius=127 / 2000,
	mass=14.426,
	inertia=(6.321, 6.321, 0.034),
	power_off_drag=ROCKETPY_PATH + "/data/rockets/calisto/powerOffDragCurve.csv",
	power_on_drag=ROCKETPY_PATH + "/data/rockets/calisto/powerOnDragCurve.csv",
	center_of_mass_without_motor=0,
	coordinate_system_orientation="tail_to_nose",
)
calisto.add_motor(Pro75M1670, position=-1.255)
nose_cone = calisto.add_nose(length=0.55829, kind="von karman", position=1.278)
fin_set = calisto.add_trapezoidal_fins(
	n=4,
	root_chord=0.120,
	tip_chord=0.060,
	span=0.110,
	position=-1.04956,
	cant_angle=0.5,
	airfoil=(ROCKETPY_PATH + "/data/airfoils/NACA0012-radians.txt","radians"),
)
tail = calisto.add_tail(top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656)
main = calisto.add_parachute(
	name="main",
	cd_s=10.0,
	trigger=800,      # ejection altitude in meters
	sampling_rate=105,
	lag=1,
	noise=(0, 8.3, 0.5),
)
drogue = calisto.add_parachute(
	name="drogue",
	cd_s=1.0,
	trigger="apogee",  # ejection at apogee
	sampling_rate=105,
	lag=1,
	noise=(0, 8.3, 0.5),
)
calisto.draw()
test_flight = Flight(rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0)
calisto.plots.static_margin()
print("########## Flight Info ##########")
test_flight.all_info()
print("#################################")
# test_flight.plots.trajectory_3d()
# test_flight.plots.linear_kinematics_data()
# test_flight.plots.flight_path_angle_data()
# test_flight.plots.angular_kinematics_data()
# test_flight.plots.rail_buttons_forces()
# test_flight.plots.aerodynamic_forces()
# test_flight.export_data(
#    "calisto_flight_data.csv",
# )
