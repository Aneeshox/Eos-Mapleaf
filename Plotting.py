# Created by: Henry Stoldt
# April 2020

'''
Functions to create plots and animations of simulations results.
'''

import re
from bisect import bisect_right
from collections import OrderedDict
from math import radians
from statistics import mean, stdev

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import pandas as pd

from MAPLEAF.Motion import interpolateRigidBodyStates, linInterpWeights
from MAPLEAF.IO import RocketFlight, getAbsoluteFilePath
from MAPLEAF.Motion import Quaternion
from MAPLEAF.Motion import Vector

# For all plots for publications: (match publication font)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
plt.rcParams["font.weight"] = "normal"

logFileCache = OrderedDict()
''' Will keep the contents of the last n log files loaded cached, to avoid re-loading repeatedly when making lots of plots '''
maxLogFileCacheSize = 10
''' Max number of log files to keep cached simultaneously '''

### Plot data from log files ###
def plotFromLogFiles(logFilePaths, plotDefinitionString, showPlot=True):
    if logFilePaths == None:
        print("ERROR: can't plot {} - must log data to plot it".format(plotDefinitionString))
        return

    if plotDefinitionString == "Velocity":
        # Prevents Velocity from also plotting angularVelocity
        columns = [ "VelocityX(m/s)", "VelocityY(m/s)", "VelocityZ(m/s)" ]  # Removes Angular Velocities from plot (otherwise plots all columns containing "Velocity")
    elif '&' in plotDefinitionString:
        # Split 'col' at each '&' symbol - this allows for plotting multiple things on a single graph
        columns = plotDefinitionString.split('&')
    else:
        columns = plotDefinitionString

    # Create a figure and axes that all data will be plotted
    fig, ax = plt.subplots(figsize=(6,4))

    plottedCols = [] # Keep track of whether any data matching "col" has been found
    print("Plotting log columns containing '{}'".format(columns))
    for logFilePath in logFilePaths:
        plottedCols += tryPlottingFromLog(logFilePath, columns, ax=ax, showPlot=False, columnsToExclude=plottedCols)
        
    if len(plottedCols) == 0:
        print("ERROR: no column names containing {} were found in the logFiles {}".format(plotDefinitionString, logFilePaths))    
    elif showPlot:        
        plt.show()

def tryPlottingFromLog(logPath, columnSpecs, columnsToExclude=[], ax=None, showPlot=True):
    '''
        Load the file at logPath, which should be either a simulationLog or a forceEvaluationLog
        Plots any columns whose names a) contain any of the strings in "columnNames", or b) match the regex defined by any of the strings in "columnNames" vs. time

        Returns True if a matching column was found and plotted, False if not

        Only data from the uppermost stage is plotted
    '''
    if ax == None:
        fig, ax = plt.subplots(figsize=(6,4))

    # Plot any column whose name includes any of the strings in columnNames
    if isinstance(columnSpecs, str):
        columnSpecs = [ columnSpecs ]

    if "Time(s)" not in columnSpecs:
        columnSpecs.append("Time(s)")

    # Get the matching column data
    data, names = getLoggedColumns(logPath, columnSpecs, columnsToExclude)

    # Grab x-axis
    for i in range(len(names)):
        if names[i] == "Time(s)":
            names.pop(i)
            x = data.pop(i)
            break

    # Plot data
    for i in range(len(names)):
        ax.plot(x, data[i], label=names[i])

    if len(names) > 0:
        ax.autoscale()
        ax.legend()

        if showPlot:
            plt.show()
    
    return names

def getLoggedColumns(logPath, columnSpecs, columnsToExclude=[], sep="\s+", enableCache=True):
    '''
        Obtains columns matching one or more regex expressions in a log file.
        By default, log file contents are cached by file name.

        Inputs:
            logPath:            (string) path to a simulationLog or forceEvaluationLog file
            columnSpecs:        (list (string)) list of partial/full column names and/or regex expressions, to identify the desired columns
            columnsToExcluse:   (list (string)) list of full column names to exclude
            sep:                (string) controls the separator pandas.read_csv uses to load the file ('\s+' / whitespace) for MAPLEAF's log files. "," for .csv

        Outputs:
            Returns: matchingColumnData (list (list (float))), matchingColumnNames (list (string))
    '''
    # Make sure columnSpecs is a list of strings - otherwise we will iterate over every character in the string and match everything
    if isinstance(columnSpecs, str):
        columnSpecs = [ columnSpecs ]
    
    if logPath in logFileCache and enableCache:
        df = logFileCache[logPath]
    else:
        if ".csv" in logPath:
            df = pd.read_csv(logPath, dtype=np.float64)
        
        else:
            # LEGACY TODO: Remove
            with open(logPath, 'r') as file:
                lines = file.readlines()

            #### Figure out which lines are not part of the data table ####
            if "simulationLog" in logPath:
                skipRows, skipFooter = _getSkipRowsAndFooters_mainSimLog(lines)
            elif "forceEvaluationLog" in logPath:
                skipRows, skipFooter = _getSkipRowsAndFooters_forceEvalLog(lines)
            else:
                skipRows, skipFooter = (0, 0)

            #### Read file, grab desired columns ####
            # Read file, skipping any lines found in the previous section
            df = pd.read_csv(logPath, sep=sep, skiprows=skipRows, dtype=np.float64, skipfooter=skipFooter, engine='python')
            
        logFileCache[logPath] = df
        if len(logFileCache) > maxLogFileCacheSize:
            logFileCache.popitem(last=False)
    
    matchingColumnData = []
    matchingColumnNames = []

    for columnSpec in columnSpecs:
        for colName in df.columns:
            if (columnSpec in colName or re.match(columnSpec, colName)) and (colName not in columnsToExclude):
                y = list(df[colName])
                
                matchingColumnData.append(y)
                matchingColumnNames.append(colName)

    return matchingColumnData, matchingColumnNames

def _getSkipRowsAndFooters_mainSimLog(lines):
    '''
        Pass in an array of strings, containing all lines of a mainSimulationLog file.
        Returns the skiprows and skipfooter parameters used in the call to pd.readcsv
        skiprows and skipfooter are intended to skip everything in the log file that is not part of the main data table
    '''
    i = 0
    while len(lines[i]) < 4 or lines[i][0:4] != "Time":
        i += 1
    skipRows = list(range(i))

    # If in simulation log, find out how many footer lines to skip
    simEndLine = lines.index("Simulation Complete\n")
    skipFooter = len(lines) - simEndLine

    # Also find the parachute deployment lines and skip those
    stageDroppingSimStartPt = None
    for i in range(skipRows[-1]+2, simEndLine):
        if len(lines[i].split()) > 0:
            try:
                float(lines[i].split()[0])
            except (ValueError, IndexError):
                if lines[i].split()[0] == "Computing":
                    stageDroppingSimStartPt = i
                skipRows.append(i)

    if stageDroppingSimStartPt != None:
        # Skip values calculated for dropped stages
        skipFooter = len(lines) - stageDroppingSimStartPt

    return skipRows, skipFooter

def _getSkipRowsAndFooters_forceEvalLog(lines):
    '''
        Pass in an array of strings, containing all lines of a forceEvaluationLog file.
        Returns the skiprows and skipfooter parameters used in the call to pd.readcsv
        skiprows and skipfooter are intended to skip everything in the log file that is not part of the main data table
    '''
    skipRows = 0

    lastTime = float(lines[1].split()[0])
    firstDroppedStageRow = None
    for row in range(2, len(lines)):
        currTime = float(lines[row].split()[0])
        if lastTime - currTime > 5: # Assume time steps are generally < 5sec and Flight/Drop durations are generally > 5 sec
            firstDroppedStageRow = row
            break

        lastTime = currTime
    
    if firstDroppedStageRow != None:
        skipFooter = len(lines) + 1 - firstDroppedStageRow
    else:
        # Unknown file - try reading all rows
        skipFooter = 0

    return skipRows, skipFooter

### Plot Flight Paths ###
def plotFlightPaths_NoEarth(flightPaths, showPlot=True):
    '''
        Pass in an iterable of RocketFlight objects. Their flight paths will be plotted in a 3D graph.
    '''
    print("Plotting flight paths")

    plt.figure()
    ax = plt.axes(projection='3d')

    allX = []
    allY = []
    allZ = []

    for flight in flightPaths:
        x = []
        y = []
        z = []

        for state in flight.rigidBodyStates:
            x.append(state.position.X)
            y.append(state.position.Y)
            z.append(state.position.Z)

        allX += x
        allY += y
        allZ += z

        ax.plot3D(x, y, z)

    allCoords = allX + allY + allZ
    maxCoord = max(allCoords)
    minCoord = min(allCoords)
    halfLength = (maxCoord - minCoord)/2

    ax.set_xlim(mean(allX) - halfLength, mean(allX) + halfLength)
    ax.set_ylim(mean(allY) - halfLength, mean(allY) + halfLength)
    ax.set_zlim(min(allZ), min(allZ) + 2*halfLength)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    if showPlot:
        plt.show()

def plot_Earth_Mayavi(earthTexture=r'C:\Users\taiya\anaconda3\envs\SOAR\Lib\site-packages\MAPLEAF\IO\blue_marble_spherical_splitFlipped.jpg'):
    from mayavi import mlab
    from tvtk.api import tvtk  # python wrappers for the C++ vtk ecosystem
    # create a figure window (and scene)
    fig = mlab.figure(size=(600, 600))

    # load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = getAbsoluteFilePath(earthTexture)
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    # (interpolate for a less raster appearance when zoomed in)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 6371007.1809
    Nrad = 180

    # create the sphere source with a given radius and angular resolution
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,
                                       phi_resolution=Nrad)

    # assemble rest of the pipeline, assign texture    
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)

def plotFlightPaths_FullEarth(flightPaths, showPlot=True):
    try:
        from mayavi import mlab
        from tvtk.api import tvtk  # python wrappers for the C++ vtk ecosystem
    
        plot_Earth_Mayavi()

        # Plot flight paths
        for flight in flightPaths:
            x = []
            y = []
            z = []

            for state in flight.rigidBodyStates:
                x.append(state.position.X)
                y.append(state.position.Y)
                z.append(state.position.Z)

            mlab.plot3d(x, y, z, tube_radius=20000, colormap='Spectral')

        if showPlot:
            mlab.show()

    except ModuleNotFoundError:
        print("Mayavi missing - falling back to plotting FlightPaths using matplotlib. Earth will not be shown.")
        plotFlightPaths_NoEarth(flightPaths)

### Flight Animation ###
def _keepNTimeSteps(flights, nFramesToKeep=600):
    '''
        Pass in list of RocketFlight objects and the number of frames/timesteps you'd like to keep
        Will return a matching list of Rocket Flight objects with linearly-interpolated, evenly spaced (in time) timesteps
    '''
    unpackResult = False
    if isinstance(flights, RocketFlight):
        unpackResult = True
        flights = [ flights ]

    maxLength = len(flights[0].times)
    
    # Find the max flight time of any stage
    maxTime = 0
    for flight in flights:
        if flight.times[-1] > maxTime:
            maxTime = flight.times[-1]

    newTimes = np.linspace(0, maxTime, num=nFramesToKeep)

    for flight in flights:
        # Interpolate flights to the times in newTimes
        interpStates = []

        interpCanardDefls = None
        if flight.actuatorDefls != None:
            nCanards = len(flight.actuatorDefls)
            interpCanardDefls = [ [] for i in range(nCanards) ]

        for time in newTimes:
            # SmallYIndex, SmallYWeight, largeYIndex, largeYWeight (for linear interpolation)
            smY, smYW, lgY, lgYW = linInterpWeights(flight.times, time)

            # Interpolate rigid body state
            if type(flight.rigidBodyStates[smY]) == type(flight.rigidBodyStates[lgY]):
                interpolatedState = interpolateRigidBodyStates(flight.rigidBodyStates[smY], flight.rigidBodyStates[lgY], smYW)
            else:
                # Handles the switch from 6DoF to 3DoF, where two adjacent states will be of different types
                interpolatedState = flight.rigidBodyStates[lgY] if lgYW > smYW else flight.rigidBodyStates[smY]

            interpStates.append(interpolatedState)

            # Interpolate canard deflections
            if flight.actuatorDefls != None:
                for i in range(nCanards):
                    # Interpolate the deflection of each canard
                    interpolatedDeflection = flight.actuatorDefls[i][smY]*smYW + flight.actuatorDefls[i][lgY]*lgYW
                    interpCanardDefls[i].append(interpolatedDeflection)
        
        flight.times = newTimes
        flight.rigidBodyStates = interpStates
        if flight.actuatorDefls != None:
            flight.actuatorDefls = interpCanardDefls

    if unpackResult:
        return flights[0]
    else:
        return flights

def _get3DPlotSize(flight, sizeMultiple=1.1):
    '''
        Finds max X, Y, or Z distance from the origin reached during the a flight. Used to set the 3D plot size (which will be equal in all dimensions)
    '''
    Positions = flight.Positions
    centerOfPlot = Vector(mean(Positions[0]), mean(Positions[1]), mean(Positions[2]))

    xRange = max(Positions[0]) - min(Positions[0])
    yRange = max(Positions[1]) - min(Positions[1])
    zRange = max(Positions[2]) - min(Positions[2])

    if max(xRange, yRange, zRange) == 0:
        # For cases where the object does not move
        axisDimensions = 1.0
    else:
        axisDimensions = max([xRange, yRange, zRange]) * sizeMultiple
    
    return axisDimensions, centerOfPlot

def _findEventTimeStepNumber(flight, time):
    '''
        Given a time and a RocketFlight object, finds the time step that passes the given time
    '''
    if time == None:
        return None
    else:
        return bisect_right(flight.times, time) # Binary search for time value in 

def _createReferenceVectors(nCanards, maxAbsCoord, rocketLengthFactor=0.25, finLengthFactor=0.05):
    '''
        Creates a longitudinal vector and an array of nCanards perpendicular vectors. 
        These represent the rocket in the local frame and are rotated according to it's rigid body state at each time step.
        The size of the longitudinal and perpendicular lines are controlled by the rocketLength and finLength factor arguments and the maxAbsCoord.
    '''
    # Create vector reoresenting longitudinal axis in local frame
    refAxis = Vector( 0, 0, maxAbsCoord*rocketLengthFactor )
    
    # Create vectors going perpedicularly out, in the direction of each fin/canard in local frame
    radiansPerFin = radians(360 / nCanards)
    finToFinRotation = Quaternion(axisOfRotation=Vector(0,0,1), angle=radiansPerFin)
    perpVectors = [ Vector( maxAbsCoord*finLengthFactor, 0, 0 ) ]
    for i in range(nCanards-1):
        newVec = finToFinRotation.rotate(perpVectors[-1])
        perpVectors.append(newVec)

    return refAxis, perpVectors

def _createAnimationFigure(axisDimensions, centerOfPlot):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    halfDim = axisDimensions/2
    
    ax.set_xlim3d([centerOfPlot[0] - halfDim, centerOfPlot[0] + halfDim])
    ax.set_ylim3d([centerOfPlot[1] - halfDim, centerOfPlot[1] + halfDim])
    ax.set_zlim3d([centerOfPlot[2] - halfDim, centerOfPlot[2] + halfDim])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    return fig, ax

def _createInitialFlightAnimationPlot_singleRocket(ax, nCanards, flight, refAxis, perpVectors):
    '''
        Called once for each rocket to set up the animation.
        Creates all the lines required for the rocket (longitudinal + perpendiculars + canards)
        These are then modified during each time step
    '''
    # Create flight path line
    Positions = flight.Positions
    flightPath = ax.plot(Positions[0][0:1], Positions[1][0:1], Positions[2][0:1])[0]
    if flight.engineOffTime != None:
        flightPath.set_color("red")
    else:
        flightPath.set_color("black")
    
    colors = ["blue", "red", "green", "purple", "brown", "grey", "peru", "skyblue", "pink", "darkblue", "darkgreen" ]

    # Create rocket/fin lines
    cg, rocketBase, tip, perpTips, canardTips, canardTails = _getRocketPoints(0, flight, refAxis, perpVectors)
    cgPoint = ax.scatter3D([ cg.X ], [ cg.Y ], [ cg.Z ])
    longitudinalLine = ax.plot([rocketBase.X, tip.X], [rocketBase.Y, tip.Y], [rocketBase.Z, tip.Z])[0]
    
    # Create fin and canard lines
    canardLines = []
    perpLines = []
    for i in range(nCanards):
        perpLines.append(ax.plot([rocketBase.X, perpTips[i].X], [rocketBase.Y, perpTips[i].Y], [rocketBase.Z, perpTips[i].Z])[0])
        perpLines[-1].set_color(colors[i % len(colors)])
        canardLines.append(ax.plot([canardTips[i].X, canardTails[i].X], [canardTips[i].Y, canardTails[i].Y], [canardTips[i].Z, canardTails[i].Z])[0])
        canardLines[-1].set_color(colors[i % len(colors)])

    # Plot target location if it exists
    target = flight.targetLocation
    if target != None:
        ax.scatter3D( [ target.X ], [ target.Y ], [ target.Z ])

    return cgPoint, flightPath, longitudinalLine, canardLines, perpLines

def _createInitialFlightAnimationPlot(ax, nCanards, flights, refAxis, perpVectors):
    '''
        Loops through all the rockets, calls the _singleRocket version of this function for each one.
    '''
    cgPoints = []
    flightPathLines = []
    longitudinalRocketLines = []
    allCanardLines = []
    allPerpLines = []
    for flight in flights:
        cgPoint, flightPath, longitudinalLine, canardLines, perpLines = _createInitialFlightAnimationPlot_singleRocket(ax, nCanards, flight, refAxis, perpVectors)
        cgPoints.append(cgPoint)
        flightPathLines.append(flightPath)
        longitudinalRocketLines.append(longitudinalLine)
        allCanardLines.append(canardLines)
        allPerpLines.append(perpLines)
    return cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines

def _getRocketPoints(timeStepNumber, flight, refAxis, perpVectors):
    '''
        For each rocket and time step, called to find the coordinates of the:
            cg, rocketBase, tip, perpTips, canardTips, canardTails
    '''
    nCanards = len(perpVectors)
    try:
        # Try plotting the actual rocket - using the orienation (only present in a 6DoF sim)
        cg = flight.rigidBodyStates[timeStepNumber].position
        currOrientation = flight.rigidBodyStates[timeStepNumber].orientation 

        axisVector = currOrientation.rotate(refAxis)
        perpVectors = [ currOrientation.rotate(x) for x in perpVectors ]
        
        # Assume rocket's CG is about halfway along it's axis
        rocketBase = flight.rigidBodyStates[timeStepNumber].position - axisVector*0.5        
        tip = rocketBase + axisVector
        perpTips = [ rocketBase + x for x in perpVectors ]

        if flight.actuatorDefls != None:
            # Get lengthwise center of each canard
            canardLongitudinalLocation = tip - axisVector*0.25
            canardCenters = [ canardLongitudinalLocation + x*0.5 for x in perpVectors ]

            # Get lengthwise vector of canard
            canardVecs = [ Quaternion(axisOfRotation=perpVectors[i], angle=radians(flight.actuatorDefls[i][timeStepNumber])).rotate(axisVector) for i in range(nCanards) ]

            canardRocketLengthFraction = 0.2
            canardTips = [ canardCenters[i] + canardVecs[i]*canardRocketLengthFraction*0.25 for i in range(nCanards) ]
            canardTails = [ canardCenters[i] - canardVecs[i]*canardRocketLengthFraction*0.75 for i in range(nCanards) ]
        else:
            # Initialize lines to zero co-ordinates - these lines can be used to plot the parachutes later, if necessary
            canardTips = []
            canardTails = []
            for i in range(nCanards):
                canardTips.append(Vector(0,0,0))
                canardTails.append(Vector(0,0,0))

    except AttributeError:
        # Otherwise plot parachute for 3DoF simulation (3DoF only used for descent)
        rocketBase = flight.rigidBodyStates[timeStepNumber].position   
        cg = flight.rigidBodyStates[timeStepNumber].position        
        axisVector = refAxis*0.75
        tip = rocketBase - refAxis*0.2

        if flight.mainChuteDeployTime == None or timeStepNumber < flight.mainChuteTimeStep:
            chuteSize = 0.35
        else:
            chuteSize = 1.0

        perpTips = [ rocketBase + x*chuteSize + axisVector for x in perpVectors ]
        canardTips = perpTips

        parachuteTip = rocketBase + axisVector*1.2
        canardTails = [parachuteTip]*nCanards

    return cg, rocketBase, tip, perpTips, canardTips, canardTails



def _update_plot(timeStepNumber, flights, refAxis, perpVectors, cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines):
    ''' 
        Plot Update function - This gets called every time step of the simulation, updates the data for each point and line in the plot
    '''

    rocketColors = [ 'black', 'blue', 'fuchsia', 'olive', 'maroon', 'purple', 'sienna' ]

    for i in range(len(flights)):
        # Get important rocket coordinates at current time step
        cg, rocketBase, tip, perpTips, canardTips, canardTails = _getRocketPoints(timeStepNumber, flights[i], refAxis, perpVectors)
        nCanards = len(perpVectors)
        
        # Extract data for current stage to simplify code below
        flightPath = flightPathLines[i]
        Positions = flights[i].Positions
        engineOffTimeStep = flights[i].engineOffTimeStep
        cgPoint = cgPoints[i]
        longitudinalLine = longitudinalRocketLines[i]
        perpLines = allPerpLines[i]
        canardLines = allCanardLines[i]

        def setRocketColor(clr):
            flightPath.set_color(clr)
            cgPoint.set_color(clr)
            longitudinalLine.set_color(clr)
            for line in perpLines:
                line.set_color(clr)
            for line in canardLines:
                line.set_color(clr)

        if timeStepNumber == 0:
            clr = rocketColors[i % len(rocketColors)]
            setRocketColor(clr)
            flightPath.set_color('black')    

        # Update Flight path
        flightPath.set_data(Positions[0][:timeStepNumber], Positions[1][:timeStepNumber])
        flightPath.set_3d_properties(Positions[2][:timeStepNumber])
        if i > 0:
            if timeStepNumber < engineOffTimeStep:
                flightPath.set_color("red")
                flightPath.set_alpha(1)
            else:
                flightPath.set_color("gray")
                flightPath.set_alpha(0.5)
        elif engineOffTimeStep == None or timeStepNumber >= engineOffTimeStep:
            flightPath.set_color("black")
        else:
            flightPath.set_color("red")
        
        # Update rocket CG and main line
        cgPoint._offsets3d = ([ cg.X ], [ cg.Y ], [ cg.Z ])
        longitudinalLine.set_data([rocketBase.X, tip.X], [rocketBase.Y, tip.Y])
        longitudinalLine.set_3d_properties([rocketBase.Z, tip.Z])

        # Update fins and canards
        for c in range(nCanards):
            # Update tail fins / orientation indicators / parachute
            perpLines[c].set_data([rocketBase.X, perpTips[c].X], [rocketBase.Y, perpTips[c].Y])
            perpLines[c].set_3d_properties([rocketBase.Z, perpTips[c].Z])
            # Update canards / parachute
            canardLines[c].set_data([canardTips[c].X, canardTails[c].X], [canardTips[c].Y, canardTails[c].Y])
            canardLines[c].set_3d_properties([canardTips[c].Z, canardTails[c].Z])

        # Turn lines gray if landed
        if cg == flights[i].rigidBodyStates[-1].position:
            setRocketColor('gray')

def flightAnimation(flights, showPlot=True, saveAnimFileName=None):
    '''
        Pass in a list of RocketFlight object(s). Intended to contain a single RocketFlight object, or multiple if it was a staged flight (one object per stage)
        showPlot controls where the animation is shown or not
        saveAnimFileName should be a string file name/path that the animation should be saved to ex: "SampleRocketFlightAnimation.mp4"
    '''
    #### Set up data for animation ####
    # Filter out extra frames - seems like there's a limit to the number of frames that will work well with matplotlib, otherwise the end of the animation is weirdly sped up
    flights = _keepNTimeSteps(flights, nFramesToKeep=900)

    # Transform position info into arrays of x, y, z coordinates
    for flight in flights:
        Positions = [ [], [], [] ]
        for state in flight.rigidBodyStates:
            for i in range(3):
                Positions[i].append(state.position[i])
        flight.Positions = Positions

    # Set xyz size of plot - equal in all dimensions
    axisDimensions, centerOfPlot = _get3DPlotSize(flights[0]) # Assumes the top stage travels the furthest

    # Calculate frames at which engine turns off and main chute deploys
    for flight in flights:
        flight.engineOffTimeStep = _findEventTimeStepNumber(flight, flight.engineOffTime)
        flight.mainChuteTimeStep = _findEventTimeStepNumber(flight, flight.mainChuteDeployTime)

    if flights[0].actuatorDefls != None:
        # Assume all actuated systems are canards #TODO: This needs updating
        # Assuming canards always on the top stage
        nCanards = len(flights[0].actuatorDefls)
    else:
        nCanards = 4 # Create canard lines to reuse them for the parachute

    refAxis, perpVectors = _createReferenceVectors(nCanards, axisDimensions)

    #### Create Initial Plot ####
    fig, ax = _createAnimationFigure(axisDimensions, centerOfPlot)
    cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines = _createInitialFlightAnimationPlot(ax, nCanards, flights, refAxis, perpVectors)
#################################################################################################################################################################################################################################################
    # ---- Apogee AGL legend (meters + feet) ---- #Aneesh Start
    from matplotlib.offsetbox import AnchoredText

    Z = np.asarray(flights[0].Positions[2], dtype=float)
    z0 = float(Z[0]) if Z.size else 0.0                   # ground/launch altitude
    ap_asl_m = float(np.max(Z)) if Z.size else 0.0        # (not shown, kept for reference)
    ap_agl_m = float(np.max(Z - z0)) if Z.size else 0.0   # AGL
    ap_agl_ft = ap_agl_m * 3.28084




    # Clean title + compact legend box
    ax.set_title(f"Apogee AGL: {ap_agl_m:.1f} m ({ap_agl_ft:.0f} ft)")
    _ap_box = AnchoredText(
        f"Apogee AGL\n{ap_agl_m:.1f} m / {ap_agl_ft:.0f} ft",
        loc="upper right",
        prop=dict(size=9),
        frameon=True,
        borderpad=0.4
    )
    _ap_box.patch.set_alpha(0.85)
    ax.add_artist(_ap_box)
    # ---- Aneesh End ----

#################################################################################################################################################################################################################################################
 

    # Play animation
    #ani = animation.FuncAnimation(fig, _update_plot, frames=(len(Positions[0]) - 1), fargs=(flights, refAxis, perpVectors, cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines), interval=1, blit=False, save_count=0, repeat_delay=5000)
    ani = animation.FuncAnimation(
        fig,
        _update_plot,
        frames=(len(Positions[0]) - 1),
        fargs=(flights, refAxis, perpVectors, cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines),
        interval=1,
        blit=False,
        repeat_delay=5000
)

    if showPlot:
        plt.show()

    if saveAnimFileName != None:
        ani.save("SampleRocket.mp4", bitrate=2500, fps=60)

### Plot Monte Carlo Results ###
def plotAndSummarizeVectorResult(vectorList, name="Landing location", monteCarloLogger=None, showPlot=True):
    '''
        Plots 3D scatter plot of vector result, outputs data summary to console.
    '''
    x = []
    y = []
    z = []

    consoleOutput = []

    consoleOutput.append("{}s:   X       Y        Z".format(name))
    for i in range(len(vectorList)):
        loc = vectorList[i]
        consoleOutput.append("Simulation #{}:   {:>7.1f}".format(i+1, loc))
        x.append(loc.X)
        y.append(loc.Y)
        z.append(loc.Z)

    consoleOutput.append("")
    consoleOutput.append("Mean {}: ({:>8.1f} {:>8.1f} {:>8.1f})".format(name, mean(x), mean(y), mean(z)))
    consoleOutput.append("Standard deviation:    ({:>8.1f} {:>8.1f} {:>8.1f})".format(stdev(x), stdev(y), stdev(z)))
    consoleOutput.append("")

    for line in consoleOutput:
        if monteCarloLogger != None:
            monteCarloLogger.log(line)
        else:
            print(line)

    print("Plotting {}".format(name))     

    maxCoord = max(x + y + z)
    minCoord = min(x + y + z)
    halfLength = (maxCoord - minCoord)/2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)

    ax.set_xlim(mean(x) - halfLength, mean(x) + halfLength)
    ax.set_ylim(mean(y) - halfLength, mean(y) + halfLength)
    ax.set_zlim(min(z), min(z) + 2*halfLength)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    ax.set_title(name)

    if showPlot:
        plt.show()

def plotAndSummarizeScalarResult(scalarList, name="Apogee", monteCarloLogger=None, showPlot=True):
    '''
        Plots Histogram of scalar result, outputs data summary to console
    '''
    consoleOutput = []

    consoleOutput.append("{}s:".format(name))
    for i in range(len(scalarList)):
        consoleOutput.append("Simulation #{}: {}".format(i+1, scalarList[i]))

    consoleOutput.append("")
    consoleOutput.append("Mean {}:        {:>8.1f}".format(name, mean(scalarList)))
    consoleOutput.append("Standard Deviation: {:>8.1f}".format(stdev(scalarList)))
    consoleOutput.append("")

    for line in consoleOutput:
        if monteCarloLogger != None:
            monteCarloLogger.log(line)
        else:
            print(line)

    print("Plotting {}s".format(name))

    plt.hist(scalarList)
    plt.xlabel(name)
    plt.ylabel("Count")

    if showPlot:
        plt.show()

### Flight summary ('plot summary') ###
def _getMotor(rocket):
    ''' Returns the first stage motor found on the rocket, or None '''
    for stage in getattr(rocket, "stages", []):
        motor = getattr(stage, "motor", None)
        if motor is not None:
            return motor
    return None


def _detectLiquidCutoff(times, thrusts):
    '''
        Find the 'knee' where a self-pressurizing hybrid runs out of LIQUID oxidizer and
        drops onto vapour blowdown. Returns None if there is no clear knee.

        Only used when the motor file does not declare LiquidCutoffTime (Thyme writes it).
        It is deliberately CONSERVATIVE - it would rather miss a real knee than label a
        solid motor's ignition spike or tail-off as a liquid cutoff, because a wrong
        number here would mis-size an oxidizer fill.

        A genuine liquid cutoff has three features together, and solid/pressure-fed
        motors fail at least one:
          1. a sustained high-thrust plateau before it (not an ignition spike),
          2. a sharp step down partway through the burn,
          3. a long, low, decaying tail afterwards (the vapour blowdown).
    '''
    if len(times) < 6:
        return None
    t0, tEnd = times[0], times[-1]
    burnTime = tEnd - t0
    peak = max(thrusts)
    if burnTime <= 0 or peak <= 0:
        return None

    def thrustAt(t):
        if t <= times[0]:
            return thrusts[0]
        if t >= times[-1]:
            return thrusts[-1]
        for i in range(1, len(times)):
            if times[i] >= t:
                span = times[i] - times[i-1]
                f = (t - times[i-1]) / span if span > 0 else 0.0
                return thrusts[i-1] + f * (thrusts[i] - thrusts[i-1])
        return thrusts[-1]

    # (3) the tail must actually decay away - excludes motors with a flat sustain phase
    if thrusts[-1] > 0.25 * peak:
        return None

    window = 0.08 * burnTime          # step is measured over this much time
    best, bestDrop = None, 0.0
    for i, t in enumerate(times):
        frac = (t - t0) / burnTime
        # (1)+(3) knee must sit in the middle of the burn: past the ignition transient,
        # and leaving at least 25% of the burn as vapour tail. The upper bound is what
        # separates a liquid cutoff from a solid motor's terminal tail-off.
        if frac < 0.35 or frac > 0.75:
            continue
        before = thrusts[i]
        if before < 0.55 * peak:
            continue                  # not on the main plateau any more
        # sustained plateau check: mean thrust over the preceding 15% of the burn
        preStart = max(t0, t - 0.15 * burnTime)
        preSamples = [th for tt, th in zip(times, thrusts) if preStart <= tt <= t]
        if not preSamples or (sum(preSamples) / len(preSamples)) < 0.60 * peak:
            continue
        drop = (before - thrustAt(t + window)) / before
        if drop > bestDrop:
            best, bestDrop = t, drop

    # (2) the step must be a real cliff, not the general decline
    if best is None or bestDrop < 0.30:
        return None
    return best


def _motorStats(motor):
    '''
        Returns (burnTime, burnoutTime, avgThrust, peakThrust, totalImpulse,
                 liquidCutoffTime, liquidCutoffSource) from a TabulatedMotor.
        liquidCutoffTime is None for motors with no liquid/vapour transition.
    '''
    times, thrusts = motor.times, motor.thrustLevels
    burnTime = times[-1] - times[0]
    totalImpulse = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        totalImpulse += dt * (thrusts[i] + thrusts[i-1]) / 2   # trapezoid, matches linear thrust interp
    avgThrust = totalImpulse / burnTime if burnTime > 0 else float('nan')
    burnoutTime = motor.ignitionTime + burnTime

    # Prefer a value declared in the motor file (Thyme writes LiquidCutoffTime);
    # otherwise try to find the knee in the thrust curve
    declared = getattr(motor, 'liquidCutoffTime', None)
    if declared is not None:
        liquidCutoff, source = declared, 'declared'
    else:
        detected = _detectLiquidCutoff(times, thrusts)
        liquidCutoff, source = (detected, 'detected') if detected is not None else (None, None)

    return burnTime, burnoutTime, avgThrust, max(thrusts), totalImpulse, liquidCutoff, source


def _stateAtRailExit(flight, railLength):
    ''' Returns (time, state) at the first flight state that has travelled >= railLength from the start '''
    states, times = flight.rigidBodyStates, flight.times
    if len(states) < 2 or railLength <= 0:
        return None   # no (positive-length) rail -> off-rail metrics are not meaningful; caller shows n/a
    start = states[0].position
    for i in range(len(states)):
        if (states[i].position - start).length() >= railLength:
            return times[i], states[i]
    return None


def _maxAccel_g(flight):
    ''' Peak |acceleration| over the flight, in g, by differencing the logged velocity '''
    states, times = flight.rigidBodyStates, flight.times
    maxA = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        if dt <= 0:
            continue
        a = (states[i].velocity - states[i-1].velocity).length() / dt
        maxA = max(maxA, a)
    return maxA / 9.80665


def _maxMachFromLog(logFilePaths):
    ''' Reads the peak Mach from the derivative-evaluation log if available, else NaN '''
    if not logFilePaths:
        return float('nan')
    for path in logFilePaths:
        try:
            data, names = getLoggedColumns(path, ["Mach"])
        except Exception:
            continue
        for name, col in zip(names, data):
            if name == "Mach" and len(col) > 0:
                return max(col)
    return float('nan')


def plotFlightSummary(rocket, flight, logFilePaths, simDefinition, showPlot=True):
    '''
        'plot summary' handler. Prints and plots a one-page flight summary:
        motor burn time / average thrust, off-rail velocity + static margin (CG & CP at rail exit),
        apogee AGL, max speed / Mach / acceleration, and a verification of the configured wind.
    '''
    import math

    # --- Motor: burn time & average thrust ---
    motor = _getMotor(rocket)
    if motor is not None and len(getattr(motor, "times", [])) >= 2:
        burnTime, burnoutTime, avgThrust, peakThrust, totalImpulse, \
            liquidCutoff, liquidCutoffSource = _motorStats(motor)
    else:
        burnTime = burnoutTime = avgThrust = peakThrust = totalImpulse = float('nan')
        liquidCutoff = liquidCutoffSource = None

    # --- Off-rail static margin (CG & CP at the instant the rocket clears the rail) ---
    try:
        railLength = float(simDefinition.getValue("Environment.LaunchSite.railLength"))
    except Exception:
        railLength = 0.0
    offRail = _stateAtRailExit(flight, railLength)
    if offRail is not None:
        tOR, stateOR = offRail
        offRailSpeed = stateOR.velocity.length()
        try:
            cgOR, cpOR, marginOR = rocket.getStaticMargin(tOR, stateOR)
        except Exception:
            cgOR = cpOR = marginOR = float('nan')
    else:
        tOR = offRailSpeed = cgOR = cpOR = marginOR = float('nan')

    # --- General flight metrics ---
    apogeeASL = flight.getApogee()
    # AGL is referenced to ground elevation (MAPLEAF's convention: AGL = ASL - launchSiteElevation),
    # NOT the rocket's initial CG altitude (which sits a few m above the pad).
    try:
        launchAltitude = float(simDefinition.getValue("Environment.LaunchSite.elevation"))
    except Exception:
        launchAltitude = flight.rigidBodyStates[0].position.Z
    apogeeAGL = apogeeASL - launchAltitude
    maxSpeed = flight.getMaxSpeed()
    flightTime = flight.getFlightTime()
    maxAccel = _maxAccel_g(flight)
    maxMach = _maxMachFromLog(logFilePaths)

    # --- Wind verification ---
    try:
        windVec = Vector(simDefinition.getValue("Environment.ConstantMeanWind.velocity"))
        windSpeed = windVec.length()
        windDesc = "{:.2f} m/s   (East {:+.2f}, North {:+.2f})".format(windSpeed, windVec.X, windVec.Y)
    except Exception:
        windDesc = "N/A (non-constant wind model)"
    try:
        turbModel = simDefinition.getValue("Environment.TurbulenceModel")
    except Exception:
        turbModel = "None"

    def _f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "   n/a"

    def _mft(v):  # metres and feet
        return "{:8.1f} m   ({:8.0f} ft)".format(v, v * 3.28084) if isinstance(v, (int, float)) and v == v else "   n/a"

    def _impulseUpTo(motor, tCut):
        ''' Trapezoidal impulse from ignition to tCut, interpolating the final partial step. '''
        times, thrusts = motor.times, motor.thrustLevels
        total = 0.0
        for i in range(1, len(times)):
            if times[i] <= tCut:
                total += (times[i] - times[i-1]) * (thrusts[i] + thrusts[i-1]) / 2
            else:
                span = times[i] - times[i-1]
                if span > 0 and times[i-1] < tCut:
                    frac = (tCut - times[i-1]) / span
                    tAt = thrusts[i-1] + frac * (thrusts[i] - thrusts[i-1])
                    total += (tCut - times[i-1]) * (thrusts[i-1] + tAt) / 2
                break
        return total

    rocketName = getattr(rocket, "name", "Rocket")
    lines = [
        "==================== FLIGHT SUMMARY: {} ====================".format(rocketName),
        "",
        " Motor",
        "   Burn time (thrust cutoff) : {}".format(_f(burnTime,   "{:8.2f} s")),
    ]
    if liquidCutoff is not None:
        # For a self-pressurizing hybrid, thrust cutoff includes the vapour blowdown
        # tail. Liquid cutoff is the number that sizes the oxidizer fill.
        liqImpulse = _impulseUpTo(motor, liquidCutoff)
        lines += [
            "   Liquid cutoff ({:8s})  : {}".format(liquidCutoffSource,
                                                    _f(liquidCutoff, "{:8.2f} s")),
            "   Vapour tail duration      : {}".format(_f(burnTime - liquidCutoff, "{:8.2f} s")),
            "   Impulse before liq cutoff : {}".format(_f(liqImpulse, "{:8.1f} N.s")),
        ]
    lines += [
        "   Average thrust            : {}".format(_f(avgThrust,  "{:8.1f} N")),
        "   Peak thrust               : {}".format(_f(peakThrust, "{:8.1f} N")),
        "   Total impulse             : {}".format(_f(totalImpulse, "{:8.1f} N.s")),
        "",
        " Off the rail  (rail length {})".format(_f(railLength, "{:.2f} m").strip()),
        "   Time to clear rail        : {}".format(_f(tOR,         "{:8.2f} s")),
        "   Velocity off rail         : {}".format(_f(offRailSpeed,"{:8.2f} m/s")),
        "   CG off rail  (z)          : {}".format(_f(cgOR,        "{:8.4f} m")),
        "   CP off rail  (z)          : {}".format(_f(cpOR,        "{:8.4f} m")),
        "   Static margin off rail    : {}".format(_f(marginOR,    "{:8.2f} cal")),
        "",
        " Trajectory",
        "   Apogee (AGL)              : {}".format(_mft(apogeeAGL)),
        "   Apogee (ASL)              : {}".format(_mft(apogeeASL)),
        "   Max speed                 : {}".format(_f(maxSpeed,   "{:8.1f} m/s")),
        "   Max Mach                  : {}".format(_f(maxMach,    "{:8.2f}")),
        "   Max acceleration          : {}".format(_f(maxAccel,   "{:8.2f} g")),
        "   Flight time               : {}".format(_f(flightTime, "{:8.1f} s")),
        "",
        " Wind (verification)",
        "   Mean wind                 : {}".format(windDesc),
        "   Turbulence model          : {}".format(turbModel),
        "=" * 62,
    ]
    summaryText = "\n".join(lines)
    print(summaryText)

    # --- Figure: text card + thrust curve + altitude trace ---
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    axText = fig.add_subplot(gs[:, 0]); axText.axis('off')
    axText.text(0.0, 1.0, summaryText, va='top', ha='left', family='monospace', fontsize=9,
                transform=axText.transAxes)

    # Thrust curve with burnout marked
    axThrust = fig.add_subplot(gs[0, 1])
    if motor is not None and len(getattr(motor, "times", [])) >= 2:
        axThrust.plot(motor.times, motor.thrustLevels, color="#C44E52")
        if avgThrust == avgThrust:
            axThrust.axhline(avgThrust, color="0.4", ls="--", lw=1, label="avg = {:.0f} N".format(avgThrust))
        if burnoutTime == burnoutTime:
            axThrust.axvline(motor.times[-1], color="k", ls=":", lw=1, label="burnout {:.2f} s".format(burnTime))
        if liquidCutoff is not None:
            axThrust.axvline(liquidCutoff, color="#1f77b4", ls="-.", lw=1.4,
                             label="liquid cutoff {:.2f} s".format(liquidCutoff))
            axThrust.axvspan(liquidCutoff, motor.times[-1], color="#1f77b4", alpha=0.08)
        axThrust.set_xlabel("Time (s)"); axThrust.set_ylabel("Thrust (N)")
        axThrust.set_title("Motor thrust curve"); axThrust.legend(fontsize=8)
    else:
        axThrust.axis('off'); axThrust.text(0.5, 0.5, "No motor thrust data", ha='center', va='center')

    # Altitude AGL with rail-exit / burnout / apogee markers
    axAlt = fig.add_subplot(gs[1, 1])
    altAGL = [st.position.Z - launchAltitude for st in flight.rigidBodyStates]
    axAlt.plot(flight.times, altAGL, color="#4C72B0")
    if tOR == tOR:
        axAlt.axvline(tOR, color="green", ls=":", lw=1, label="rail exit")
    if liquidCutoff is not None and liquidCutoff <= flight.times[-1]:
        axAlt.axvline(liquidCutoff, color="#1f77b4", ls="-.", lw=1.2, label="liquid cutoff")
    if burnoutTime == burnoutTime and burnoutTime <= flight.times[-1]:
        axAlt.axvline(burnoutTime, color="red", ls=":", lw=1, label="burnout")
    apogeeIdx = max(range(len(altAGL)), key=lambda i: altAGL[i])
    axAlt.axvline(flight.times[apogeeIdx], color="purple", ls=":", lw=1, label="apogee")
    axAlt.set_xlabel("Time (s)"); axAlt.set_ylabel("Altitude AGL (m)")
    axAlt.set_title("Altitude"); axAlt.legend(fontsize=8)

    fig.tight_layout()
    if showPlot:
        plt.show()


def plotMonteCarloSummary(apogees, windSpeeds, monteCarloLogger=None, showPlot=True):
    '''
        Monte Carlo 'summary' output: apogee distribution with mean / standard deviation / variance,
        plus an apogee-vs-wind-speed scatter to show the effect of the (varied) wind.
    '''
    ap = np.asarray(apogees, dtype=float)
    n = ap.size
    m = float(np.mean(ap)) if n else float('nan')
    s = float(np.std(ap, ddof=1)) if n > 1 else 0.0
    variance = s * s

    lines = [
        "",
        "Monte Carlo apogee summary ({} runs):".format(n),
        "  Mean apogee         : {:.1f} m".format(m),
        "  Standard deviation  : {:.2f} m".format(s),
        "  Variance            : {:.2f} m^2".format(variance),
    ]
    if s > 0 and m == m and m != 0:
        lines.append("  Coeff. of variation : {:.3f} %".format(100 * s / m))
    for line in lines:
        if monteCarloLogger is not None:
            monteCarloLogger.log(line)
        else:
            print(line)

    print("Plotting Monte Carlo apogee summary")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Apogee distribution with mean and +/- sigma bands (variance)
    nbins = min(20, max(5, n // 3)) if n > 0 else 5
    ax1.hist(ap, bins=nbins, color="#4C72B0", edgecolor="white", alpha=0.85)
    if m == m:
        ax1.axvline(m, color="k", lw=1.5, label="mean = {:.0f} m".format(m))
    if s > 0:
        for k, alpha in ((1, 0.25), (2, 0.15)):
            ax1.axvspan(m - k * s, m + k * s, color="orange", alpha=alpha, label="±{}σ".format(k))
    ax1.set_xlabel("Apogee (m ASL)"); ax1.set_ylabel("Count")
    ax1.set_title("Apogee distribution\nσ = {:.1f} m   (variance {:.0f} m²)".format(s, variance))
    ax1.legend(fontsize=8)

    # Apogee vs wind speed
    ws = np.asarray(windSpeeds, dtype=float) if windSpeeds is not None else np.array([])
    if ws.size == ap.size and ws.size > 0 and np.isfinite(ws).any():
        ax2.scatter(ws, ap, s=18, color="#4C72B0", alpha=0.7)
        ax2.set_xlabel("Mean wind speed (m/s)"); ax2.set_ylabel("Apogee (m ASL)")
        ax2.set_title("Apogee vs wind")
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "No per-run wind captured", ha='center', va='center')

    fig.tight_layout()
    if showPlot:
        plt.show()


### Colors ###
def getColorPalette(nColors=4):
    ''' Returns a list of nColors colors '''
    if nColors > 8:
        raise ValueError("Too many colors requested: {} (Max 8)".format(nColors))
    if nColors < 3:
        raise ValueError("Too few colors requested: {} (Min 3)".format(nColors))

    colorPaletteDict = {        
        3: [ '#003f5c', '#bc5090', '#ffa600'],
        4: [ '#003f5c', '#7a5195', '#ef5675', '#ffa600'],
        5: [ '#003f5c', '#58508d', '#bc5090', '#ff6361','#ffa600'],
        6: [ '#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600'],
        7: [ '#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675', '#ff764a', '#ffa600'],
        8: [ '#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600' ]
    }

    return colorPaletteDict[nColors]