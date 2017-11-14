from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurement, a pair of robot pose, i.e. last time
                step pose and current time step pose

        Returns: the list of particle represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """ 
    newParticles = []

    oldPos = odom[0]
    newPos = odom[1]
    alpha1 = 0.02
    alpha2 = 0.02
    alpha3 = 0.02
    alpha4 = 0.02

    rot1 = math.atan2(newPos[1] - oldPos[1], newPos[0] - oldPos[0]) - oldPos[2]
    trans = grid_distance(oldPos[0],oldPos[1],newPos[0], newPos[1])
    rot2 = newPos[2] - oldPos[2] - rot1

    newRot1 = rot1 - add_gaussian_noise(alpha1 * rot1 + alpha2 * trans, ODOM_HEAD_SIGMA)
    newTrans = trans - add_gaussian_noise(alpha3 * trans + alpha4 * (rot1 + rot2), ODOM_TRANS_SIGMA)
    newRot2 = rot2 - add_gaussian_noise(alpha1 * rot2 + alpha2 * trans , ODOM_HEAD_SIGMA)

    for particle in particles:
        particle.x = particle.x + newTrans * math.cos(math.radians(particle.h + newRot1))
        particle.y = particle.y + newTrans * math.sin(math.radians(particle.h + newRot1))
        particle.h = particle.h + newRot1 + newRot2

        newParticles.append(particle)

    return newParticles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map, which contains the marker information, 
                see grid.h and CozGrid for definition

        Returns: the list of particle represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    counter = 0
    weightArr = []

    if len(measured_marker_list) != 0:
        for particle in particles:
            visibleMarkers = particle.read_markers(grid)

            shouldAppendParticle = particle.x >= grid.width or particle.x < 0 or particle.y >= grid.height or particle.y < 0 or (particle.x, particle.y) in grid.occupied
            if shouldAppendParticle:
                weightArr.append((particle, 0))
            else:
                mmlLength = len(measured_marker_list)
                vmLength = len(visibleMarkers)
                pairs = []
                for measuredMarker in measured_marker_list:
                    if len(visibleMarkers) != 0:
                        nearestMarker = findNearestMarker(measuredMarker, visibleMarkers)
                        visibleMarkers.remove(nearestMarker)
                        pairs.append((nearestMarker, measuredMarker))

                probability = getProbability(pairs, mmlLength, vmLength)
                weightArr.append((particle, probability))

        counter2 = 0
        remove = int(PARTICLE_COUNT / 100)
        weightArr.sort(key=lambda x: x[1])
        weightArr = weightArr[remove:]
        for i, j in weightArr:
            if j != 0:
                counter2 += j
            else:
                counter += 1
        weightArr = weightArr[counter:]
        counter += remove
    else:
        counter2 = 1
        for p in particles:
            weightArr.append((p, 1 / len(particles)))

    particleList = []
    weightList = []
    for i, j in weightArr:
        newParticle = Particle(i.x, i.y, i.h)
        weightList.append(j / counter2)
        particleList.append(newParticle)

    newParticleList = []
    if particleList != []:
        newParticleList = numpy.random.choice(particleList, size=len(particleList), replace=True, p=weightList)

    measured_particles = getMeasuredParticles(Particle.create_random(counter, grid)[:], newParticleList)

    return measured_particles

def findNearestMarker(measuredMarker, visibleMarkers):
    measuredMarkerX, measuredMarkerY, _ = add_marker_measurement_noise(measuredMarker, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)
    nearestMarker = visibleMarkers[0]
    nearestDistance = grid_distance(measuredMarkerX, measuredMarkerY, nearestMarker[0], nearestMarker[1])
    for visibleMarker in visibleMarkers:
        visibleMarkerX, visibleMarkerY, _ = visibleMarker[0], visibleMarker[1], visibleMarker[2]
        dist = grid_distance(measuredMarkerX, measuredMarkerY, visibleMarkerX, visibleMarkerY)
        if dist < nearestDistance:
            nearestMarker = visibleMarker
            nearestDistance = dist
    
    return nearestMarker

def getProbability(pairs, mmlLength, vmLength):
    probability = 1
    transConstantMax = 0
    transConstant = 2 * (MARKER_TRANS_SIGMA ** 2)
    rotConstant = 2 * (MARKER_ROT_SIGMA ** 2)
    for p1, p2 in pairs:
        markerDistance = grid_distance(p1[0], p1[1], p2[0], p2[1])
        markerAngle = diff_heading_deg(p1[2], p2[2])
        newTransConstant = (markerDistance ** 2) / transConstant
        newRotConstant = (markerAngle ** 2) / rotConstant
        transConstantMax = max(transConstantMax, newTransConstant)
        probability = probability * numpy.exp(-newTransConstant - newRotConstant)

    rotConstantMax = (45 ** 2) / (2 * (MARKER_ROT_SIGMA ** 2))

    difference = int(math.fabs(mmlLength - vmLength))
    for _ in range(difference):
        probability = probability * numpy.exp(-transConstantMax - rotConstantMax)
    
    return probability

def getMeasuredParticles(measured_particles, newParticleList):
    for particle in newParticleList:
        particleX = add_gaussian_noise(particle.x, ODOM_TRANS_SIGMA)
        particleY = add_gaussian_noise(particle.y, ODOM_TRANS_SIGMA)
        particleH = add_gaussian_noise(particle.h, ODOM_HEAD_SIGMA)
        newParticle = Particle(particleX, particleY, particleH)
        measured_particles.append(newParticle)

    return measured_particles