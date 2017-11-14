from grid import *
from particle import Particle
from utils import *
from setting import *

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
    oldPos = odom[0]
    newPos = odom[1]
    newParticles = []
    deg = diff_heading_deg(oldPos[2], newPos[1])
    dx = newPos[0] - oldPos[0]
    dy = newPos[1] - oldPos[1]
    #print(deg)
    for p in particles:
        #print(particles)
        x, y = rotate_point(dx,dy,p.h)
        p.x +=x
        p.y+=y
        p.x = add_gaussian_noise(p.x, ODOM_TRANS_SIGMA)
        p.y= add_gaussian_noise(p.y, ODOM_TRANS_SIGMA)

        p.h += deg
        p.h = add_gaussian_noise(p.h, ODOM_TRANS_SIGMA)

        newParticles.append(p)



    #print("odom")
    #print(odom)

    #print("particles")
    #for x in particles:
        #print(x)

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
    return measured_particles
