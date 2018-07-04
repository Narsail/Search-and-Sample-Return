from enum import Enum

import numpy as np
# from drive_rover import RoverState


class Mode(Enum):
    LOCATE_NUGGET = 0
    GO_TO_NUGGET = 1
    PICK_UP_NUGGET = 2
    GO_TO_START = 3


class DriveMode(Enum):
    FORWARD = 0
    STOP = 1


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(rover):
    '''

    :param rover:
    :type rover: RoverState
    :return:
    '''

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    print()
    print('## Current Mode: {}'.format(rover.mode))
    print()

    if rover.mode == Mode.LOCATE_NUGGET:

        locate_nugget(rover)

    elif rover.mode == Mode.GO_TO_NUGGET:

        go_to_nugget(rover)

    elif rover.mode == Mode.PICK_UP_NUGGET:

        pick_up_nugget(rover)

    elif rover.mode == Mode.GO_TO_START:

        locate_nugget(rover)

    return rover


def pick_up_nugget(rover):
    """

    :param rover:
    :type rover: RoverState
    :return:
    """

    if rover.near_sample and not rover.picking_up:
        rover.send_pickup = True
    elif not rover.near_sample:

        if rover.samples_to_find > 0:
            rover.mode = Mode.LOCATE_NUGGET
        else:
            rover.mode = Mode.GO_TO_START


def go_to_nugget(rover):

    max_vel = rover.max_vel / 4

    if rover.near_sample and rover.vel > 0.2:
        rover.throttle = 0
        rover.brake = rover.brake_set * 10
    elif rover.near_sample and rover.vel < 0.2:
        rover.mode = Mode.PICK_UP_NUGGET
    elif not rover.near_sample:
        drive(rover, rover.nugget_angles, max_vel, 1, 1)


def locate_nugget(rover):

    nugget_threshold = 20

    if rover.near_sample or len(rover.nugget_angles) > nugget_threshold:
        rover.mode = Mode.GO_TO_NUGGET
    else:
        drive(rover, rover.nav_angles, rover.max_vel, rover.stop_forward, rover.go_forward)


def drive(rover, nav_angles, max_vel, stop_forward, go_forward):
    if nav_angles is not None:
        # Check for Rover.mode status
        if rover.drive_mode == DriveMode.FORWARD:
            # Check the extent of navigable terrain
            if len(nav_angles) >= stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if rover.vel < max_vel:
                    # Set throttle value to throttle setting
                    rover.throttle = rover.throttle_set
                else:  # Else coast
                    rover.throttle = 0
                rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                rover.steer = np.clip(np.mean(nav_angles * 180 / np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(nav_angles) < stop_forward:
                # Set mode to "stop" and hit the brakes!
                rover.throttle = 0
                # Set brake to stored brake value
                rover.brake = rover.brake_set
                rover.steer = 0
                rover.drive_mode = DriveMode.STOP

        # If we're already in "stop" mode then make different decisions
        elif rover.drive_mode == DriveMode.STOP:
            # If we're in stop mode but still moving keep braking
            if rover.vel > 0.2:
                rover.throttle = 0
                rover.brake = rover.brake_set
                rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(nav_angles) < go_forward:
                    rover.throttle = 0
                    # Release the brake to allow turning
                    rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    rover.steer = 15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(nav_angles) >= go_forward:
                    # Set throttle back to stored value
                    rover.throttle = rover.throttle_set
                    # Release the brake
                    rover.brake = 0
                    # Set steer to mean angle
                    rover.steer = np.clip(np.mean(nav_angles * 180 / np.pi), -15, 15)
                    rover.drive_mode = DriveMode.FORWARD
    else:
        rover.throttle = rover.throttle_set
        rover.steer = 0
        rover.brake = 0

