from enum import Enum

import numpy as np
# from drive_rover import RoverState


class Mode(Enum):
    LOCATE_NUGGET = 0
    GO_TO_NUGGET = 1
    PICK_UP_NUGGET = 2
    GO_TO_START = 3
    FOUND_START = 4


class DriveMode(Enum):
    FORWARD = 0
    STOP = 1
    STUCK = 2


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
    print('## Current Drive Mode: {}'.format(rover.drive_mode))
    print()
    print('## Distance from Start: {}'.format(np.linalg.norm(np.asarray(rover.pos) - np.asarray(rover.start_pos))))
    print('## Distance from visible nugget: {}'.format(np.mean(rover.nugget_dist)))

    if rover.mode == Mode.LOCATE_NUGGET:

        locate_nugget(rover)

    elif rover.mode == Mode.GO_TO_NUGGET:

        go_to_nugget(rover)

    elif rover.mode == Mode.PICK_UP_NUGGET:

        pick_up_nugget(rover)

    elif rover.mode == Mode.GO_TO_START:

        go_to_start(rover)

    elif rover.mode == Mode.FOUND_START:
        rover.steer = 0
        rover.throttle = 0
        rover.brake = rover.brake_set

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

        if rover.samples_to_find > rover.samples_collected:
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

    # Use an offset to tend to a specific side as described in
    # https://medium.com/@fernandojaruchenunes/udacity-robond-project-1-search-and-sample-return-2d8165a53a78
    offset = 0

    if rover.total_time > 20:
        offset = 0.5 * np.std(rover.nav_angles)

    nugget_dist_threshold = 60
    nugget_threshold = 20

    if rover.near_sample or len(rover.nugget_angles) > nugget_threshold and np.mean(rover.nugget_dist) < nugget_dist_threshold:
        rover.mode = Mode.GO_TO_NUGGET
    else:
        drive(rover, rover.nav_angles, rover.max_vel, rover.stop_forward, rover.go_forward, offset)


def go_to_start(rover):

    if np.linalg.norm(np.asarray(rover.pos) - np.asarray(rover.start_pos)) < 5:
        rover.mode = Mode.FOUND_START
    elif np.linalg.norm(np.asarray(rover.pos) - np.asarray(rover.start_pos)) > 20:
        drive(rover, rover.nav_angles, rover.max_vel, rover.stop_forward, rover.go_forward, 0.5 * np.std(rover.nav_angles))
    else:
        drive(rover, rover.nav_angles, rover.max_vel, rover.stop_forward, rover.go_forward)


def drive(rover, nav_angles, max_vel, stop_forward, go_forward, offset=0):
    if nav_angles is not None:
        # Check for Rover.mode status
        if rover.drive_mode == DriveMode.FORWARD:
            # Check the extent of navigable terrain
            if len(nav_angles) >= stop_forward:
                # Use the stuck mode as described in
                # https://medium.com/@fernandojaruchenunes/udacity-robond-project-1-search-and-sample-return-2d8165a53a78
                if rover.vel <= 0.1 and rover.total_time - rover.stuck_time > 4:
                    # Set mode to "stuck" and hit the brakes!
                    rover.throttle = 0
                    # Set brake to stored brake value
                    rover.brake = rover.brake_set
                    rover.steer = 0
                    rover.drive_mode = DriveMode.STUCK
                    rover.stuck_time = rover.total_time
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if rover.vel < max_vel:
                    # Set throttle value to throttle setting
                    rover.throttle = rover.throttle_set
                else:  # Else coast
                    rover.throttle = 0
                rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                rover.steer = np.clip(np.mean((nav_angles + offset) * 180 / np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(nav_angles) < stop_forward:
                # Set mode to "stop" and hit the brakes!
                rover.throttle = 0
                # Set brake to stored brake value
                rover.brake = rover.brake_set
                rover.steer = 0
                rover.drive_mode = DriveMode.STOP

        # Use the stuck mode as described in
        # https://medium.com/@fernandojaruchenunes/udacity-robond-project-1-search-and-sample-return-2d8165a53a78
        elif rover.drive_mode == DriveMode.STUCK:
            # if 1 sec passed go back to previous mode
            if rover.total_time - rover.stuck_time > 1:
                rover.drive_mode = DriveMode.FORWARD
            # Now we're stopped and we have vision data to see if there's a path forward
            else:
                rover.throttle = 0
                # Release the brake to allow turning
                rover.brake = 0
                # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                rover.steer = -15 if offset > 0 else 15 if rover.steer > 0 else -15

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
                    rover.steer = -15 if offset > 0 else 15 if rover.steer > 0 else -15
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(nav_angles) >= go_forward:
                    # Set throttle back to stored value
                    rover.throttle = rover.throttle_set
                    # Release the brake
                    rover.brake = 0
                    # Set steer to mean angle
                    rover.steer = np.clip(np.mean((nav_angles + offset) * 180 / np.pi), -15, 15)
                    rover.drive_mode = DriveMode.FORWARD
    else:
        rover.throttle = rover.throttle_set
        rover.steer = 0
        rover.brake = 0

