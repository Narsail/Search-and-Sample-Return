import numpy as np
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh_between(img, rgb_thresh=((160, 255), (160, 255), (160, 255))):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (rgb_thresh[0][0] <= img[:, :, 0]) \
                   & (img[:, :, 0] < rgb_thresh[0][1]) \
                   & (rgb_thresh[1][0] <= img[:, :, 1]) \
                   & (img[:, :, 1] < rgb_thresh[1][1]) \
                   & (rgb_thresh[2][0] <= img[:, :, 2]) \
                   & (img[:, :, 2] < rgb_thresh[2][1])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def image_segmentation(img):
    ground = color_thresh_between(img)

    obstacles = color_thresh_between(img, ((0, 160), (0, 160), (0, 160)))

    nuggets = color_thresh_between(img, ((100, 255), (100, 255), (20, 30)))

    return ground, obstacles, nuggets


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()

    image = Rover.img

    # 1) Define source and destination points for perspective transform

    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])

    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    ground, obstacle, nugget = image_segmentation(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = 255 * obstacle
    Rover.vision_image[:, :, 1] = 255 * nugget
    Rover.vision_image[:, :, 2] = 255 * ground

    # 5) Convert map image pixel values to rover-centric coords
    groundPix = rover_coords(ground)
    obstaclePix = rover_coords(obstacle)
    nuggetPix = rover_coords(nugget)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    rover_xpos, rover_ypos, rover_yaw = Rover.pos[0], Rover.pos[1], Rover.yaw

    groundWorld = pix_to_world(
        groundPix[0], groundPix[1], rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], scale
    )
    obstacleWorld = pix_to_world(
        obstaclePix[0], obstaclePix[1], rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], scale
    )
    nuggetWorld = pix_to_world(
        nuggetPix[0], nuggetPix[1], rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], scale
    )

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obstacleWorld[1], obstacleWorld[0], 0] += 1
    Rover.worldmap[nuggetWorld[1], nuggetWorld[0], 1] += 1
    Rover.worldmap[groundWorld[1], groundWorld[0], 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(groundPix[0], groundPix[1])

    Rover.nav_dists = dist
    Rover.nav_angles = angles

    return Rover
