## Project: Search and Sample Return

---

[image0]: ./calibration_images/example_rock1.jpg
[image1]: ./misc/area.jpg  

### Writeup

#### Provide a Writeup that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

In this section i will give you an overview about how i tackled the project:

Obviously i started working on the perception of the rover step by step through the udacity course content and with
the help of the jupyter notebook. How the content of the notebook was solved will be handled in the next rubric.

In the next step i copied the `process_image()` content to the `perception.py` and started simple trial and error 
runs with the rover in autonomous mode. The chosen values for the perception were good enough to reach the 
expected 40% mapping with 60% fidelity in several cases. But overall the algorithm was not stable enough.

##### The State Machine

But before improving the fidelity i split the rovers state machine into two hierarchies:

1. we have a `.mode` which has the following states:

* LOCATE_NUGGET
* GO_TO_NUGGET
* PICK_UP_NUGGET
* GO_TO_START
* FOUND_START

The rover will go through the first 3 states in a circle until it found all golden nuggets. Then it will switch into the
`GO_TO_START` until it finds the starting point where it will stop.

2. we have the `.drive_mode` which has the following states:

* FORWARD
* STOP
* STUCK (was added later)

Those are used to drive the rover through the terrain in the different modes.

##### Hunting the nugget

In the next step i focused (in the assumption that my mapping is accurate enough, which turned out to be incorrect later) on finding and navigating to the golden nuggets
to pick them up. A more detailed explanation about this step will follow in a later rubric.

##### Improving the fidelity

To improve the fidelity and the driving behavior of the robot i researched a bit for different approaches. I found the following medium post of a fellow udacity student which helped me to improve the behavior of my robot a lot.
Medium Post: https://medium.com/@fernandojaruchenunes/udacity-robond-project-1-search-and-sample-return-2d8165a53a78

Through that post i extended my perception module with masking for the obstacle section and with limiting the view distance to only use the area til 8 meters away. Those additions improved the fidelity by a high factor. 
It also helped with the decision module: I introduced his idea of the stuck driving mode as well as the idea of 'hugging' the left wall (though i used different values) to stabilize the rover.

##### Outcome

My rover is able to collect all nuggets and drive back to the starting point in most of the case. It only might take a while because i did not optimize his acceleration/braking behaviour. 

### Notebook Analysis

The main difference to the standard course material and my jupyter notebook is the introduction of the `image_segmentation()` method
to return a triple tuple with the navigable ground (called `ground`), the obstacles and the nugget area. I extended the underlying `color_thresh` with between values which means that i
can set the range in which the values of the chosen channel (red, green or blue) will be considered. 

```
def color_thresh_between(img, rgb_thresh=((160, 255), (160, 255), (160, 255))):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (rgb_thresh[0][0] <= img[:,:,0]) \
                & (img[:,:,0] < rgb_thresh[0][1]) \
                & (rgb_thresh[1][0] <= img[:,:,1]) \
                & (img[:,:,1] < rgb_thresh[1][1]) \
                & (rgb_thresh[2][0] <= img[:,:,2]) \
                & (img[:,:,2] < rgb_thresh[2][1])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
``` 

With minor tweaks it was possible to extract all three maps (ground, obstacle and nugget) with this function through the `image_segmentation()` method:

```
def image_segmentation(img):
    
    ground = color_thresh_between(img)
    
    obstacles = color_thresh_between(img, ((0, 160), (0, 160), (0, 160)))
    
    nuggets = color_thresh_between(img, ((100, 255), (100, 255), (20, 30)))
    
    return ground, obstacles, nuggets
```

At the beginning i mentioned that i masked the obstacle map with the help of Fernando Nunes with

```
obstacle_mask = np.ones_like(rock_img)
obstacle_mask[:, :] = 255
obstacle_mask = perspect_transform(obstacle_mask, source, destination)

obstacle = np.absolute(obstacle * obstacle_mask[:, :, 0])
```

Which just means that we take a normal vision image of the robot (should be the current one), create a new image based on the same shape but with only one color channel. 
We set that channel to white, then transform it and obtain the top down view area of the robot. We now use this to mask the area where an obstacle can be.

In the following image the result of the image segmentation is shown 

![Ground, Obstacle and Nugget][image1]

based on the following input:

![Base nugget image][image0]

In the right most image of the image segmentation you can see the pixel of the golden nugget.

#### Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

In the `process_image()` i merely added all the previous introduced perception steps and generated the output video in /output/test_mapping.mp4

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

##### Perception Step

The perception step differs in four places compared to the udacity standard behaviour. 

1. I introduced the image segmentation to have the mapped areas of the navigable ground, the obstacles and the nuggets
2. I masked the obstacle view to exclude areas beyond the rovers field of vision
3. I constrained the viewed area to 8 meters to exclude areas which are to far to be precisely mapped

    For that i used the `impose_range` method from Fernando Nunes to constrain the sight to 8 meters.
    
    ```
    def impose_range(xpix, ypix, range=80):
    dist = np.sqrt(xpix**2 + ypix**2)
    return xpix[dist < range], ypix[dist < range]
    ```

4. I stored the distance and the angles for a nugget in the field of vision to the rovers state.

##### Decision Step

The rover has two different mode types:

1. The Driving Modes

Those modes handle the driving of the rover within the method
`def drive(rover, nav_angles, max_vel, stop_forward, go_forward, offset=0):`
which is the standard method provided by udacity. It has the forward and stop behavior extended with the new stuck mode. 
The stuck mode will be activated when the rover has not moved for longer than 4 seconds. Then it will turn on the spot for 1 seconds and jump back into forward mode to try to move again.
Further more i added the possibility of using a steering offset in case we want to tend to a specific side.

2. The Rover Modes

These modes have been introduced to differentiate between the driving modes and the actual mission step the rover is in.
The rover starts in the `LOCATE_NUGGET` and drives around the map, hugging the left wall until it recognizes a nugget. It uses the angles of the navigable terrain to drive around.

It will then jump into the `GO_TO_NUGGET` mode when we found a nugget. In this mode the rover will use the nugget angles to navigate towards the nugget. When the `.near_sample` property is true the rover will jump into the

`PICK_UP_NUGGET` mode where we just pickup the nugget. Afterwards we decide dependent on whether the rover still needs to collect some nugget in which mode to go next.
If there are still nuggets to collect it will jump back into the `LOCATE_NUGGET` mode. If we are done collecting nuggets, the next mode will be the 

`GO_TO_START` mode. In this mode the rover behaves like in the `LOCATE_NUGGET` mode and just drives around while hugging the left wall until it is close enough to the starting point (<20 meters).
If so it will remove the steering offset to drive more towards the center of the navigable area. 

If it is close to the starting point < 5 meter, it will go into the `FOUND_START` mode and stop the vehicle.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

Simulator Settings:

* Screen Resolution: 1440x900
* Graphics Quality: Good
* Windowed: true

As mentioned in the beginning of the write up i took the following approach:

1. Complete the feedback cycle with a simple but working perception and decision step
2. Extend the state machine to have a baseline for the advanced challenge
3. Add the ability to located and drive towards golden nuggets to pick them up
4. In this state my rover had the following problems:
    * will stuck if driving into an obstacle with some navigable terrain ahead
    * will sometimes drive around in circles in big areas
5. I worked on the fidelity through masking the obstacles area and constraining the vision field to 8 meters
6. Add the stuck mode to have an answer to the situation where the rover is stuck
7. Add a steering offset so that the rover will tend to drive towards the left wall
8. Add the condition to remove the offset if the rover comes close to its starting point to drive through the center of an area (where the starting point potentially is)

The work results in a rover which navigates around the map hugging the left wall with an average fidelity of 70% with a mapped area > 90%. In most of the cases it will collect all nuggets (with some rare exceptions) and then will continue his journey around the map until it comes close to the starting point. If it is within a range of 5 meters of the starting point it will completely stop.

Further improvements might happen in the following areas:

* acceleration, braking and overall speed
* increasing the fidelity with excluding mappings mapped with yaw / pitch / roll values unequal to zero
* let the rover navigate towards the starting point if it found all nuggets instead of continuing his wall hugging behavior until it randomly finds the starting point
* improve the go to nugget behavior as well as the recognition of the nugget  
