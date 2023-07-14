# Homography-based Method to Compute Green Screen Coordinates

## What does this code do?
The Python program [gs_homography.py](gs_homography.py) computes the coordinates of the 4 corners of the green screen on each video frame in the videos of the [multi-object tracking scenario in Armory](https://github.com/twosixlabs/armory/blob/master/armory/data/adversarial/carla_mot_dev.py), **assuming accurate green screen coordinates are present on the first frame of the video**. 

## Why write this code?
We develop this program because we have observed that the existing green screen coordinates are not stationary in the virtual world of some video frames, causing the green screen (or patch that would be placed on the green screen) to "move."

## What does the program do the video frames and green screen coordinates?
The program copies the original videos from a user-provided path (i.e., the "old" dataset), to a new location at a path provided the user (we call this the "new" dataset), and computes and stores the computed green screen coordinates in the new dataset as calculated according to the parameters provided in [homography.json](homography.json). 

The program also visualizes the green screen coordinates of the old data as red dots and new data as yellow dots. The video frames with the visualized coordinates are stored in the `rgb_w_coords` subfolder within each video folder at the new dataset's path. Below we show an example of how the coordinates may be corrected by the program ("before" in red vs "after" in yellow).

![zoomed in result images](image_comparison.jpg)

## How to run the program?

```bash
python3 gs_homography.py {json file}
```





