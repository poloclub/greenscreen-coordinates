**Green Screen Coordinate Modification**

The python file will copy the older dataset from the given path, to the newer path and will modify the green screen coordinates in the new dataset as calculated according to the parameters provided in the json

The file will also generate a visualization that will show the green screen coordinates of the old and new data. This visulaization will be stored in the same path as the new generated dataset

**Run the file**

python3 gs_homography.py {json file}

**Results**

The green screen coordinates in the "new_data" path will be modified

The older and newer coordinates can be compared by the images in the "rgb_w_coords" folder in each video in the new dataset. By default the old coordinates are colored red , and the new coordinates are colored yellow.

![zoomed in result images](image_comparison.jpg)
