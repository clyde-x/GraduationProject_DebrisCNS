# Overview

This is my graduation project for my undergraduate degree in SA BUAA. The main goal of this project is to build a system that can realize automatic navigation for LEO space craft (in this project: China Space Station), using space debris as the beacon. The project can be devide into several parts: 1 simulating the star sensor, which photos the pictures of space debris and star background, with the help of STK; 2 processing the images to get the position of the space debris, and the main point is to process the star background to get the targets/debris we need; 3 when 2 or more space debris are visible, we use optical vectors to calculate the position of the space craft and use stars to determin the attitude; 4 when only 1 space debris is visible, we Kalman filter to estimate the position of the space craft with IMU information.

# Code Structure

# Usage 

## star ap simulation

- Assume you have gloned the repositiry and installed the required packages(python, matlab, stk)
- download the TLE data of LEO objects from [space-track](https://www.space-track.org/documentation#/faq). Login-Recent ELSETs, choose the category you want, and download the TLE data.(for me, I choose LEO objects, and copy it to my local file)
- In **STK11.6**, I find that it's not necessary to interact with MATLAB. Creat a new scenario, import the TLE data to the scenario, and import stars from Hipparcos catalog. Creat a LEO satellite with sensor, and set the relative properties. 
- Now it's time to export data due to STK doesn't support export sensor images. You can also use matlab or python to export the data we need. We set the simulation time to 10 minutes and the step time to 0.1 secend. For sensor, we need its position, velocity, and quaternions. For stars, we need its position and magnitude. For debris, we need its position. All the date are set in ICRF coordinate system. 
- For debris's magnitude, we can simulate it by the normal distribution and the distance between the debris and the sensor.
- And we are going to generate the images of the sensor. Run `generate_star_map.py`. Remember to change the path and relative parameters. Besides images, we can get a csv file `loga_data.csv`, which plays a important role in the following steps.

## image processing and debris classification/matching

- Run `star_mate.ipynb` to process the star map and get the uv-coordinates of the image-points of stars and debris. This file also use multiple frame trajectory based debris classification method to classify debris and stars. The output csv: `star_data.csv` and `debris_data.csv` contains the uv-coordinates of every image-point of stars and debris. The python code also provides a way to compare the results of centroid extraction method: opencv's `cv2.connectedComponentsWithStats` and Gaussion 2d fitting method.
- Here may be a star identification algorithm, but I haven't implemented it yet. 
- Run `debris_matchin.ipynb` to match the deris in images and our catalog. The output csv: `visible_debris.csv` contains every debris' uv-coordinates and its ICRF position. 
- `star_matchin.ipynb` may be used to match the stars in images and our catalog. But it doesn't work well now. So we skip this step.
- There may be some problems: 1 centroid extraction method cannot extract the debris' centroid precisely, and Gaussion method cost too much time (thanks to the matrix inv), 2 debris classification and matching 差强人意，3 star matching is not implemented yet. So to address the problem above, **I cheated**, adding noise to the uv-coordinate in `log_data.csv`. All the following steps are based on this file.

## navigtion
In `my_navigation.py` and `cns.py`, we defined `class Navigation`, `class CNS` and some reltative functions. 

`navigation.ipynb` implements the navigation process.

- Block-TORCH uses PyTorch to optimize the position of SC. Relative math model is established in my 中期报告. It seems to be 'superior', but requires a lot of debris to be visible, more time to optimize, more computation resources. And result in a 'not that bad' potioning result. So it does not appear in the final report.
- Block-CNS completes CNS positioning and attitude determin. And the following parts are plot and result analysis. However, quaternions q[3] changed its sign, which results in a problem that the attitude determination's result not that good at the nearby time. This is a big problem.

Also when using 2 star sensor, the relevant code is in folder `double_sensor`. `main.py` is the main file, you can run it directly, and you can comment part of lines to run the code you want. 

- Due to the complex and confused coordinate system, I use `Csb` and `deltaq` to transform the coordinate system to determine the attitude, which seems a little bit complicated. But it works well. As how I determine the above parameters, 这是我凑出来的

INS and integrated navigation is completed in `integrate.ipynb`. 

- [UKF](https://github.com/balghane/pyUKF). However, there are some prolems. 1 it seems cannot handle non-linear observation model (or I didn't understand it well) 2 nabla--accelerator bias and epsilon--gyroscope bias are not estimated well. 3 the UKF parameters are not well tuned, which results in a bad performance.

# copyright and open source
This project is open source, and you can use it freely. However, please note that the code is not well organized and may contain some bugs. If you have any questions or suggestions, please feel free to contact me.

Relative data is not provided due to the size of the data is too large. You can generate the data by yourself according to the above steps.

[Github](https://github.com/clyde-x/GraduationProject_DebrisCNS)
[E-mail](clydeisakid@outlook.com)
You can also access the LaTex version of the thesis via [Github](https://github.com/clyde-x/bachelor_BUAA)