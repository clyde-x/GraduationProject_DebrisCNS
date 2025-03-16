# Overview

This is my graduation project for my undergraduate degree in SA BUAA. The main goal of this project is to build a system that can realize automatic navigation for LEO space craft, using space debris as the beacon. The project can be devide into several parts: 1 simulating the star sensor, which photos the pictures of space debris and star background, with the help of STK and MATLAB; 2 processing the images to get the position of the space debris, and the main point is to process the star background to get the targets we need; 3 when 2 or more space debris are visible, we use optical vectors to calculate the orbit/position of the space craft and use stars to determin the attitude; 4 when only 1 space debris is visible, we Kalman filter to estimate the position of the space craft with IMU information.

# Code Structure

# Usage 
- Assume you have gloned the repositiry and installed the required packages(python, matlab, stk)
- download the TLE data of LEO objects from [space-track](https://www.space-track.org/documentation#/faq). Login-Recent ELSETs, choose the category you want, and download the TLE data.(for me, I choose LEO objects, and copy it to my local file)
- In **STK11.6**, I find that it's not necessary to interact with MATLAB. Creat a new scenario, import the TLE data to the scenario, and import stars from Hipparcos catalog. Creat a LEO satellite with sensor, and set the relative properties. 
- Now it's time to export data due to STK doesn't support export sensor images. You can also use matlab or python to export the data we need. We set the simulation time to 10 minutes and the step time to 0.1 secend. For sensor, we need its position, velocity, and quaternions. For stars, we need its position and magnitude. For debris, we need its position. All the date are set in ICRF coordinate system. 
- For debris's magnitude, we can simulate it by the normal distribution and the distance between the debris and the sensor.
- And we are going to generate the images of the sensor. Run `generate_star_map.py`. Remember to change the path and relative parameters.