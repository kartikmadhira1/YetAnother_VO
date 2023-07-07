# YA_VO_2

Iterative addition on top of the YA_VO with Stereo setup.


#### Primary goal of the project:

1. Efficient data-pipelineing and data-processing.
2. Focus on optimization with more emphasis on explainable optimization code.
3. Efficient initialization of stereo setup.
4. Local BA and full BA fully utilized.
5. Use OpenCV codes as little as possible.
6. Lastly, be able to validate the odometry with an error rate. Math to evaluate trajectories also essential.

#### Secondary goals of the project

1. Replace image processing with cuda accelerated code.
2. Replace keypoint detection, matching and Optical flow with deep learning model.
3. If everything goes well, generate a dense 3D map of the environment using NeRF.




##### Compiilation steps (Linux):
1. Set g2o path in CMakeLists.txt:
```set(G2O_DIR /Users/kartikmadhira/g2o/cmake_modules)```
2. If g2o could not be found copy FindG2O.cmake to `/usr/share/cmake-3.16/Modules/`
