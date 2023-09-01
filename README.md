# YA_VO_2

Iterative addition on top of the YA_VO with Stereo setup.


#### Primary goal of the project:

- [x] Efficient data-pipelineing and data-processing.
- [x] Focus on optimization with more emphasis on explainable optimization code.
- [x] Efficient initialization of stereo setup.
- [ ] Local BA and full BA fully utilized.
- [ ] Get Loop closure working.
- [ ] Use OpenCV codes as little as possible.
- [ ] Lastly, be able to validate the odometry with an error rate. Math to evaluate trajectories also essential.

#### Secondary goals of the project

- [x] Replace image processing with cuda accelerated code.
- [ ] Replace keypoint detection, matching and Optical flow with deep learning model.
- [ ] If everything goes well, generate a dense 3D map of the environment using NeRF.




##### Compiilation steps (Linux):

0. OpenCV and OpenCV contrib with flags. [To be updated]
1. Set g2o path in CMakeLists.txt:
```set(G2O_DIR /Users/kartikmadhira/g2o/cmake_modules)```
2. If g2o could not be found copy FindG2O.cmake to 
`/usr/share/cmake-3.16/Modules/`
3. Install jsoncpp :
```sudo apt-get install libjsoncpp-dev```
4. Install Sophus from :
```git clone https://github.com/strasdat/Sophus.git```
5. Install Pangolin using [these](https://github.com/stevenlovegrove/Pangolin) instructions.
6. Install google test from [these](https://github.com/google/googletest/blob/main/googletest/README.md) instructions.
7. Install glog from [these](https://github.com/google/glog#id5) instructions.
8. Install Eigen using [these](https://stackoverflow.com/a/72150616) instructions.
