cmake_minimum_required(VERSION 2.8)
project(sfm-cpp)
set(CMAKE_CXX_STANDARD 14)
# opencv dependencies
# set(OpenCV_DIR /usr/share/OpenCV)
find_package(CUDA REQUIRED)

find_package(OpenCV COMPONENTS core highgui cuda)
include_directories(( ${OpenCV_INCLUDE_DIRS}))
add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

#adding googletest

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# LIST(APPEND CMAKE_MODULE_PATH /home/kartik/g2o/cmake_modules)
set(G2O_DIR /home/kartik/Downloads/softwares/g2o/cmake_modules)
set(G2O_ROOT /home/kartik/Downloads/softwares/g2o)


find_package(G2O REQUIRED)
find_package(Iconv REQUIRED)
find_package(Boost REQUIRED COMPONENTS)
#INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )
find_package(OpenGL REQUIRED)
message(STATUS "GL included at ${OPENGL_INCLUDE_DIR}")
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
message(STATUS "GLEW included at ${GLEW_INCLUDE_DIR}")
find_package(jsoncpp REQUIRED)
find_package(Boost 1.3 COMPONENTS filesystem) 
find_package(Eigen3 3.3 REQUIRED)
find_package(Sophus REQUIRED)

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
# g2o   
include_directories(${G2O_INCLUDE_DIR})
add_definitions("-D SOPHUS_USE_BASIC_LOGGING")
if (TARGET Eigen3::Eigen)
  set(G2O_EIGEN3_EIGEN_TARGET Eigen3::Eigen)
else()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif ()

include_directories(${Boost_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS}) 
find_package (glog 0.6.0 REQUIRED)
FIND_LIBRARY(OPENGL_LIBRARY             
NAMES GL 
PATHS /usr/local /usr 
PATH_SUFFIXES lib)

IF (G2O_INCLUDE_DIR)
    MESSAGE(STATUS "Found g2o headers in: ${G2O_INCLUDE_DIR}")
  ENDIF (G2O_INCLUDE_DIR)

  FIND_LIBRARY(G2O_CORE_LIB             
    NAMES g2o_core 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_STUFF_LIB            
    NAMES g2o_stuff 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_TYPES_SLAM3D_LIB     
    NAMES g2o_types_slam3d 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_SOLVER_CHOLMOD_LIB   
    NAMES g2o_solver_cholmod 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_SOLVER_PCG_LIB       
    NAMES g2o_solver_pcg 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_SOLVER_CSPARSE_LIB   
    NAMES g2o_solver_csparse 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_INCREMENTAL_LIB      
    NAMES g2o_incremental 
    PATHS /usr/local /usr 
    PATH_SUFFIXES lib)
  FIND_LIBRARY(G2O_CSPARSE_EXTENSION_LIB
    NAMES g2o_csparse_extension
    PATHS /usr/local /usr
    PATH_SUFFIXES lib)

  SET(G2O_LIBRARIES ${G2O_CSPARSE_EXTENSION_LIB}
                    ${G2O_CORE_LIB}           
                    ${G2O_STUFF_LIB}          
                    ${G2O_TYPES_SLAM3D_LIB}   
                    ${G2O_SOLVER_CHOLMOD_LIB} 
                    ${G2O_SOLVER_PCG_LIB}     
                    ${G2O_SOLVER_CSPARSE_LIB} 
                    ${G2O_INCREMENTAL_LIB}                        
                    )
 
  IF(G2O_LIBRARIES AND G2O_INCLUDE_DIR)
    SET(G2O_FOUND "YES")
    IF(NOT G2O_FIND_QUIETLY)
      MESSAGE(STATUS "Found libg2o: ${G2O_LIBRARIES}")
    ENDIF(NOT G2O_FIND_QUIETLY)
  ELSE(G2O_LIBRARIES AND G2O_INCLUDE_DIR)
    IF(NOT G2O_LIBRARIES)
      IF(G2O_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find libg2o!")
      ENDIF(G2O_FIND_REQUIRED)
    ENDIF(NOT G2O_LIBRARIES)

    IF(NOT G2O_INCLUDE_DIR)
      IF(G2O_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find g2o include directory!")
      ENDIF(G2O_FIND_REQUIRED)
    ENDIF(NOT G2O_INCLUDE_DIR)
  ENDIF(G2O_LIBRARIES AND G2O_INCLUDE_DIR)


add_executable(
  yavio2 include/VO.cc src/Utils.cc src/Kitti.cc src/3DHandler.cc src/Viz.cc src/Map.cc include/Frame.hpp include/MapPoint.hpp include/Features.hpp include/OptFlow.hpp
)
target_link_libraries(
    yavio2 PRIVATE Iconv::Iconv jsoncpp_lib opencv_calib3d ${Boost_LIBRARIES} ${Pangolin_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBS}  ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIB} ${G2O_STUFF_LIBRARY} ${G2O_LIBRARIES}
    glog::glog 
    nvToolsExt
    opencv_cudafeatures2d
    opencv_cudaoptflow
  )


  add_executable(
    soExample src/soExample.cc
  )
  target_link_libraries(
      soExample PRIVATE Iconv::Iconv Sophus::Sophus)




# add_executable(
#   temp src/temp.cc src/Utils.cc
# )

# target_link_libraries(
#   temp PRIVATE Iconv::Iconv ${OpenCV_LIBS} ${Boost_LIBRARIES} 
#     glog::glog)



# add_executable(
#   dataHandler src/Kitti.cc src/Utils.cc
# )

# target_link_libraries(
#   dataHandler PRIVATE Iconv::Iconv jsoncpp_lib ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Pangolin_LIBRARIES} 
#     glog::glog)
# Link CHOLMOD if available, CSparse otherwise

enable_testing()

add_executable(
  DataHandlerTest tests/DataHandlerTest.cc src/Kitti.cc src/Utils.cc
)
target_link_libraries(
  DataHandlerTest
  GTest::gtest_main jsoncpp_lib ${Boost_LIBRARIES} ${OpenCV_LIBS} glog::glog
)

add_executable(
  FrameTest tests/FrameTest.cc src/Kitti.cc src/Utils.cc include/Frame.hpp
)
target_link_libraries(
  FrameTest
  GTest::gtest_main jsoncpp_lib ${Boost_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${OPENGL_LIBRARIES}  glog::glog
)



add_executable(
  FeaturesTest tests/FeaturesTest.cc src/Kitti.cc src/Utils.cc  include/Features.hpp)
target_link_libraries(
  FeaturesTest
  GTest::gtest_main jsoncpp_lib ${Boost_LIBRARIES} ${OpenCV_LIBS} glog::glog
  nvToolsExt
  opencv_cudafeatures2d

)



include(GoogleTest)
