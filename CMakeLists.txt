cmake_minimum_required(VERSION 3.24)

project(face_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)

if(True)
    find_package(OpenMP)
    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(${OpenCV_DIR}/include
        /ncnn/build/install/include/ncnn)
link_directories(${OpenCV_DIR}/lib/
        /ncnn/build/install/lib)

set(OpenCV_LIBS opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio)

add_executable(face_detection main.cpp
        FaceDetector.cpp
        FaceDetector.h
)

target_link_libraries(face_detection ncnn ${OpenCV_LIBS})
