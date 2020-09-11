# FaceDetection_CPU_GPU
A visual studio project for Facedetection on CPU and GPU

## Prequisites

1. Install cuda toolkit (v11.0 was used during this project)
2. Install opencv built libraries

[ Windows Installation ] \
https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html#windows-installation

[ Linux Installation ] \
https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation

3. Install opencv source and additional libraires and build them using cmake \
https://github.com/opencv/opencv/ \
https://github.com/opencv/opencv_contrib/ \
https://cmake.org/download/

4. Make sure to add the include directories and install paths after cmake was built in the respective paths in visual studio.

## Main Steps

1. The Project can be downloaded and opened with visual studio.
2. After opening the code, the source files need to be edited.
3. Ensure that only the source file cuda_object_detection/kernel_webcam_gpu_cpu.cu is used.
4. Modify the directory path of opencv folder that was installed in the pre-requisites.
5. Build the project and Run the executable
