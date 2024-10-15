cmake_minimum_required(VERSION 3.28)
project(approximating_images)

set(CMAKE_CXX_STANDARD 20)

add_executable(approximating_images main.cpp src/3rdparty/stb_image.h src/3rdparty/stb_image_write.h)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -mcpu=apple-m1 -openmp -O3 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -framework OpenCL")
