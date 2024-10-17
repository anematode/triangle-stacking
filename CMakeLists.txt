cmake_minimum_required(VERSION 3.0)
project(approximating_images)

set(CMAKE_CXX_STANDARD 20)

add_executable(approximating_images main.cpp src/3rdparty/stb_image.h src/3rdparty/stb_image_write.h)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /Wall /O3")
elseif(MACOSX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -ffast-math -mcpu=apple-m1 -openmp -O3 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -framework OpenCL -g")
else() # Linux
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -fopenmp -O3 -march=native -g")
endif()
