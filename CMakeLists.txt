cmake_minimum_required(VERSION 3.5)
project(approximating_images)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES src/3rdparty/stb_image.h
        src/3rdparty/stb_image_write.h
        src/colour.cc
        src/colour.h
        src/triangle.cc
        src/triangle.h
        src/image.cc
        src/image.h
        src/triangulator.cc
        src/triangulator.h)

add_executable(approximating_images
        src/main.cc
        ${SOURCE_FILES})
add_executable(test
        src/test.cc
        ${SOURCE_FILES})

if (${NO_SFML})
else()
    include(FetchContent)
    FetchContent_Declare(SFML
            GIT_REPOSITORY https://github.com/SFML/SFML.git
            GIT_TAG 2.6.x
            GIT_SHALLOW ON
            EXCLUDE_FROM_ALL
            SYSTEM)
    FetchContent_MakeAvailable(SFML)

    target_link_libraries(approximating_images PRIVATE sfml-graphics)
    target_link_libraries(test PRIVATE sfml-graphics)

    # add SFML_SUPPORTED
    target_compile_definitions(approximating_images PRIVATE SFML_SUPPORTED)
    target_compile_definitions(test PRIVATE SFML_SUPPORTED)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOS TRUE)
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /Wall /O3")
elseif(MACOS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -ffast-math -mcpu=apple-m1 -openmp -O3 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -g")
else() # Linux
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -fopenmp -O3 -march=native -g")
endif()
