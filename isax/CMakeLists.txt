cmake_minimum_required(VERSION 3.16)
project(isax)


set(CMAKE_CXX_STANDARD 14)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -pthread -fopenmp")


set(CMAKE_BUILD_TYPE Release)


find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    add_compile_options(-fopenmp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


link_directories(/usr/lib/x86_64-linux-gnu)


add_executable(index index.cpp random_data.cpp)
add_executable(search search.cpp random_data.cpp)


target_link_libraries(search
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
    gsl
    ${OpenMP_CXX_FLAGS}
)

target_link_libraries(index
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
    gsl
    ${OpenMP_CXX_FLAGS}
)
