cmake_minimum_required(VERSION 3.13)

project(ov_pointpillars_extensions)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenVINO
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Source files
set(SOURCES
    iou3d_utils.cpp
    voxelization_op.cpp
    postprocessing_op.cpp
    ov_pointpillars_ext.cpp
)

set(HEADERS
    iou3d_utils.hpp
    voxelization_op.hpp
    postprocessing_op.hpp
)

# Build shared library
add_library(ov_pointpillars_extensions SHARED ${SOURCES} ${HEADERS})

target_link_libraries(ov_pointpillars_extensions PRIVATE openvino::runtime)

# Set output directory
set_target_properties(ov_pointpillars_extensions PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Install
install(TARGETS ov_pointpillars_extensions
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
