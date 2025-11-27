cmake_minimum_required(VERSION 3.15...4.2)

project(
    MyProject VERSION 0.1
    DESCRIPTION "Red Pattern Simulation"
    LANGUAGES C CUDA)

# ------------------------------------------------------------------------------------------------
# General Settings
# ------------------------------------------------------------------------------------------------


if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_C_EXTENSION OFF)
endif()

if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Default to Ampere architecture
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Use RELEASE builds by default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "CHoose the type of Build: Debug Release"
        FORCE)
endif()

# For clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------------------------
# argtable 3
# ------------------------------------------------------------------------------------------------

add_library(argtable3 STATIC third_party/argtable3/argtable3.c)
target_include_directories(argtable3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/argtable3)
target_compile_options(argtable3 PRIVATE -Wall -Wextra)

# ------------------------------------------------------------------------------------------------
# HDF5
# ------------------------------------------------------------------------------------------------

find_package(HDF5 COMPONENTS C HL REQUIRED)
if(TARGET hdf5::hdft)
    set(HDF5_LIB_TARGETS hdf5::hdf5 hdf5::hdf5_hl)
else()
    set(HDF5_LIB_TARGETS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif()

# ------------------------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------------------------

add_executable(red-patterns
    src/main.cu
    src/hdf5_file.c
)

target_include_directories(red-patterns
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/argtable3
        ${HDF5_INCLUDE_DIRS}
)

target_compile_options(red-patterns
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

target_compile_definitions(red-patterns
    PRIVATE
        ${HDF5_DEFINITIONS}
)

target_link_libraries(red-patterns
    PRIVATE
        argtable3
        ${HDF5_LIB_TARGETS}
)

if(HDF5_LIBRARY_DIRS)
    set_target_properties(red-patterns PROPERTIES
    BUILD_RPATH "${HDF5_LIBRARY_DIRS}"
    INSTALL_RPATH "${HDF5_LIBRARY_DIRS}")
endif()
