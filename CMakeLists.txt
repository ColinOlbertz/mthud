cmake_minimum_required(VERSION 3.20)

project(mthud LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options ---------------------------------------------------------------
option(USE_XSENS "Enable Xsens MT SDK integration" ON)

# ---- Dependencies via vcpkg / system --------------------------------------
find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio aruco highgui)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG QUIET)

if(UNIX AND NOT APPLE)
  find_package(OpenGL REQUIRED) # for GL symbols when linking with GLAD/GLFW
endif()

# ---- Sources ---------------------------------------------------------------
set(MTHUD_SOURCES
  src/main.cpp
  src/aruco_tracker.cpp
  src/hud_renderer.cpp
  src/config.cpp
)

if(USE_XSENS)
  list(APPEND MTHUD_SOURCES src/sensor_xsens.cpp)
endif()

add_executable(mthud ${MTHUD_SOURCES})

# String macro used by your code
target_compile_definitions(mthud PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Warnings
if(MSVC)
  target_compile_options(mthud PRIVATE /W4 /permissive-)
else()
  target_compile_options(mthud PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Base include paths
target_include_directories(mthud PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

# Base libraries
target_link_libraries(mthud PRIVATE
  glad::glad
  glfw
  ${OpenCV_LIBS}
  Freetype::Freetype
  nlohmann_json::nlohmann_json
  Threads::Threads
)

if(spdlog_FOUND)
  target_link_libraries(mthud PRIVATE spdlog::spdlog_header_only)
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(mthud PRIVATE OpenGL::GL dl)
endif()

# ---- Xsens MT SDK ----------------------------------------------------------
if(USE_XSENS)
  target_compile_definitions(mthud PRIVATE USE_XSENS=1)

  # Resolve SDK root from cache var or env
  if(NOT XSENS_SDK_ROOT AND DEFINED ENV{XSENS_SDK_ROOT})
    set(XSENS_SDK_ROOT "$ENV{XSENS_SDK_ROOT}")
  endif()
  set(XSENS_SDK_ROOT "${XSENS_SDK_ROOT}" CACHE PATH "Path to Xsens MT SDK root (contains include/ and lib/)")

  if(WIN32)
    # Headers
    find_path(XSENS_INCLUDE_DIR xsensdeviceapi.h
      HINTS "${XSENS_SDK_ROOT}/include"
            "$ENV{ProgramFiles}/Xsens/MT SDK/include"
            "$ENV{ProgramFiles(x86)}/Xsens/MT SDK/include")

    # Libs (names vary by SDK version)
    find_library(XSENS_LIB   NAMES xsensdeviceapi xsensdeviceapi64
      HINTS "${XSENS_SDK_ROOT}/lib"
            "$ENV{ProgramFiles}/Xsens/MT SDK/lib"
            "$ENV{ProgramFiles(x86)}/Xsens/MT SDK/lib")
    find_library(XSTYPES_LIB NAMES xstypes xstypes64
      HINTS "${XSENS_SDK_ROOT}/lib"
            "$ENV{ProgramFiles}/Xsens/MT SDK/lib"
            "$ENV{ProgramFiles(x86)}/Xsens/MT SDK/lib")
  else()
    # Linux default locations
    find_path(XSENS_INCLUDE_DIR xsensdeviceapi.h
      HINTS "${XSENS_SDK_ROOT}/include" "/usr/local/xsens/mt-sdk/include")
    find_library(XSENS_LIB   NAMES xsensdeviceapi
      HINTS "${XSENS_SDK_ROOT}/lib" "/usr/local/xsens/mt-sdk/lib")
    find_library(XSTYPES_LIB NAMES xstypes
      HINTS "${XSENS_SDK_ROOT}/lib" "/usr/local/xsens/mt-sdk/lib")

    # Make runtime find the shared libs
    set_target_properties(mthud PROPERTIES
      BUILD_RPATH   "${XSENS_SDK_ROOT}/lib;$ORIGIN"
      INSTALL_RPATH "${XSENS_SDK_ROOT}/lib;$ORIGIN")

    # Also drop the .so files next to the exe for convenience
    add_custom_command(TARGET mthud POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${XSENS_SDK_ROOT}/lib/libxsensdeviceapi.so" "$<TARGET_FILE_DIR:mthud>/"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${XSENS_SDK_ROOT}/lib/libxstypes.so" "$<TARGET_FILE_DIR:mthud>/")
  endif()

  if(NOT XSENS_INCLUDE_DIR OR NOT XSENS_LIB OR NOT XSTYPES_LIB)
    message(FATAL_ERROR "Xsens SDK not found. Set XSENS_SDK_ROOT to your mt-sdk root (with include/ and lib/).")
  endif()

  target_include_directories(mthud PRIVATE "${XSENS_INCLUDE_DIR}")
  target_link_libraries(mthud PRIVATE "${XSENS_LIB}" "${XSTYPES_LIB}")
else()
  target_compile_definitions(mthud PRIVATE USE_XSENS=0)
endif()

# ---- Runtime assets --------------------------------------------------------
foreach(f calibration.json calibration.yaml)
  if(EXISTS "${CMAKE_SOURCE_DIR}/src/${f}")
    add_custom_command(TARGET mthud POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CMAKE_SOURCE_DIR}/src/${f}" "$<TARGET_FILE_DIR:mthud>/")
  endif()
endforeach()

# ---- Install (optional) ----------------------------------------------------
install(TARGETS mthud RUNTIME DESTINATION bin)

