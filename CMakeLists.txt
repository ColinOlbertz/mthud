cmake_minimum_required(VERSION 3.20)
project(mthud LANGUAGES CXX)

# ---- toolchain-agnostic project setup ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Warnings
if (MSVC)
  add_compile_options(/W4 /EHsc /permissive- /Zc:preprocessor)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Let code find repo assets at runtime if you use PROJECT_SOURCE_DIR in C++
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---- dependencies (resolved by vcpkg manifest) ----
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(Freetype CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Ask OpenCV for specific modules, including ArUco
# If COMPONENTS fails on your setup, drop "COMPONENTS ..." and link modules explicitly.
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc highgui videoio aruco)

# ---- build target ----
add_executable(mthud
  src/main.cpp
  src/app.hpp
  src/sensor.hpp
  src/aruco_tracker.hpp
  src/aruco_tracker.cpp
  src/hud_renderer.hpp
  src/hud_renderer.cpp
  src/config.hpp
  src/config.cpp
)

# ---- XSens (optional, ON by default on Windows, OFF elsewhere) ----
option(USE_XSENS "Build with XSens MT SDK" ${WIN32})

if (USE_XSENS)
  message(STATUS "XSens support enabled")

  # Select headers/libs per platform.
  if (WIN32)
    # Prefer an environment variable; fall back to default install path.
    set(XSENS_SDK_ROOT "$ENV{XSENS_SDK_ROOT}" CACHE PATH "XSens MT SDK root")
    if (NOT XSENS_SDK_ROOT)
      set(XSENS_SDK_ROOT "C:/Program Files/Xsens/MT SDK/x64")
    endif()
    set(XSENS_INCLUDE_DIR "${XSENS_SDK_ROOT}/include")
    set(XSENS_LIB_DIR     "${XSENS_SDK_ROOT}/lib")
    set(XSENS_LIBS xsensdeviceapi64 xstypes64)

    target_include_directories(mthud PRIVATE "${XSENS_INCLUDE_DIR}")
    target_link_directories(mthud  PRIVATE "${XSENS_LIB_DIR}")
    target_link_libraries(mthud    PRIVATE ${XSENS_LIBS})

    # Copy Windows runtime DLLs next to the exe (Windows only)
    add_custom_command(TARGET mthud POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${XSENS_LIB_DIR}/xsensdeviceapi64.dll" "$<TARGET_FILE_DIR:mthud>/"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${XSENS_LIB_DIR}/xstypes64.dll" "$<TARGET_FILE_DIR:mthud>/"
    )
  elseif(UNIX)
    # On Linux you must install the MT SDK yourself or supply paths.
    # Example layout if you placed the SDK under /opt/xsens:
    set(XSENS_SDK_ROOT "$ENV{XSENS_SDK_ROOT}" CACHE PATH "XSens MT SDK root")
    if (XSENS_SDK_ROOT)
      set(XSENS_INCLUDE_DIR "${XSENS_SDK_ROOT}/include")
      set(XSENS_LIB_DIR     "${XSENS_SDK_ROOT}/lib")
      # Typical Linux libs are libxstypes.so and libxsensdeviceapi.so
      find_library(XSTYPES   NAMES xstypes   PATHS "${XSENS_LIB_DIR}")
      find_library(XSDEVICE  NAMES xsensdeviceapi PATHS "${XSENS_LIB_DIR}")

      if (XSTYPES AND XSDEVICE)
        target_include_directories(mthud PRIVATE "${XSENS_INCLUDE_DIR}")
        target_link_directories(mthud  PRIVATE "${XSENS_LIB_DIR}")
        target_link_libraries(mthud    PRIVATE ${XSDEVICE} ${XSTYPES})
      else()
        message(FATAL_ERROR "XSens libs not found under ${XSENS_SDK_ROOT}. Set XSENS_SDK_ROOT or disable USE_XSENS.")
      endif()
    else()
      message(FATAL_ERROR "Set XSENS_SDK_ROOT on Linux or configure with -DUSE_XSENS=OFF.")
    endif()
  endif()

  target_compile_definitions(mthud PRIVATE USE_XSENS=1)
  # Only compile the XSens source when enabled.
  target_sources(mthud PRIVATE src/sensor_xsens.cpp)
else()
  message(STATUS "XSens support disabled")
  target_compile_definitions(mthud PRIVATE USE_XSENS=0)
  # If you have a stub, add it here so the build still provides a sensor interface.
  if (EXISTS "${CMAKE_SOURCE_DIR}/src/sensor_stub.cpp")
    target_sources(mthud PRIVATE src/sensor_stub.cpp)
  endif()
endif()

# ---- link third-party libraries ----
target_link_libraries(mthud PRIVATE
  glfw
  glad::glad
  Freetype::Freetype
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  opencv_core
  opencv_imgproc
  opencv_highgui
  opencv_videoio
  opencv_aruco
)

# ---- example: copy runtime data if you still depend on side-by-side files ----
foreach(f calibration.json calibration.yaml)
  if (EXISTS "${CMAKE_SOURCE_DIR}/src/${f}")
    add_custom_command(TARGET mthud POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/${f}" "$<TARGET_FILE_DIR:mthud>/")
  endif()
endforeach()

# ---- Intel OpenMP DLL copy (Windows only, optional) ----
if (WIN32)
  set(_iomp_hint_dirs
    "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.2/redist/intel64_win/compiler"
    "$ENV{ONEAPI_ROOT}/compiler/latest/windows/redist/intel64_win/compiler"
    "$ENV{ONEAPI_ROOT}/redist/intel64_win/compiler"
  )
  find_file(LIBIOMP5_DLL NAMES libiomp5md.dll PATHS ${_iomp_hint_dirs} NO_DEFAULT
  )