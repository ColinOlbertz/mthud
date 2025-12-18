cmake_minimum_required(VERSION 3.20)

project(mthud LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
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
  src/sensor_xsens.cpp
)

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

  option(XSENS_USE_STATIC "Use locally built static Xsens libs (xspublic)" OFF)

  # Resolve SDK root from cache var, env, or bundled third_party/xsens
  if(NOT XSENS_SDK_ROOT AND DEFINED ENV{XSENS_SDK_ROOT})
    set(XSENS_SDK_ROOT "$ENV{XSENS_SDK_ROOT}")
  endif()
  if(NOT XSENS_SDK_ROOT)
    set(XSENS_SDK_ROOT "${PROJECT_SOURCE_DIR}/third_party/xsens")
  endif()
  set(XSENS_SDK_ROOT "${XSENS_SDK_ROOT}" CACHE PATH "Path to Xsens MT SDK root (contains include/ and lib/)")

  # Validate paths up front and point directly at the bundled copy
  set(XSENS_INCLUDE_DIR "${XSENS_SDK_ROOT}/include")
  if(NOT EXISTS "${XSENS_INCLUDE_DIR}/xsensdeviceapi.h")
    set(_xsens_fallback "${PROJECT_SOURCE_DIR}/third_party/xsens")
    if(NOT XSENS_SDK_ROOT STREQUAL "${_xsens_fallback}" AND EXISTS "${_xsens_fallback}/include/xsensdeviceapi.h")
      message(WARNING "XSENS_SDK_ROOT (${XSENS_SDK_ROOT}) invalid; falling back to bundled ${_xsens_fallback}")
      set(XSENS_SDK_ROOT "${_xsens_fallback}" CACHE PATH "Path to Xsens MT SDK root (contains include/ and lib/)" FORCE)
      set(XSENS_INCLUDE_DIR "${XSENS_SDK_ROOT}/include")
    else()
      message(FATAL_ERROR "Xsens headers not found under ${XSENS_INCLUDE_DIR}. Set XSENS_SDK_ROOT to the SDK root containing include/ and lib/.")
    endif()
  endif()

  if(XSENS_USE_STATIC)
    if(NOT XSENS_STATIC_ROOT AND DEFINED ENV{XSENS_STATIC_ROOT})
      set(XSENS_STATIC_ROOT "$ENV{XSENS_STATIC_ROOT}")
    endif()
    set(XSENS_STATIC_ROOT "${XSENS_STATIC_ROOT}" CACHE PATH "Root to xspublic (containing xscontroller/xscommon/xstypes) if using static libs")

    set(_xsroot "${XSENS_STATIC_ROOT}")
    if(NOT _xsroot)
      set(_xsroot "/tmp/xsens-example/xda_public_cpp/xspublic")
    endif()

    set(_XS_CTRL "${_xsroot}/xscontroller/libxscontroller.a")
    set(_XS_COMM "${_xsroot}/xscommon/libxscommon.a")
    set(_XS_TYPE "${_xsroot}/xstypes/libxstypes.a")

    if(NOT EXISTS "${_XS_CTRL}" OR NOT EXISTS "${_XS_COMM}" OR NOT EXISTS "${_XS_TYPE}")
      # Build static libs from vendored xspublic if they are missing
      find_program(MAKE_EXECUTABLE make)
    if(NOT MAKE_EXECUTABLE)
      message(FATAL_ERROR "XSENS_USE_STATIC=ON but static libs not found and 'make' is unavailable to build them under ${_xsroot}")
    endif()
    add_custom_target(xsens_static_libs ALL
      COMMAND ${MAKE_EXECUTABLE} -C "${_xsroot}"
        BYPRODUCTS "${_XS_CTRL}" "${_XS_COMM}" "${_XS_TYPE}"
        WORKING_DIRECTORY "${_xsroot}"
        COMMENT "Building Xsens static libraries from ${_xsroot}")
    endif()

    message(STATUS "Using Xsens static libs from ${_xsroot}")
    target_compile_definitions(mthud PRIVATE XSENS_USE_STATIC=1)
    if(NOT XSENS_INCLUDE_DIR)
      set(XSENS_INCLUDE_DIR "${_xsroot}")
    endif()
    target_include_directories(mthud PRIVATE
      "${XSENS_INCLUDE_DIR}"
      "${_xsroot}" "${_xsroot}/xscontroller" "${_xsroot}/xscommon" "${_xsroot}/xstypes")

    target_link_libraries(mthud PRIVATE
      "${_XS_CTRL}" "${_XS_COMM}" "${_XS_TYPE}"
      pthread rt dl)
    if(TARGET xsens_static_libs)
      add_dependencies(mthud xsens_static_libs)
    endif()
  else()
    if(WIN32)
      set(_xs_lib_dir "${XSENS_SDK_ROOT}/lib")
      set(XSENS_LIB   "${_xs_lib_dir}/xsensdeviceapi64.lib")
      set(XSTYPES_LIB "${_xs_lib_dir}/xstypes64.lib")
      if(NOT EXISTS "${XSENS_LIB}" OR NOT EXISTS "${XSTYPES_LIB}")
        message(FATAL_ERROR "Xsens libraries not found under ${_xs_lib_dir}. Expected xsensdeviceapi64.lib and xstypes64.lib. Set XSENS_SDK_ROOT if your layout differs.")
      endif()
    else()
      set(_xs_lib_dir "${XSENS_SDK_ROOT}/lib")
      set(XSENS_LIB   "${_xs_lib_dir}/libxsensdeviceapi.so")
      set(XSTYPES_LIB "${_xs_lib_dir}/libxstypes.so")
      if(NOT EXISTS "${XSENS_LIB}" OR NOT EXISTS "${XSTYPES_LIB}")
        message(FATAL_ERROR "Xsens libraries not found under ${_xs_lib_dir}. Expected libxsensdeviceapi.so and libxstypes.so. Set XSENS_SDK_ROOT if your layout differs.")
      endif()

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

    target_include_directories(mthud PRIVATE "${XSENS_INCLUDE_DIR}")
    target_link_libraries(mthud PRIVATE "${XSENS_LIB}" "${XSTYPES_LIB}")

    if(WIN32)
      # Drop the DLLs next to the exe so runtime can find them
      file(GLOB _XSENS_DLLS "${XSENS_SDK_ROOT}/lib/*.dll")
      add_custom_command(TARGET mthud POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_XSENS_DLLS} "$<TARGET_FILE_DIR:mthud>/"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:mthud>/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${XSENS_SDK_ROOT}/lib/platforms/qwindows.dll"
                "$<TARGET_FILE_DIR:mthud>/platforms/")
    endif()
  endif()
else()
  target_compile_definitions(mthud PRIVATE USE_XSENS=0)
endif()

# ---- Runtime assets --------------------------------------------------------
foreach(f calibration.json calibration.yaml)
  if(EXISTS "${CMAKE_SOURCE_DIR}/${f}")
    add_custom_command(TARGET mthud POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CMAKE_SOURCE_DIR}/${f}" "$<TARGET_FILE_DIR:mthud>/")
  endif()
endforeach()

# Copy OpenCV runtime DLLs next to the executable (helps ensure HighGUI backend is found)
set(_opencv_bin_dir "")
if(OpenCV_LIBS)
  list(GET OpenCV_LIBS 0 _opencv_lib_hint)
  get_filename_component(_opencv_lib_dir "${_opencv_lib_hint}" DIRECTORY)
  get_filename_component(_opencv_root "${_opencv_lib_dir}" DIRECTORY)
  set(_opencv_bin_dir "${_opencv_root}/bin")
endif()
if(EXISTS "${_opencv_bin_dir}")
  file(GLOB _opencv_runtime_dlls "${_opencv_bin_dir}/opencv_*.dll")
  if(_opencv_runtime_dlls)
    add_custom_command(TARGET mthud POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${_opencv_runtime_dlls}
              "$<TARGET_FILE_DIR:mthud>/")
  endif()
endif()

# ---- Install (optional) ----------------------------------------------------
install(TARGETS mthud RUNTIME DESTINATION bin)
