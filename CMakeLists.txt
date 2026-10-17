cmake_minimum_required(VERSION 3.13)
project(codec2_mod_neural C)

# Codec2-mod source tree (https://github.com/M17-Project/Codec2-mod)
set(C2MOD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Codec2-mod" CACHE PATH "Codec2-mod source directory")
option(C2F_AVX2 "x86-64 only: use AVX2 and FMA (Intel Haswell or AMD Excavator and newer)" ON)
option(C2F_TOOLS "build the command line tools" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_STANDARD 11)

# Codec2-mod as a static library, unless the parent project already has one
if(NOT TARGET codec2_mod)
  if(NOT EXISTS "${C2MOD_DIR}/inc/codec2_mod.h")
    message(FATAL_ERROR "Codec2-mod not found in ${C2MOD_DIR} - set -DC2MOD_DIR=...")
  endif()
  file(GLOB C2MOD_SRC "${C2MOD_DIR}/src/*.c")
  add_library(codec2_mod STATIC ${C2MOD_SRC})
  target_include_directories(codec2_mod PUBLIC "${C2MOD_DIR}/inc")
endif()
if(NOT MSVC)
  set(C2F_LIBM m)
endif()

set(OPUS third_party/opus)
add_library(c2fargan STATIC
  src/c2fargan.c
  src/c2f_features.c
  model/fargan_data.c
  ${OPUS}/dnn/fargan.c
  ${OPUS}/dnn/nnet.c
  ${OPUS}/dnn/nnet_default.c
  ${OPUS}/dnn/parse_lpcnet_weights.c)
target_include_directories(c2fargan
  PUBLIC include
  PRIVATE src model ${OPUS} ${OPUS}/dnn ${OPUS}/celt ${OPUS}/include)
target_compile_definitions(c2fargan PRIVATE SUPPRESS_PERF_WARNINGS)
target_link_libraries(c2fargan PUBLIC codec2_mod ${C2F_LIBM})

if(C2F_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC)
    target_compile_options(c2fargan PRIVATE /arch:AVX2)
  else()
    target_compile_options(c2fargan PRIVATE -mavx2 -mfma)
  endif()
endif()

if(C2F_TOOLS)
  add_executable(c2f_decode tools/c2f_decode.c)
  target_link_libraries(c2f_decode c2fargan)
  add_executable(c2enc tools/c2enc.c)
  target_link_libraries(c2enc codec2_mod ${C2F_LIBM})
endif()
