cmake_minimum_required(VERSION 3.16)
project(llmtrace VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LLMTRACE_ENABLE_LLAMA "Build the llama.cpp capture backend" OFF)

add_executable(llmtrace
  src/main.cpp
  src/event.cpp
  src/topology.cpp
  src/trace_io.cpp
  src/source.cpp
)
target_include_directories(llmtrace PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(llmtrace PRIVATE LLMTRACE_VERSION="${PROJECT_VERSION}")
