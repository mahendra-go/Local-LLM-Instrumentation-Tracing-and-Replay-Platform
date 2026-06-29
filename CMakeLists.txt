cmake_minimum_required(VERSION 3.16)
project(llmtrace VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LLMTRACE_ENABLE_LLAMA "Build the llama.cpp capture backend" OFF)

find_package(Threads REQUIRED)

include(FetchContent)

# Terminal UI library (header + compiled components), fetched & built locally.
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
  GIT_TAG v5.0.0
  GIT_SHALLOW TRUE
)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)

add_executable(llmtrace
  src/main.cpp
  src/event.cpp
  src/topology.cpp
  src/trace_io.cpp
  src/source.cpp
  src/session.cpp
  src/dashboard.cpp
)
target_include_directories(llmtrace PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(llmtrace PRIVATE LLMTRACE_VERSION="${PROJECT_VERSION}")
target_link_libraries(llmtrace PRIVATE
  ftxui::screen ftxui::dom ftxui::component
  Threads::Threads
)
