cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

include(cmake_config/Paths.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
add_compile_options(-Wno-psabi)

include_directories(
    ${M1_SDK_INC_DIR}
    ${M1_SDK_INC_DIR}/smartsoc
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/task
    ${CMAKE_SOURCE_DIR}/app
)

file(GLOB PLATFORM_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(eye_tracking_core
    app/eye_tracking_app.cpp
    task/eye_tracking.cpp
    task/tracking.cpp
    task/attention.cpp
    ${PLATFORM_SOURCES}
)

target_link_libraries(eye_tracking_core
    ${M1_SSNE_LIB}
    ${M1_CMABUFFER_LIB}
    ${M1_OSD_LIB}
    ${M1_SSZLOG_LIB}
    ${M1_ZLOG_LIB}
    ${M1_EMB_LIB}
)

add_executable(ssne_ai_demo app/main.cpp)
target_compile_definitions(ssne_ai_demo PRIVATE EYE_TRACK_DEFAULT_CLASSIC=1)
target_link_libraries(ssne_ai_demo eye_tracking_core)

add_executable(ssne_ai_demo_model app/main.cpp)
target_link_libraries(ssne_ai_demo_model eye_tracking_core)

install(TARGETS ssne_ai_demo ssne_ai_demo_model DESTINATION bin)
