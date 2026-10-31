cmake_minimum_required(VERSION 3.16)
project(robotics_evidence_lab_cpp_2 VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_robotics_evidence_core tests/test_robotics_evidence_core.cpp)

enable_testing()
add_test(NAME TestRoboticsEvidenceAndNISTCompliance COMMAND test_robotics_evidence_core)
