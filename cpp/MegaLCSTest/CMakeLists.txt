add_executable(MegaLCSTest
        OpenCL/Test_CpuLCSMinMax.cpp
        OpenCL/Test_HostLCSShared.cpp
        OpenCL/Test_MegaLCSFusion_Coverage.cpp
        OpenCL/Test_MegaLCSFusion_Value.cpp
)

target_link_libraries(MegaLCSTest PRIVATE
        MegaLCSLib
        OpenCL::OpenCL
        GTest::GTest
        GTest::Main
)

target_include_directories(MegaLCSTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/MegaLCSLib
        ${PROJECT_SOURCE_DIR}/MegaLCSLib/OpenCL
)

add_executable(MegaLCSPerf
        OpenCL/Perf_HostLCSShared.cpp
)

target_link_libraries(MegaLCSPerf PRIVATE
        MegaLCSLib
        OpenCL::OpenCL
)
target_include_directories(MegaLCSPerf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/MegaLCSLib
        ${PROJECT_SOURCE_DIR}/MegaLCSLib/OpenCL
)