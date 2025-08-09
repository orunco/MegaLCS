file(GLOB_RECURSE SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_library(MegaLCSLib ${SOURCES})

target_link_libraries(MegaLCSLib PRIVATE OpenCL::OpenCL)

target_include_directories(MegaLCSLib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL
)
