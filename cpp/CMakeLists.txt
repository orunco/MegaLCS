cmake_minimum_required(VERSION 3.10)
project(MegaLCSWorkspace)

# gtest需要大于17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 使用 vcpkg manifest 模式 - 修正路径
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

# 查找依赖包
set(OpenCL_DIR "$ENV{VCPKG_ROOT}/packages/opencl_x64-mingw-dynamic/share/opencl")
find_package(OpenCL CONFIG REQUIRED)

set(GTest_DIR "$ENV{VCPKG_ROOT}/packages/gtest_x64-mingw-dynamic/share/gtest/")
find_package(GTest REQUIRED)

# 需要在WINDOWS的PATH中加入C:\xx\bin\mingw\bin
# 以及%VCPKG_ROOT%\packages\gtest_x64-mingw-dynamic\debug\bin\
# 否则会提示找不到libgtest_main.dll

# 添加子目录
add_subdirectory(MegaLCSLib)
add_subdirectory(MegaLCSTest)

# 创建一个包含所有目标的元目标
add_custom_target(all_projects ALL
        DEPENDS MegaLCSLib MegaLCSTest
)
