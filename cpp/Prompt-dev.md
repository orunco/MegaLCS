## 现有目录
```
[D:\Work\Code\1_exchideas.github.com\Orunco-MegaLCS\cpp]$ tree /f
cpp
│  CMakeLists.txt
├─MegaLCSLib
│  └─OpenCL
│          Mega.cpp
│          Mega.h
│          Mega.Kernel.Shared.cpp
|          Mega.Host.cpp
│    CMakeLists.txt 
├─MegaLCSTest
│    Test_HostLCSShared.cpp
│    CMakeLists.txt
```

## 现有构建配置
{|text:[](vcpkg.json)|}
{|text:[](CMakeLists.txt)|}
{|text:[](MegaLCSLib/CMakeLists.txt)|}
{|text:[](MegaLCSTest/CMakeLists.txt)|}

## 现有代码
声明
{|text:[](MegaLCSLib/OpenCL/Mega.h)|}
实现
{|text:[](MegaLCSLib/OpenCL/Mega.cpp)|}
{|text:[](MegaLCSLib/OpenCL/Mega.Kernel.Shared.cpp)|}
{|text:[](MegaLCSLib/OpenCL/Mega.Host.cpp)|}

测试：
{|text:[](MegaLCSTest/OpenCL/Test_HostLCSShared.cpp)|}

## 指令
MegaLCSLib/CMakeLists.txt 中的 add_library需要罗列每一个cpp h，可以*.cpp *.h?