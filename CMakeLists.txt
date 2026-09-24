cmake_minimum_required(VERSION 3.16)

project(image_filtration LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)

# Явно указываем пути к библиотекам из vcpkg (убираем зависимость от toolchain)
set(CMAKE_PREFIX_PATH
    C:/vcpkg/installed/x64-windows
    C:/vcpkg/installed/x64-windows/share
)

# Явно указываем директории для find_package
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4")
set(Qt6_DIR "D:/Programs/Qt/6.9.0/msvc2022_64/lib/cmake/Qt6")

# Находим библиотеки
find_package(OpenCV REQUIRED)


find_package(Qt6 REQUIRED COMPONENTS Core)

add_executable(image_filtration main.cpp)

target_link_libraries(image_filtration
    Qt6::Core
    ${OpenCV_LIBS}
)

# Сообщаем о найденных библиотеках
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "Qt6 found")
