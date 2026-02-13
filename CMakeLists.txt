cmake_minimum_required(VERSION 3.20)
project(Project2)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(OpenGL REQUIRED) # Add OpenGL explicitly

# Common source files for both command-line and GUI
set(COMMON_SOURCES cbir.cpp)

# Command-line application
add_executable(Project2 main.cpp ${COMMON_SOURCES})
target_link_libraries(Project2 ${OpenCV_LIBS})

# GUI application
add_executable(Project2_GUI
    gui.cpp
    ${COMMON_SOURCES}
    libs/imgui_file_dialog/ImGuiFileDialog.cpp # ImGuiFileDialog source
    
    # ImGui core sources
    libs/imgui/imgui.cpp
    libs/imgui/imgui_draw.cpp
    libs/imgui/imgui_tables.cpp
    libs/imgui/imgui_widgets.cpp
    
    # ImGui backends for GLFW and OpenGL3
    libs/imgui/backends/imgui_impl_glfw.cpp
    libs/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(Project2_GUI PRIVATE
    libs/imgui # ImGui headers
    libs/imgui/backends # ImGui backend headers
    libs/imgui_file_dialog # ImGuiFileDialog headers
)

target_link_libraries(Project2_GUI PRIVATE
    ${OpenCV_LIBS}
    glfw
    OpenGL::GL # Link against OpenGL target from find_package(OpenGL)
)

# Embedding generator utility
add_executable(generate_embeddings generate_embeddings.cpp ${COMMON_SOURCES})
target_link_libraries(generate_embeddings ${OpenCV_LIBS})
