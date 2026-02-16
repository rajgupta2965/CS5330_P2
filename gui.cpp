#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "cbir.h"
#include "ImGuiFileDialog.h"

void matToTexture(const cv::Mat& mat, GLuint& textureID) {
    if (mat.empty()) return;
    if (textureID == 0) glGenTextures(1, &textureID);

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    cv::Mat bgra;
    cv::cvtColor(mat, bgra, cv::COLOR_BGR2BGRA);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bgra.cols, bgra.rows, 0, GL_BGRA, GL_UNSIGNED_BYTE, bgra.data);
}

int main(int, char**) {
    // Setup window
    if (!glfwInit()) return 1;
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "CBIR GUI", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    std::string target_image_path = "";
    GLuint target_texture = 0;
    std::vector<Match> matches;
    std::vector<GLuint> match_textures;
    
    const char* tasks[] = { "Baseline", "Histogram (Color)", "Multi-Histogram", "Texture & Color", "Deep Network (DNN)", "Custom DNN (ResNet18)", "Custom Design", "Blue Trash Can Finder", "Face Detector", "Gabor Filter (Texture)" };
    const char* task_keys[] = { "baseline", "histogram", "multi-histogram", "texture-color", "dnn", "custom_dnn", "custom", "trashcan", "face", "gabor" };
    static int current_task = 0;

    const char* results_options[] = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" };
    static int num_results = 3; // Corresponds to "4" in the options array

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Main GUI Window
        ImGui::SetNextWindowSize(ImVec2(800, 600));
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::Begin("Content-Based Image Retrieval");

        // File Dialog
        if (ImGui::Button("Browse...")) {
            const IGFD::FileDialogConfig config = { .path = "olympus/" };
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".jpg,.png", config);
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                target_image_path = ImGuiFileDialog::Instance()->GetFilePathName();
                cv::Mat img = cv::imread(target_image_path);
                matToTexture(img, target_texture);
                matches.clear();
                for(auto& tex : match_textures) glDeleteTextures(1, &tex);
                match_textures.clear();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::SameLine();
        ImGui::Text("Target Image: %s", target_image_path.c_str());

        // Target image display
        if (target_texture != 0) {
            ImGui::Image((void*)(intptr_t)target_texture, ImVec2(200, 200));
        }

        ImGui::Separator();

        // Controls
        ImGui::Combo("Task", &current_task, tasks, IM_ARRAYSIZE(tasks));
        ImGui::Combo("Number of Results", &num_results, results_options, IM_ARRAYSIZE(results_options));

        if (ImGui::Button("Execute Search")) {
            if (!target_image_path.empty()) {
                std::string task = task_keys[current_task];
                std::string csv_path = "ResNet18_olym.csv";
                if (task == "custom_dnn") {
                    csv_path = "Custom_ResNet18_olym.csv";
                }
                
                matches = find_matches(target_image_path, task, std::stoi(results_options[num_results]), "olympus", csv_path);
                
                // Clear old textures
                for(auto& tex : match_textures) glDeleteTextures(1, &tex);
                match_textures.assign(matches.size(), 0);

                // Create new textures
                for (size_t i = 0; i < matches.size(); ++i) {
                    cv::Mat img = cv::imread(matches[i].filename);
                    matToTexture(img, match_textures[i]);
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Close")) {
             glfwSetWindowShouldClose(window, true);
        }

        ImGui::Separator();

        // Results display
        ImGui::Text("Results:");
        int actual_num_results = std::stoi(results_options[num_results]);
        for (size_t i = 0; i < matches.size() && i < actual_num_results; ++i) {
            if (i % 5 != 0) ImGui::SameLine();
            ImGui::BeginGroup();
            if (match_textures[i] != 0) {
                ImGui::Image((void*)(intptr_t)match_textures[i], ImVec2(150, 150));
                ImGui::Text("Dist: %.4f", matches[i].distance);
            }
            ImGui::EndGroup();
        }

        ImGui::End();

        // Rendering
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
