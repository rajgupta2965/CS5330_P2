/*
Name: Sangeeth Deleep Menon
NUID: 002524579
Program: MSCS - Boston
Course: CS5330; Pattern Recognition and Computer Vision
Section: 03 | CRN: 40669 | Online

Name: Raj Gupta
NUID: 002068701
Program: MSCS - Boston
Course: CS5330; Pattern Recognition and Computer Vision
Section: 01 | CRN: 38745 | Online
*/

#include "cbir.h"
#include <iostream>
#include <string>
#include <vector>

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <task> <target_image> <num_results> [csv_path] [dnn_metric]" << std::endl;
        std::cout << "Tasks: baseline, histogram, multi-histogram, texture-color, dnn, custom" << std::endl;
        std::cout << "For dnn task: provide csv_path as 4th arg" << std::endl;
        std::cout << "Optional dnn_metric: cosine (default) or ssd" << std::endl;
        return -1;
    }

    std::string task = argv[1];
    std::string target_image_path = argv[2];
    int num_results = std::stoi(argv[3]);
    std::string image_database_path = "olympus";

    // Optional arguments
    std::string csv_path = (argc > 4) ? argv[4] : "ResNet18_olym.csv";
    std::string dnn_metric = (argc > 5) ? argv[5] : "cosine";

    // Find matches using the core function
    std::vector<Match> matches = find_matches(target_image_path, task, num_results, image_database_path, csv_path, dnn_metric);

    if (task == "dnn" && matches.empty()) {
        std::cerr << "Could not find matches for DNN task. Please check if the target image is in the CSV and the CSV path is correct." << std::endl;
        return -1;
    }
    
    // Display results
    std::cout << "\nTop " << num_results << " matches for " << target_image_path << " (" << task << "):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    for (int i = 0; i < num_results && i < (int)matches.size(); ++i) {
        std::cout << (i + 1) << ". " << matches[i].filename
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }

    // Also show least similar (for Task 7 analysis)
    if (task == "custom" && matches.size() > 3) {
        std::cout << "\nLeast similar images:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        for (int i = std::max(0, (int)matches.size() - 3); i < (int)matches.size(); ++i) {
            std::cout << matches[i].filename
                      << " (distance: " << matches[i].distance << ")" << std::endl;
        }
    }

    return 0;
}