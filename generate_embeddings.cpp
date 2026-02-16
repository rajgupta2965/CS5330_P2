#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "cbir.h"

int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net) {
    const int ORNet_size = 224;
    cv::Mat blob;

    cv::dnn::blobFromImage(src, blob, (1.0/255.0) * (1/0.226), cv::Size(ORNet_size, ORNet_size), cv::Scalar(124, 116, 104), true, false, CV_32F);
    
    net.setInput(blob);
    embedding = net.forward();
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_directory> <output_csv_path>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_dir = argv[2];
    std::string output_csv = argv[3];

    // Load the network
    cv::dnn::Net net = cv::dnn::readNet(model_path);
    if (net.empty()) {
        std::cerr << "Could not load the network model: " << model_path << std::endl;
        return -1;
    }
    std::cout << "Network loaded successfully." << std::endl;

    // Get image files
    std::vector<std::string> image_files = get_image_files(image_dir);
    if (image_files.empty()) {
        std::cerr << "No images found in directory: " << image_dir << std::endl;
        return -1;
    }
    std::cout << "Found " << image_files.size() << " images to process." << std::endl;

    // Open output file
    std::ofstream csv_file(output_csv);
    if (!csv_file.is_open()) {
        std::cerr << "Could not open output CSV file: " << output_csv << std::endl;
        return -1;
    }

    // Process each image and write to CSV
    for (const auto& image_path : image_files) {
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "Could not read image: " << image_path << std::endl;
            continue;
        }

        cv::Mat embedding;
        getEmbedding(img, embedding, net);

        // Write to CSV
        csv_file << extract_filename(image_path);
        for (int i = 0; i < embedding.cols; ++i) {
            csv_file << "," << embedding.at<float>(0, i);
        }
        csv_file << "\n";
        
        std::cout << "Processed: " << extract_filename(image_path) << std::endl;
    }

    csv_file.close();
    std::cout << "Embeddings successfully generated and saved to " << output_csv << std::endl;

    return 0;
}
