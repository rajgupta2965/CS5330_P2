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

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

// Function to get list of image files from a directory
std::vector<std::string> get_image_files(const std::string& dir_path) {
    std::vector<std::string> files;
    DIR *dirp;
    struct dirent *dp;

    if ((dirp = opendir(dir_path.c_str())) == NULL) {
        std::cerr << "Cannot open directory " << dir_path << std::endl;
        return files;
    }

    while ((dp = readdir(dirp)) != NULL) {
        std::string filename = dp->d_name;
        if (filename.find(".jpg") != std::string::npos ||
            filename.find(".png") != std::string::npos ||
            filename.find(".ppm") != std::string::npos ||
            filename.find(".tif") != std::string::npos) {
            files.push_back(dir_path + "/" + filename);
        }
    }
    closedir(dirp);
    return files;
}

// --- Baseline Matching ---
cv::Mat baseline_features(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return cv::Mat();
    }
    int x = image.cols / 2 - 3;
    int y = image.rows / 2 - 3;
    cv::Rect roi(x, y, 7, 7);
    return image(roi).clone();
}

double ssd(const cv::Mat& f1, const cv::Mat& f2) {
    double sum = 0;
    for (int i = 0; i < f1.rows; ++i) {
        for (int j = 0; j < f1.cols; ++j) {
            for (int c = 0; c < f1.channels(); ++c) {
                double diff = f1.at<cv::Vec3b>(i, j)[c] - f2.at<cv::Vec3b>(i, j)[c];
                sum += diff * diff;
            }
        }
    }
    return sum;
}

// --- Histogram Matching ---
cv::Mat rg_chromaticity_histogram(const cv::Mat& image, int bins) {
    cv::Mat histogram = cv::Mat::zeros(bins, bins, CV_32F);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            float b = pixel[0];
            float g = pixel[1];
            float r = pixel[2];
            float sum = r + g + b;

            if (sum > 0) {
                float r_chroma = r / sum;
                float g_chroma = g / sum;

                int r_bin = std::min((int)(r_chroma * bins), bins - 1);
                int g_bin = std::min((int)(g_chroma * bins), bins - 1);

                histogram.at<float>(r_bin, g_bin)++;
            }
        }
    }
    
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram;
}

// --- Multi-histogram Matching ---
cv::Mat rgb_histogram(const cv::Mat& image, int bins) {
    int histSize[] = {bins, bins, bins};
    cv::Mat histogram = cv::Mat::zeros(3, histSize, CV_32F);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            int b_bin = std::min(pixel[0] * bins / 256, bins - 1);
            int g_bin = std::min(pixel[1] * bins / 256, bins - 1);
            int r_bin = std::min(pixel[2] * bins / 256, bins - 1);
            histogram.at<float>(r_bin, g_bin, b_bin)++;
        }
    }
    
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram;
}


double histogram_intersection(const cv::Mat& h1, const cv::Mat& h2) {
    double intersection = 0;
    if (h1.dims == 2) {
        for (int i = 0; i < h1.rows; ++i) {
            for (int j = 0; j < h1.cols; ++j) {
                intersection += std::min(h1.at<float>(i, j), h2.at<float>(i, j));
            }
        }
    } else if (h1.dims == 3) {
        for (int i = 0; i < h1.size[0]; ++i) {
            for (int j = 0; j < h1.size[1]; ++j) {
                for (int k = 0; k < h1.size[2]; ++k) {
                    intersection += std::min(h1.at<float>(i, j, k), h2.at<float>(i, j, k));
                }
            }
        }
    }
    return 1.0 - intersection;
}


struct Match {
    std::string filename;
    double distance;
};

bool compareMatches(const Match& a, const Match& b) {
    return a.distance < b.distance;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <task> <target_image> <num_results>" << std::endl;
        std::cout << "Tasks: baseline, histogram, multi-histogram" << std::endl;
        return -1;
    }

    std::string task = argv[1];
    std::string target_image_path = argv[2];
    int num_results = std::stoi(argv[3]);
    std::string image_database_path = "olympus";

    std::vector<std::string> image_files = get_image_files(image_database_path);
    std::vector<Match> matches;

    if (task == "baseline") {
        cv::Mat target_features = baseline_features(target_image_path);
        if (target_features.empty()) return -1;

        for (const auto& file_path : image_files) {
            if (file_path == target_image_path) continue;
            cv::Mat current_features = baseline_features(file_path);
            if (current_features.empty()) continue;
            double distance = ssd(target_features, current_features);
            matches.push_back({file_path, distance});
        }
    } else if (task == "histogram") {
        int bins = 16;
        cv::Mat image = cv::imread(target_image_path);
        if (image.empty()) return -1;
        cv::Mat target_hist = rg_chromaticity_histogram(image, bins);
        
        for (const auto& file_path : image_files) {
            if (file_path == target_image_path) continue;
            cv::Mat current_image = cv::imread(file_path);
            if(current_image.empty()) continue;
            cv::Mat current_hist = rg_chromaticity_histogram(current_image, bins);
            double distance = histogram_intersection(target_hist, current_hist);
            matches.push_back({file_path, distance});
        }
    } else if (task == "multi-histogram") {
        int bins = 8;
        cv::Mat target_image = cv::imread(target_image_path);
        if (target_image.empty()) return -1;
        cv::Mat target_top = target_image(cv::Rect(0, 0, target_image.cols, target_image.rows / 2));
        cv::Mat target_bottom = target_image(cv::Rect(0, target_image.rows / 2, target_image.cols, target_image.rows / 2));
        cv::Mat target_hist_top = rgb_histogram(target_top, bins);
        cv::Mat target_hist_bottom = rgb_histogram(target_bottom, bins);

        for (const auto& file_path : image_files) {
            if (file_path == target_image_path) continue;
            cv::Mat current_image = cv::imread(file_path);
            if(current_image.empty()) continue;
            cv::Mat current_top = current_image(cv::Rect(0, 0, current_image.cols, current_image.rows / 2));
            cv::Mat current_bottom = current_image(cv::Rect(0, current_image.rows / 2, current_image.cols, current_image.rows / 2));
            cv::Mat current_hist_top = rgb_histogram(current_top, bins);
            cv::Mat current_hist_bottom = rgb_histogram(current_bottom, bins);
            
            double dist_top = histogram_intersection(target_hist_top, current_hist_top);
            double dist_bottom = histogram_intersection(target_hist_bottom, current_hist_bottom);
            matches.push_back({file_path, (dist_top + dist_bottom) / 2.0});
        }
    } else {
        std::cerr << "Invalid task: " << task << std::endl;
        return -1;
    }

    std::sort(matches.begin(), matches.end(), compareMatches);

    std::cout << "Top " << num_results << " matches for " << target_image_path << ":" << std::endl;
    for (int i = 0; i < num_results && i < matches.size(); ++i) {
        std::cout << matches[i].filename << " (distance: " << matches[i].distance << ")" << std::endl;
    }

    return 0;
}