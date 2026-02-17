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

#ifndef CBIR_H
#define CBIR_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

struct Match {
    std::string filename;
    double distance;
};

bool compareMatches(const Match& a, const Match& b);

// main search function - dispatches to the right feature/distance for each task
std::vector<Match> find_matches(const std::string& target_image_path,
                                const std::string& task,
                                int num_results,
                                const std::string& image_database_path,
                                const std::string& csv_path = "ResNet18_olym.csv",
                                const std::string& dnn_metric = "cosine");

// feature extraction
cv::Mat baseline_features(const std::string& image_path);
cv::Mat rg_chromaticity_histogram(const cv::Mat& image, int bins);
cv::Mat rgb_histogram(const cv::Mat& image, int bins);
cv::Mat sobel_magnitude_histogram(const cv::Mat& image, int bins);
cv::Mat hsv_histogram(const cv::Mat& image, int h_bins, int s_bins);
cv::Mat top_region_hsv_histogram(const cv::Mat& image, int h_bins, int s_bins);
cv::Mat bottom_region_hsv_histogram(const cv::Mat& image, int h_bins, int s_bins);
double compute_edge_density(const cv::Mat& image);
double banana_feature(const cv::Mat& image);
double trash_can_feature(const cv::Mat& image);
int face_feature(const cv::Mat& image);
std::vector<double> gabor_feature(const cv::Mat& image);

// distance metrics
double ssd(const cv::Mat& f1, const cv::Mat& f2);
double histogram_intersection(const cv::Mat& h1, const cv::Mat& h2);
double cosine_distance(const std::vector<float>& v1, const std::vector<float>& v2);
double ssd_embedding(const std::vector<float>& v1, const std::vector<float>& v2);
double euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2);

// utilities
std::vector<std::string> get_image_files(const std::string& dir_path);
std::map<std::string, std::vector<float>> read_embeddings_csv(const std::string& csv_path);
std::string extract_filename(const std::string& path);

#endif // CBIR_H