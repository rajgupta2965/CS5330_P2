#ifndef CBIR_H
#define CBIR_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

// Match struct and comparator
struct Match {
    std::string filename;
    double distance;
};

bool compareMatches(const Match& a, const Match& b);

// Core image retrieval functions
std::vector<Match> find_matches(const std::string& target_image_path,
                                const std::string& task,
                                int num_results,
                                const std::string& image_database_path,
                                const std::string& csv_path = "ResNet18_olym.csv",
                                const std::string& dnn_metric = "cosine");

// Feature extraction functions
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

// Distance metric functions
double ssd(const cv::Mat& f1, const cv::Mat& f2);
double histogram_intersection(const cv::Mat& h1, const cv::Mat& h2);
double cosine_distance(const std::vector<float>& v1, const std::vector<float>& v2);
double ssd_embedding(const std::vector<float>& v1, const std::vector<float>& v2);
double euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2);

// Utility functions
std::vector<std::string> get_image_files(const std::string& dir_path);
std::map<std::string, std::vector<float>> read_embeddings_csv(const std::string& csv_path);
std::string extract_filename(const std::string& path);

#endif // CBIR_H
