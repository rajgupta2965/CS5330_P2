#include "cbir.h"
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

// grabs all image files from a directory and sorts them
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
    std::sort(files.begin(), files.end());
    return files;
}

// sort matches by distance (ascending)
bool compareMatches(const Match& a, const Match& b) {
    return a.distance < b.distance;
}

// Task 1 - extracts 7x7 center patch as feature
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

// sum of squared differences between two patches
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

// Task 2 - rg chromaticity histogram (brightness invariant)
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

// Task 3 - 3D RGB histogram
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

// Task 4 - sobel gradient magnitude histogram for texture
cv::Mat sobel_magnitude_histogram(const cv::Mat& image, int bins) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat sobel_x, sobel_y;
    cv::Sobel(gray, sobel_x, CV_32F, 1, 0);
    cv::Sobel(gray, sobel_y, CV_32F, 0, 1);

    cv::Mat magnitude;
    cv::magnitude(sobel_x, sobel_y, magnitude);

    cv::Mat histogram = cv::Mat::zeros(1, bins, CV_32F);
    double max_val;
    cv::minMaxLoc(magnitude, nullptr, &max_val);

    if (max_val == 0) max_val = 1;

    for (int i = 0; i < magnitude.rows; ++i) {
        for (int j = 0; j < magnitude.cols; ++j) {
            int bin = std::min((int)(magnitude.at<float>(i, j) * bins / max_val), bins - 1);
            histogram.at<float>(0, bin)++;
        }
    }

    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram;
}

// histogram intersection - handles 1D, 2D, 3D histograms
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
    } else if (h1.dims == 1 || h1.rows == 1 || h1.cols == 1) {
        for (int i = 0; i < (int)h1.total(); ++i) {
            intersection += std::min(h1.at<float>(i), h2.at<float>(i));
        }
    }
    return 1.0 - intersection;
}

// Task 5 - reads ResNet18 embeddings from CSV file
std::map<std::string, std::vector<float>> read_embeddings_csv(const std::string& csv_path) {
    std::map<std::string, std::vector<float>> embeddings;
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        std::cerr << "Cannot open CSV file: " << csv_path << std::endl;
        return embeddings;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string filename;
        std::getline(ss, filename, ',');

        // strip whitespace from filename
        filename.erase(0, filename.find_first_not_of(" \t\r\n"));
        filename.erase(filename.find_last_not_of(" \t\r\n") + 1);

        std::vector<float> features;
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                features.push_back(std::stof(value));
            } catch (...) {
                // skip bad values
            }
        }

        if (!features.empty()) {
            embeddings[extract_filename(filename)] = features;
        }
    }

    file.close();
    return embeddings;
}

// d = 1 - cos(theta) between two vectors
double cosine_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size() || v1.empty()) return 2.0;

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 == 0 || norm2 == 0) return 2.0;

    double cosine_sim = dot / (norm1 * norm2);
    return 1.0 - cosine_sim;
}

// SSD for embedding vectors
double ssd_embedding(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) return 1e18;
    double sum = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

// pulls just the filename from a full path
std::string extract_filename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

// Task 7 - 2D hue-saturation histogram
cv::Mat hsv_histogram(const cv::Mat& image, int h_bins, int s_bins) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat histogram = cv::Mat::zeros(h_bins, s_bins, CV_32F);

    for (int i = 0; i < hsv.rows; ++i) {
        for (int j = 0; j < hsv.cols; ++j) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(i, j);
            int h_bin = std::min((int)(pixel[0] * h_bins / 180), h_bins - 1);
            int s_bin = std::min((int)(pixel[1] * s_bins / 256), s_bins - 1);
            histogram.at<float>(h_bin, s_bin)++;
        }
    }

    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram;
}

// ratio of canny edge pixels to total pixels
double compute_edge_density(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    int edge_pixels = cv::countNonZero(edges);
    return (double)edge_pixels / (double)(edges.rows * edges.cols);
}

// HSV histogram for top third of image
cv::Mat top_region_hsv_histogram(const cv::Mat& image, int h_bins, int s_bins) {
    int top_height = image.rows / 3;
    cv::Mat top_region = image(cv::Rect(0, 0, image.cols, top_height));
    return hsv_histogram(top_region, h_bins, s_bins);
}

// HSV histogram for bottom third of image
cv::Mat bottom_region_hsv_histogram(const cv::Mat& image, int h_bins, int s_bins) {
    int top_height = image.rows / 3;
    int bottom_start = image.rows - top_height;
    cv::Mat bottom_region = image(cv::Rect(0, bottom_start, image.cols, top_height));
    return hsv_histogram(bottom_region, h_bins, s_bins);
}

// scores how likely an image contains a banana (yellow + elongated shape)
double banana_feature(const cv::Mat& image) {
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    // yellow range - wide enough for ripe to slightly green bananas
    cv::Scalar lower_yellow = cv::Scalar(15, 80, 80);
    cv::Scalar upper_yellow = cv::Scalar(40, 255, 255);

    cv::Mat mask;
    cv::inRange(hsv_image, lower_yellow, upper_yellow, mask);

    // clean up noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // how much of the image is yellow overall
    double total_yellow_pixels = cv::countNonZero(mask);
    double total_pixels = image.rows * image.cols;
    double yellow_percentage = (total_yellow_pixels / total_pixels) * 100.0;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_elongation_score = 0.0;

    // check up to 5 biggest contours for banana-like shape
    for (size_t i = 0; i < std::min(contours.size(), (size_t)5); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 200) continue;

        cv::RotatedRect minRect = cv::minAreaRect(contours[i]);
        float w = minRect.size.width;
        float h = minRect.size.height;
        if (std::min(w, h) < 1) continue;
        float aspect_ratio = std::max(w, h) / std::min(w, h);

        // gaussian peaked at aspect ratio ~3 (typical banana shape)
        double elongation = std::exp(-0.5 * std::pow((aspect_ratio - 3.0) / 1.5, 2));

        // solidity check - bananas are roughly 0.6-0.8 (curved, not fully convex)
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);
        double hull_area = cv::contourArea(hull);
        double solidity = (hull_area > 0) ? area / hull_area : 0;
        double solidity_score = std::exp(-0.5 * std::pow((solidity - 0.7) / 0.15, 2));

        double shape_score = 0.6 * elongation + 0.4 * solidity_score;
        best_elongation_score = std::max(best_elongation_score, shape_score);
    }

    // combine color coverage and shape, both normalized to 0-1
    double color_score = std::min(yellow_percentage / 10.0, 1.0);
    double score = (0.5 * color_score + 0.5 * best_elongation_score) * 100.0;

    return std::min(score, 100.0);
}

// scores how likely an image contains a blue trash can
double trash_can_feature(const cv::Mat& image) {
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    // blue range - low sat/val thresholds to catch bins in shadow
    cv::Scalar lower_blue = cv::Scalar(95, 50, 40);
    cv::Scalar upper_blue = cv::Scalar(140, 255, 255);

    cv::Mat mask;
    cv::inRange(hsv_image, lower_blue, upper_blue, mask);

    // clean up with rect kernel (better for boxy shapes)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    double total_blue_pixels = cv::countNonZero(mask);
    double total_pixels = image.rows * image.cols;
    double blue_percentage = (total_blue_pixels / total_pixels) * 100.0;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_shape_score = 0.0;

    for (size_t i = 0; i < std::min(contours.size(), (size_t)5); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 300) continue;

        cv::RotatedRect minRect = cv::minAreaRect(contours[i]);
        float w = minRect.size.width;
        float h = minRect.size.height;
        if (std::min(w, h) < 1) continue;
        float aspect_ratio = std::max(w, h) / std::min(w, h);

        // gaussian peaked at ~1.5 aspect ratio (upright bin shape)
        double shape_score = std::exp(-0.5 * std::pow((aspect_ratio - 1.5) / 0.8, 2));

        // how well the contour fills its bounding rect (bins are boxy)
        double rect_area = w * h;
        double rectangularity = (rect_area > 0) ? area / rect_area : 0;
        double rect_score = std::min(rectangularity / 0.7, 1.0);

        double combined = 0.5 * shape_score + 0.5 * rect_score;
        best_shape_score = std::max(best_shape_score, combined);
    }

    // combine blue coverage and shape, both on 0-1 scale
    double color_score = std::min(blue_percentage / 8.0, 1.0);
    double score = (0.5 * color_score + 0.5 * best_shape_score) * 100.0;

    return std::min(score, 100.0);
}

// counts faces using haar cascade - tries several paths for the xml
int face_feature(const cv::Mat& image) {
    static cv::CascadeClassifier face_cascade;
    static bool cascade_loaded = false;
    static bool load_attempted = false;
    if (!cascade_loaded && !load_attempted) {
        load_attempted = true;
        std::vector<std::string> cascade_paths = {
            std::string(PROJECT_ROOT_DIR) + "/haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_default.xml"
        };
        for (const auto& path : cascade_paths) {
            if (face_cascade.load(path)) {
                cascade_loaded = true;
                std::cout << "Face cascade loaded from: " << path << std::endl;
                break;
            }
        }
        if (!cascade_loaded) {
            std::cerr << "Warning: Could not load face cascade from any known path." << std::endl;
        }
    }

    if (!cascade_loaded || image.empty()) {
        return 0;
    }

    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    return (int)faces.size();
}

// gabor filter bank - 4 orientations x 2 frequencies, returns mean+stddev per filter
std::vector<double> gabor_feature(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    std::vector<double> features;
    double a[] = {0, CV_PI/4.0, CV_PI/2.0, 3.0*CV_PI/4.0};
    double b[] = {5.0, 10.0};

    for (double theta : a) {
        for (double lambda : b) {
            cv::Mat kernel = cv::getGaborKernel(cv::Size(15, 15), 3.0, theta, lambda, 0.5, 0, CV_32F);
            cv::Mat dest;
            cv::filter2D(gray, dest, CV_32F, kernel);

            cv::Scalar mean, stddev;
            cv::meanStdDev(dest, mean, stddev);
            features.push_back(mean.val[0]);
            features.push_back(stddev.val[0]);
        }
    }
    return features;
}

double euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return std::sqrt(sum);
}

// dispatches to the right feature/distance combo for each task
std::vector<Match> find_matches(const std::string& target_image_path,
                                const std::string& task,
                                int num_results,
                                const std::string& image_database_path,
                                const std::string& csv_path,
                                const std::string& dnn_metric) {
    std::vector<Match> matches;

    if (task == "dnn" || task == "custom_dnn") {
        // try csv path as-is, then fall back to project root
        std::string resolved_csv = csv_path;
        {
            std::ifstream test(resolved_csv);
            if (!test.is_open()) {
                resolved_csv = std::string(PROJECT_ROOT_DIR) + "/" + csv_path;
                std::ifstream test2(resolved_csv);
                if (!test2.is_open()) {
                    std::cerr << "Could not find CSV at: " << csv_path << " or " << resolved_csv << std::endl;
                    return matches;
                }
            }
        }

        std::map<std::string, std::vector<float>> embeddings = read_embeddings_csv(resolved_csv);
        if (embeddings.empty()) {
            std::cerr << "No embeddings loaded from: " << resolved_csv << std::endl;
            return matches;
        }
        std::cout << "Loaded " << embeddings.size() << " embeddings from " << resolved_csv << std::endl;

        std::string target_fname = extract_filename(target_image_path);
        std::vector<float> target_embedding;

        if (embeddings.count(target_fname)) {
            target_embedding = embeddings[target_fname];
        } else {
            // print first few CSV keys so we can see what went wrong
            std::cerr << "Target not found in CSV: '" << target_fname << "'" << std::endl;
            std::cerr << "First 5 keys in CSV: ";
            int count = 0;
            for (const auto& p : embeddings) {
                if (count++ >= 5) break;
                std::cerr << "'" << p.first << "' ";
            }
            std::cerr << std::endl;
        }
        if (target_embedding.empty()) return matches;

        for (const auto& pair : embeddings) {
            std::string fname = extract_filename(pair.first);
            if (fname == target_fname) continue;
            double distance = (dnn_metric == "ssd") ? ssd_embedding(target_embedding, pair.second) : cosine_distance(target_embedding, pair.second);
            std::string full_path = image_database_path + "/" + fname;
            matches.push_back({full_path, distance});
        }
    } else {
        std::vector<std::string> image_files = get_image_files(image_database_path);
        cv::Mat target_image = cv::imread(target_image_path);
        if (target_image.empty() && task != "banana" && task != "trashcan") return matches;

        // pre-compute target features for face task so we don't redo it every iteration
        int target_face_feature = 0;
        cv::Mat target_face_hsv;
        if (task == "face") {
            target_face_feature = face_feature(target_image);
            target_face_hsv = hsv_histogram(target_image, 16, 16);
            std::cout << "Target face count: " << target_face_feature << std::endl;
        }

        std::vector<double> target_gabor_feature;
        if (task == "gabor") {
            target_gabor_feature = gabor_feature(target_image);
        }

        for (const auto& file_path : image_files) {
            if (extract_filename(file_path) == extract_filename(target_image_path)) continue;
            cv::Mat current_image = cv::imread(file_path);
            if (current_image.empty()) continue;

            double distance = 0.0;
            if (task == "baseline") {
                cv::Mat target_features = baseline_features(target_image_path);
                cv::Mat current_features = baseline_features(file_path);
                if (!target_features.empty() && !current_features.empty()) {
                    distance = ssd(target_features, current_features);
                }
            } else if (task == "histogram") {
                int bins = 16;
                cv::Mat target_hist = rg_chromaticity_histogram(target_image, bins);
                cv::Mat current_hist = rg_chromaticity_histogram(current_image, bins);
                distance = histogram_intersection(target_hist, current_hist);
            } else if (task == "histogram2") {
                int bins = 8;
                cv::Mat target_hist = rgb_histogram(target_image, bins);
                cv::Mat current_hist = rgb_histogram(current_image, bins);
                distance = histogram_intersection(target_hist, current_hist);
            } else if (task == "histogram3") {
                int h_bins = 32, s_bins = 32;
                cv::Mat target_hist = hsv_histogram(target_image, h_bins, s_bins);
                cv::Mat current_hist = hsv_histogram(current_image, h_bins, s_bins);
                distance = histogram_intersection(target_hist, current_hist);
            } else if (task == "multi-histogram") {
                int bins = 8;
                cv::Mat target_top = target_image(cv::Rect(0, 0, target_image.cols, target_image.rows / 2));
                cv::Mat target_bottom = target_image(cv::Rect(0, target_image.rows / 2, target_image.cols, target_image.rows / 2));
                cv::Mat target_hist_top = rgb_histogram(target_top, bins);
                cv::Mat target_hist_bottom = rgb_histogram(target_bottom, bins);
                cv::Mat current_top = current_image(cv::Rect(0, 0, current_image.cols, current_image.rows / 2));
                cv::Mat current_bottom = current_image(cv::Rect(0, current_image.rows / 2, current_image.cols, current_image.rows / 2));
                cv::Mat current_hist_top = rgb_histogram(current_top, bins);
                cv::Mat current_hist_bottom = rgb_histogram(current_bottom, bins);
                double dist_top = histogram_intersection(target_hist_top, current_hist_top);
                double dist_bottom = histogram_intersection(target_hist_bottom, current_hist_bottom);
                distance = (dist_top + dist_bottom) / 2.0;
            } else if (task == "texture-color") {
                int bins = 8;
                cv::Mat target_color_hist = rgb_histogram(target_image, bins);
                cv::Mat target_texture_hist = sobel_magnitude_histogram(target_image, bins);
                cv::Mat current_color_hist = rgb_histogram(current_image, bins);
                cv::Mat current_texture_hist = sobel_magnitude_histogram(current_image, bins);
                double color_dist = histogram_intersection(target_color_hist, current_color_hist);
                double texture_dist = histogram_intersection(target_texture_hist, current_texture_hist);
                distance = 0.5 * color_dist + 0.5 * texture_dist;
            } else if (task == "custom") {
                int h_bins = 16, s_bins = 16;
                cv::Mat target_whole_hsv = hsv_histogram(target_image, h_bins, s_bins);
                cv::Mat target_top_hsv = top_region_hsv_histogram(target_image, h_bins, s_bins);
                cv::Mat target_bottom_hsv = bottom_region_hsv_histogram(target_image, h_bins, s_bins);
                double target_edge_density = compute_edge_density(target_image);
                cv::Mat target_texture = sobel_magnitude_histogram(target_image, 16);

                cv::Mat curr_whole_hsv = hsv_histogram(current_image, h_bins, s_bins);
                cv::Mat curr_top_hsv = top_region_hsv_histogram(current_image, h_bins, s_bins);
                cv::Mat curr_bottom_hsv = bottom_region_hsv_histogram(current_image, h_bins, s_bins);
                double curr_edge_density = compute_edge_density(current_image);
                cv::Mat curr_texture = sobel_magnitude_histogram(current_image, 16);

                double whole_hsv_dist = histogram_intersection(target_whole_hsv, curr_whole_hsv);
                double top_hsv_dist = histogram_intersection(target_top_hsv, curr_top_hsv);
                double bottom_hsv_dist = histogram_intersection(target_bottom_hsv, curr_bottom_hsv);
                double edge_dist = std::abs(target_edge_density - curr_edge_density);
                double texture_dist = histogram_intersection(target_texture, curr_texture);

                distance = 0.30 * whole_hsv_dist + 0.20 * top_hsv_dist + 0.20 * bottom_hsv_dist + 0.15 * texture_dist + 0.15 * edge_dist;
            } else if (task == "banana") {
                double current_banana_feature = banana_feature(current_image);
                distance = 100.0 - current_banana_feature;
            } else if (task == "trashcan") {
                double current_trashcan_feature = trash_can_feature(current_image);
                distance = 100.0 - current_trashcan_feature;
            } else if (task == "face") {
                int current_face_feature = face_feature(current_image);
                double face_diff = std::abs(target_face_feature - current_face_feature);

                // use color as tiebreaker so same-face-count images aren't random
                cv::Mat current_hsv = hsv_histogram(current_image, 16, 16);
                double color_dist = histogram_intersection(target_face_hsv, current_hsv);
                distance = face_diff + 0.5 * color_dist;

                // penalize no-face images when target has faces
                if (target_face_feature > 0 && current_face_feature == 0) {
                    distance += 10.0;
                }
            } else if (task == "gabor") {
                std::vector<double> current_gabor_feature = gabor_feature(current_image);
                distance = euclidean_distance(target_gabor_feature, current_gabor_feature);
            }
            matches.push_back({file_path, distance});
        }
    }

    std::sort(matches.begin(), matches.end(), compareMatches);
    return matches;
}