# Project 2: Content-based Image Retrieval

## Team Members
- **Sangeeth Deleep Menon** | NUID: 002524579 | MSCS - Boston | CS5330 Section 03 (CRN: 40669, Online)
- **Raj Gupta** | NUID: 002068701 | MSCS - Boston | CS5330 Section 01 (CRN: 38745, Online)

## Project Description
This project implements a content-based image retrieval (CBIR) system that, given a target image, finds the most visually similar images from a database. The system supports multiple feature extraction methods (baseline pixel matching, color histograms, multi-region histograms, texture+color, deep network embeddings) and distance metrics (SSD, histogram intersection, cosine distance). A custom retrieval method combining HSV color analysis with texture and spatial features is also implemented.

## Usage
For instructions on how to build and run the project, please see the [USEME.md](USEME.md) file.

## Project Report
A detailed report on the project, including results and analysis, can be found in [REPORT.md](REPORT.md).

## Operating System and IDE
- **OS:** macOS (Apple Silicon / arm64)
- **IDE:** CLion
- **Compiler:** Apple Clang (c++) with C++20
- **Build System:** CMake 3.20+
- **Dependencies:** OpenCV 4.x (installed via Homebrew)

## Methods Overview

| Task | Feature Vector | Distance Metric |
|------|---------------|----------------|
| 1. Baseline | 7x7 center pixel patch | Sum of Squared Differences (SSD) |
| 2. Histogram | rg chromaticity histogram (16x16 bins) | Histogram Intersection |
| 3. Multi-histogram | Top/bottom half RGB histograms (8 bins each) | Weighted avg of histogram intersection |
| 4. Texture-Color | Whole-image RGB histogram + Sobel magnitude histogram (8 bins) | 50/50 weighted histogram intersection |
| 5. DNN Embeddings | 512-dim ResNet18 features from CSV | Cosine distance (or SSD) |
| 7. Custom | HSV histograms (whole, top 1/3, bottom 1/3) + Sobel texture + edge density | Weighted combination (30/20/20/15/15) |

## Extensions
None.

## Time Travel Days
None used.

## Videos
None.

## Acknowledgements
- OpenCV documentation for histogram and Sobel filter references
- Course materials and sample code provided by Prof. Bruce Maxwell
- Shapiro and Stockman, Chapter 8 for histogram matching concepts