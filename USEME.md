# How to Use This Project

This document provides instructions on how to build and run the content-based image retrieval system.

## Building the Project

This project uses CMake and requires OpenCV 4.x to be installed.

1.  **Navigate to the project directory.**

2.  **Create a build directory and run CMake and make:**
    ```bash
    mkdir -p cmake-build-debug && cd cmake-build-debug
    cmake -DCMAKE_BUILD_TYPE=Debug ..
    make
    ```
    This will create two executables inside the `cmake-build-debug` directory:
    *   `Project2`: The command-line version.
    *   `Project2_GUI`: The interactive GUI version.

## Running the Command-Line Application

The program is run from the command line with several arguments to specify the desired operation.

### Command-Line Arguments

The executable takes the following arguments:
```
./cmake-build-debug/Project2 <task> <target_image> <num_results> [csv_path] [dnn_metric]
```

| Argument      | Description                                                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------|
| `task`        | The matching method to use. One of: `baseline`, `histogram`, `multi-histogram`, `texture-color`, `dnn`, `custom`. |
| `target_image`| Path to the target image (e.g., `olympus/pic.1016.jpg`).                                                    |
| `num_results` | The number of top matches to return.                                                                      |
| `csv_path`    | (Optional, for `dnn` task only) Path to the ResNet18 embeddings CSV file. Defaults to `ResNet18_olym.csv`.   |
| `dnn_metric`  | (Optional, for `dnn` task only) The distance metric to use. `cosine` (default) or `ssd`.                     |

**Note:** The image database is assumed to be in a directory named `olympus/` relative to where you run the command.

### Example Commands

Here are examples for each task:

**1. Baseline Matching**
```bash
./cmake-build-debug/Project2 baseline olympus/pic.1016.jpg 4
```

**2. Histogram Matching**
```bash
./cmake-build-debug/Project2 histogram olympus/pic.0164.jpg 3
```

**3. Multi-histogram Matching**
```bash
./cmake-build-debug/Project2 multi-histogram olympus/pic.0274.jpg 3
```

**4. Texture and Color**
```bash
./cmake-build-debug/Project2 texture-color olympus/pic.0535.jpg 3
```

**5. Deep Network Embeddings**
```bash
./cmake-build-debug/Project2 dnn olympus/pic.0893.jpg 3 ResNet18_olym.csv cosine
```

**6. Custom Design (HSV + Texture + Spatial)**
```bash
./cmake-build-debug/Project2 custom olympus/pic.0535.jpg 5
```

## Running the GUI Application

The GUI provides an interactive way to explore the image retrieval system.

### To Run the GUI:
1.  Open your terminal and navigate to the project's root directory.
2.  Execute the following command:
    ```bash
    ./cmake-build-debug/Project2_GUI
    ```

### How to Use the GUI:
*   A window titled "CBIR GUI" will open, displaying a target image.
*   **Change Target Image**:
    *   Press the **'n'** key to view the **next** image in the `olympus` directory.
    *   Press the **'p'** key to view the **previous** image.
*   **Set Search Parameters**:
    *   Use the **"Task"** trackbar to select the matching method (0 for `baseline`, 1 for `histogram`, etc.).
    *   Use the **"Results"** trackbar to set the number of matches to display.
*   **Run Search**:
    *   With the desired target image displayed, press the **'s'** key to start the search.
*   **View Results**:
    *   A new window titled "Search Results" will appear, showing the target and top matching images.
*   **Quit**:
    *   Press **'q'** or the **ESC** key to close all windows and quit the application.
