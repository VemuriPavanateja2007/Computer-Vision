# Computer Vision — Practicals

A collection of hands-on Computer Vision lab practicals implemented in Python using **OpenCV**, **NumPy**, **Matplotlib**, and **TensorFlow/Keras**. Each notebook is a self-contained Google Colab practical that implements a core computer vision concept from first principles — from basic image transformations to feature detection, edge detection, and a convolutional neural network for image classification.

This repository serves as a structured, practical companion to a Computer Vision course, demonstrating working knowledge of classical image processing techniques as well as an introductory deep learning application.

---

## 📌 About This Repository

Each notebook in `Practicals/` tackles one specific CV concept, follows a consistent format (load image → process → visualize before/after), and is built to run directly in Google Colab (each notebook includes an "Open in Colab" badge and mounts Google Drive for image access). The practicals progress from fundamental pixel-level operations to spatial filtering, edge/line detection, feature extraction, and finally a deep learning-based classifier — mirroring the typical structure of an academic Computer Vision lab course.

---

## 🗂️ Repository Structure

```
Computer-Vision/
└── Practicals/
    ├── CV_pract-1.ipynb    # Intensity level reduction (quantization)
    ├── CV_pract_2.ipynb    # Image zooming & shrinking — Nearest Neighbor interpolation
    ├── CV_Pract_3.ipynb    # Image zooming & shrinking — Bilinear interpolation
    ├── CR_Pract_4.ipynb    # Arithmetic operations on images
    ├── CR_Pract_6.ipynb    # Spatial filtering — smoothing & sharpening
    ├── CR_Pract_7.ipynb    # Image sharpening using the Laplacian operator
    ├── CR_Pract_8.ipynb    # Unsharp masking & high-boost filtering
    ├── CV_Pract_9.ipynb    # Feature detection — SIFT (and SURF)
    ├── CR_Pract_10.ipynb   # Edge detection — Sobel & Canny
    ├── CR_Pract_12.ipynb   # Line detection — Hough Transform
    ├── CR_Pract_13.ipynb   # Image thresholding & segmentation
    └── CR_Pract_14.ipynb   # CNN-based digit classification (MNIST) using TensorFlow/Keras
```

> Practical numbers 5 and 11 are not present in the current repository.

---

## 📖 Practical-by-Practical Breakdown

### 1. Intensity Level Reduction — `CV_pract-1.ipynb`
Demonstrates image quantization by reducing the number of grayscale intensity levels (e.g., from 256 levels down to fewer discrete levels such as 4), visualized side by side in a grid to show the resulting loss of detail/posterization effect.

### 2. Image Zooming & Shrinking (Nearest Neighbor) — `CV_pract_2.ipynb`
Resizes an image up (zoom) and down (shrink) using **Nearest Neighbor interpolation** (`cv2.INTER_NEAREST`), printing original, zoomed, and shrunk image dimensions to confirm the transformation.

### 3. Image Zooming & Shrinking (Bilinear) — `CV_Pract_3.ipynb`
Repeats the zoom/shrink exercise using **Bilinear interpolation** (`cv2.INTER_LINEAR`) instead of nearest neighbor, allowing a direct visual and dimensional comparison between interpolation methods.

### 4. Arithmetic Operations on Images — `CR_Pract_4.ipynb`
Implements pixel-wise image arithmetic:
- **Addition** and **Subtraction** of two images (`cv2.add`, `cv2.subtract`)
- **Multiplication** to increase brightness by a scalar factor (`cv2.multiply`)
- **Blending** two images with weighted combination — 70%/30% — using `cv2.addWeighted`

### 6. Spatial Filtering — Smoothing & Sharpening — `CR_Pract_6.ipynb`
A comprehensive filtering practical covering both linear and non-linear smoothing filters, plus gradient-based sharpening:
- **Linear smoothing:** Mean/Averaging filter, Gaussian filter
- **Non-linear smoothing:** Median filter, Min filter (erosion), Max filter (dilation)
- **First-order sharpening (edge/gradient) filters:** Sobel, Prewitt (via `filter2D`), Roberts Cross
- Produces a final sharpened image by combining the original with the Sobel gradient

### 7. Image Sharpening — Laplacian Operator — `CR_Pract_7.ipynb`
Applies the **Laplacian operator** (second-order derivative) to detect edges, then sharpens the image by adding the Laplacian result back onto the original image.

### 8. Unsharp Masking & High-Boost Filtering — `CR_Pract_8.ipynb`
Implements classical sharpening via masking:
- Blurs the image with a Gaussian filter to obtain a low-frequency version
- Computes the **mask** as (original − blurred)
- Applies **standard unsharp masking** (k = 1)
- Applies **high-boost filtering** (A = 1.5) using `cv2.addWeighted`

### 9. Feature Detection — SIFT / SURF — `CV_Pract_9.ipynb`
Detects and visualizes keypoints using the **SIFT (Scale-Invariant Feature Transform)** algorithm (`cv2.SIFT_create`), drawing rich keypoints on the image and reporting the number of keypoints and descriptor shape. Also attempts **SURF** for comparison (gracefully handles it being unavailable in standard OpenCV builds due to patent restrictions).

### 10. Edge Detection — Sobel & Canny — `CR_Pract_10.ipynb`
Compares two classic edge detection techniques:
- **Sobel edge detection** with gradient magnitude computation and thresholding into a binary edge map
- **Canny edge detection** using OpenCV's built-in dual-threshold detector

### 12. Line Detection — Hough Transform — `CR_Pract_12.ipynb`
Detects straight lines in an image using edge detection as a preprocessing step, then applies:
- **Standard Hough Transform** (`cv2.HoughLines`)
- **Probabilistic Hough Transform** (`cv2.HoughLinesP`)

Both sets of detected lines are drawn on the original image for comparison.

### 13. Image Thresholding & Segmentation — `CR_Pract_13.ipynb`
Covers four thresholding techniques for image segmentation:
- **Global thresholding** with a manually chosen value (T = 127)
- **Otsu's thresholding**, where the optimal threshold is computed automatically
- **Adaptive Mean thresholding**
- **Adaptive Gaussian thresholding**

### 14. CNN-Based Digit Classification (MNIST) — `CR_Pract_14.ipynb`
The deep learning capstone of the series — builds, trains, and evaluates a **Convolutional Neural Network** using TensorFlow/Keras on the MNIST handwritten digit dataset:
- Loads and normalizes the MNIST dataset
- Builds a CNN architecture using `tensorflow.keras.layers` and `models`
- Trains the model and evaluates test accuracy
- Visualizes sample predictions and plots training accuracy/loss curves

---

## 🧠 Concepts Covered

**Image Fundamentals**
- Intensity quantization, image arithmetic, interpolation-based resizing

**Spatial Filtering**
- Smoothing: Mean, Gaussian, Median, Min/Max filters
- Sharpening: Sobel, Prewitt, Roberts, Laplacian, Unsharp Masking, High-Boost Filtering

**Feature & Edge Detection**
- SIFT feature/keypoint detection
- Sobel & Canny edge detection
- Hough Transform for line detection

**Segmentation**
- Global, Otsu, and Adaptive Thresholding

**Deep Learning for Vision**
- CNN architecture design, training, and evaluation on MNIST

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab / Jupyter Notebook
- **Core Libraries:**
  - `opencv-python` (`cv2`) — image I/O and core CV operations
  - `numpy` — numerical/array operations on image data
  - `matplotlib` — visualization of results
  - `tensorflow` / `keras` — CNN model building and training (Practical 14)
- **Platform notes:** Notebooks use `google.colab.drive` to mount Google Drive for image access, so images are expected to be loaded from a Drive path when run in Colab.

---

## 🚀 Getting Started

### Option 1 — Run in Google Colab (recommended)
Each notebook has a built-in **"Open in Colab"** badge. Click it, mount your Google Drive when prompted, and update the image file paths to point to your own sample images.

### Option 2 — Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/VemuriPavanateja2007/Computer-Vision.git
   cd Computer-Vision/Practicals
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy matplotlib tensorflow jupyter
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. **Remove or replace the Google Colab-specific cell** (`from google.colab import drive; drive.mount(...)`) with a local file path to your test images before running.

> Most notebooks expect a sample image (e.g., `road.jpg` or similar) in the working directory or a mounted Drive path — check the `cv2.imread(...)` call near the top of each notebook and update the path accordingly.

---

## 🎯 Purpose & Learning Outcomes

This repository was built as a structured, hands-on record of core Computer Vision concepts, reinforcing:

- How classical filtering (linear vs. non-linear, smoothing vs. sharpening) affects image quality and edge information
- The mathematical intuition behind gradient-based (Sobel, Prewitt, Roberts) vs. second-derivative-based (Laplacian) edge/sharpening operators
- How thresholding techniques adapt to different lighting conditions across an image (global vs. adaptive)
- How the Hough Transform converts an edge map into detected geometric shapes (lines)
- How local feature descriptors like SIFT enable scale- and rotation-invariant keypoint matching
- The end-to-end workflow of building and training a CNN for image classification with TensorFlow/Keras

---

## 🔭 Future Improvements

- [ ] Add a `requirements.txt` for reproducible local setup
- [ ] Include sample test images in the repo so notebooks run out-of-the-box
- [ ] Add the missing Practicals 5 and 11 (if applicable) for a complete sequence
- [ ] Add markdown explanations/theory cells at the top of each notebook
- [ ] Convert repeated OpenCV boilerplate (image loading, display grids) into a shared utility module

---

## 👤 Author

**Vemuri Venkata Satya Markandeya Pavanateja**
Final-year B.Sc Artificial Intelligence student, Government College (Autonomous), Rajahmundry

- LinkedIn: [linkedin.com/in/vemurivenkata-satya-markandeyapavanateja-0651a0403](https://www.linkedin.com/in/vemurivenkata-satya-markandeyapavanateja-0651a0403)
- Email: pavanatejavemuri2007@gmail.com

---

## 📄 License

This repository does not currently specify a license. Consider adding an [MIT License](https://choosealicense.com/licenses/mit/) if you'd like others to freely reuse or build on these practicals.

---

⭐ If you find these practicals useful for learning Computer Vision fundamentals, consider giving the repo a star!
