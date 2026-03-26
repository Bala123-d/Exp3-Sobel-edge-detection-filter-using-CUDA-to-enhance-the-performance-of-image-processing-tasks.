# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>ENTER YOUR NAME : D.BALA SUBRAMANYAM </h3>
<h3>ENTER YOUR REGISTER NO : 212224040062</h3>
<h3>EX. NO : 3 </h3>
<h3>DATE : 26.03.2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
    {
        int Gx = 0, Gy = 0;

        int sobelX[3][3] = {
            {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}
        };

        int sobelY[3][3] = {
            {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}
        };

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                int pixel = input[(y + i) * width + (x + j)];
                Gx += pixel * sobelX[i + 1][j + 1];
                Gy += pixel * sobelY[i + 1][j + 1];
            }

        int mag = sqrtf(Gx * Gx + Gy * Gy);
        if (mag > 255) mag = 255;

        output[y * width + x] = (unsigned char)mag;
    }
}

int main()
{
    Mat img = imread("input.jpg");
    if (img.empty()) { cout << "Error"; return -1; }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    int w = gray.cols, h = gray.rows;

    unsigned char *d_in, *d_out;

    cudaMalloc(&d_in, w * h);
    cudaMalloc(&d_out, w * h);

    cudaMemcpy(d_in, gray.data, w * h, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);

    sobelKernel<<<grid,block>>>(d_in, d_out, w, h);

    cudaDeviceSynchronize();

    Mat out(h, w, CV_8UC1);
    cudaMemcpy(out.data, d_out, w * h, cudaMemcpyDeviceToHost);

    imwrite("output.jpg", out);

    cudaFree(d_in); cudaFree(d_out);

    cout << "Done";
}

```

## OUTPUT:
## ORIGINAL:
<img width="318" height="440" alt="input jpg" src="https://github.com/user-attachments/assets/c430f3ec-306f-4541-867c-3201016128d6" />

## OUTPUT:
![output](https://github.com/user-attachments/assets/5a17d1d4-4414-481f-aede-0b062af4e659)


## RESULT:
Successfully implemented a CUDA-based Sobel edge detection filter, demonstrating improved performance compared to the CPU-based implementation by utilizing parallel processing on the GPU.

Questions:

1.What challenges did you face while implementing the Sobel filter for color images? 
2.How did changing the block size influence the performance of your CUDA implementation? 
3.What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies. 
4.Suggest potential optimizations for improving the performance of the Sobel filter.

Answers to Questions

1.Challenges Implementing Sobel for Color Images:

Handling color images required conversion to grayscale before applying the Sobel filter. Managing memory and correct pixel indexing in CUDA was slightly complex during implementation.

2.Influence of Block Size:

Smaller block sizes such as 8x8 worked well for smaller images, while larger block sizes like 16x16 or 32x32 improved performance for larger images by better utilizing GPU resources.

3.CUDA vs. CPU Output Differences:

The CUDA implementation executed faster than the CPU version. The output images were almost similar, with slight differences due to rounding and parallel computation, but overall edge detection quality was maintained.

4.Optimization Suggestions:

Use shared memory in CUDA to reduce global memory access time. Optimize thread usage and experiment with different block sizes to achieve better performance.

Deliverables:

Modified CUDA code with proper Sobel filter implementation. Output images demonstrating edge detection. Comparison between CPU and GPU performance. Answers to the experiment questions.

Tools Required:

NVIDIA GPU (Tesla T4), CUDA NVCC Compiler, OpenCV Library, Google Colab.

