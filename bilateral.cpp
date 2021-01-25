#include <algorithm>
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include "FreeImage.h"
#include <math.h>
#include <CL/cl.h>
#include <omp.h>

constexpr int MAX_SOURCE_SIZE = 16384;

// module load CUDA
// g++ bilateral.cpp -O3 -Wall -Wextra -pedantic -lm -lOpenCL -fopenmp -std=c++17 -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o bilateral
// srun -n1 -u -G1 --reservation=fri ./bilateral
// srun -n1 -u --cpus-per-task=8 --reservation=fri ./bilateral

namespace Memoization {
    static bool enabled = true;
    static double *distanceBetweenPixels; // S
    static std::array<double, 256> intensityBetweenColors; // V
}

namespace Gauss {
    static int sigma_v = 16;
    static int sigma_s = 4; // sigma_s * 3

    double distance(int x, int y) {
        return exp(-(x * x + y * y) / (2 * sigma_s * sigma_s));
    }

    double intensity(int v, bool cache = false) {
        if (Memoization::enabled && !cache) {
            return Memoization::intensityBetweenColors[v];
        }
        return exp(-(v * v) / (2 * sigma_v * sigma_v));
    }

    double distance(int x, int y, int w, bool cache = false) {
        if (!Memoization::enabled || cache) {
            return distance(x, y);
        }
        int id = y * w + x;
        if (Memoization::distanceBetweenPixels[id] == 0) {
            Memoization::distanceBetweenPixels[id] = distance(x, y);
        }
        return Memoization::distanceBetweenPixels[id];
    }
}

namespace BilateralCPU {
    void filter(unsigned char *image_in, unsigned char *image_out, int x, int y, int width, int height, int w) {
        const int id = (y * width + x) * 4;
        image_out[id + 3] = image_in[id + 3]; // A
        // image_out[id + 2] = image_in[id + 2]; // R
        // image_out[id + 1] = image_in[id + 1]; // G
        // image_out[id + 0] = image_in[id + 0]; // B

        double FR = 0;
        double FG = 0;
        double FB = 0;
        double WR = 0;
        double WG = 0;
        double WB = 0;
        unsigned char* currentPixel = &image_in[(std::clamp(y, 0, height - 1) * width + std::clamp(x, 0, width - 1)) * 4];
        for (int r = x - w; r < x + w; r++) {
            for (int s = y - w; s < y + w; s++) {
                unsigned char* pixelNeighbor = &image_in[(std::clamp(s, 0, height - 1) * width + std::clamp(r, 0, width - 1)) * 4];
                double gs = Gauss::distance(abs(r - x), abs(s - y), w);
                double tR = gs * Gauss::intensity(abs(*(currentPixel + 2) - *(pixelNeighbor + 2)));
                double tG = gs * Gauss::intensity(abs(*(currentPixel + 1) - *(pixelNeighbor + 1)));
                double tB = gs * Gauss::intensity(abs(*(currentPixel + 0) - *(pixelNeighbor + 0)));
                FR += *(pixelNeighbor + 2) * tR;
                FG += *(pixelNeighbor + 1) * tG;
                FB += *(pixelNeighbor + 0) * tB;
                WR += tR;
                WG += tG;
                WB += tB;
            }
        }
        image_out[id + 2] = FR / WR;
        image_out[id + 1] = FG / WG;
        image_out[id + 0] = FB / WB;
    }

    void sequential(unsigned char *image_in, unsigned char *image_out, int width, int height) {
        int w = (2 * (3 * Gauss::sigma_s) + 1) / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                filter(image_in, image_out, x, y, width, height, w);
            }
        }
    }

    void openmp(unsigned char *image_in, unsigned char *image_out, int width, int height, int numThreads) {
        int w = (2 * (3 * Gauss::sigma_s) + 1) / 2;

        if (numThreads != -1) {
            omp_set_num_threads(numThreads);
        }
        // We want to capture both loops, not only outer one, hence collapse(2)
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                filter(image_in, image_out, x, y, width, height, w);
            }
        }
    }
}

namespace BilateralGPU {
    static cl_int ret;
    static cl_kernel kernel;
    static cl_command_queue command_queue;
    static cl_context context;
    static cl_program program;

    static size_t local_item_size;
    static size_t global_item_size;
    
    static cl_mem image_mem_obj_in;
    static cl_mem image_mem_obj_out;
    static cl_mem intensity_mem_obj_in;
    static cl_mem distance_mem_obj_in;

    void setup(std::string kernelFile, std::string kernelFunction, unsigned char *image_in, int width, int height, int workgroup_size) {
        int imageSize = height * width * 4 * sizeof(unsigned char);

        FILE *fp = fopen(kernelFile.c_str(), "r");
        if (!fp) {
            std::cerr << "Error: Could not open kernel." << std::endl;
            exit(1);
        }

        // Write kernel to memory
        char *source_str = (char*) malloc(MAX_SOURCE_SIZE);
        size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        source_str[source_size] = '\0';
        fclose(fp);

        // Platform data
        cl_platform_id platform_id[10];
        cl_uint ret_num_platforms;
        ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms); // Max number of platforms, pointer to platforms, actual number of platforms

        // Device data
        cl_device_id device_id[10];
        cl_uint ret_num_devices;
        // Get platform[0] on GPU
        ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);				
            // Chosen platform, device type, how many devices
            // pointer to devices, actual number of devices

        // Context
        context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
            // Context: included platforms - NULL is default, number of devices, 
            // pointer to device(s), pointer to call-back function in case of error
            // additional parameters, error code
    
        // Command queue
        command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
            // Context, device, INORDER / OUTOFORDER, error code

        // Workgroup
        local_item_size = workgroup_size;
        size_t num_groups = (((width * height) - 1) / local_item_size + 1);
        global_item_size = num_groups * local_item_size;

        // size_t local_item_size[] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
        // size_t num_groups[] = { ((width - 1) / local_item_size[0] + 1), ((height - 1) / local_item_size[1] + 1) };
        // size_t global_item_size[] = { num_groups[0] * local_item_size[0], num_groups[1] * local_item_size[1] };

        // Memory allocation on the device
        // Image in (readable)
        image_mem_obj_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        imageSize, image_in, &ret);
        // Image out (writable)
        image_mem_obj_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        imageSize, NULL, &ret);

        // Memoization
        if (Memoization::enabled) {
            // Color intensity
            intensity_mem_obj_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                            sizeof(Memoization::intensityBetweenColors), &Memoization::intensityBetweenColors, &ret);
            // Pixel distance
            distance_mem_obj_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                            (3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1) * sizeof(cl_double), &Memoization::distanceBetweenPixels[0], &ret);
        }

        // Program preparation
        program = clCreateProgramWithSource(context, 1, (const char **) &source_str, NULL, &ret);
                // Context, number of pointers to the code, pointers to the code,
                // strings are NULL terminated, error code
    
        // Compilation
        ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
                // Program, number of devices, device list (pointer), compilation options,
                // pointer to functions, user arguments

        // Log
        size_t build_log_len;
        char *build_log;
        ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
                // Program, device, output type, 
                // maximum string length, pointer to the string, actual string length
        build_log = (char *) malloc(sizeof(char) * (build_log_len + 1));
        ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
        if (strlen(build_log) > 1) {
            std::cout << build_log << std::endl;
        }
        free(build_log);

        // Kernel: object preparation
        kernel = clCreateKernel(program, kernelFunction.c_str(), &ret);
                // Program, kernel name, error code
    }

    void run(unsigned char *image_out, int width, int height) {
        int imageSize = height * width * 4 * sizeof(unsigned char);
        int w = (2 * (3 * Gauss::sigma_s) + 1) / 2;

        // Kernel: arguments
        ret |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &image_mem_obj_in);
        ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &image_mem_obj_out);
        ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &width);
        ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &height);
        ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *) &w);
        ret |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *) &Gauss::sigma_v);
        ret |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *) &Gauss::sigma_s);
                // Kernel, argument number, data length, pointer to data

        // If memoization is enabled, pass actual memory, null pointers otherwise
        if (Memoization::enabled) {
            ret |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &intensity_mem_obj_in);
            ret |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &distance_mem_obj_in);

            // Local GPU memory
            ret |= clSetKernelArg(kernel, 9, 256 * sizeof(cl_double), nullptr);
            // ret |= clSetKernelArg(kernel, 10, (3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1) * sizeof(double), nullptr);
        }

        // Kernel: run
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        // ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
                // Queue, kernel, dimensional (x, xy, xyz...), NULL required,
                // pointer to number of all threads, pointer to number of local threads, 
                // events that need to be completed before this command can be executed

        // Data copy
        ret = clEnqueueReadBuffer(command_queue, image_mem_obj_out, CL_TRUE, 0, imageSize, image_out, 0, NULL, NULL);
                // Read to memory from device, 0 = offset
                // last three - events that need to be completed before this command can be executed
    }

    void clean() {
        // Clean-up
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(image_mem_obj_in);
        ret = clReleaseMemObject(image_mem_obj_out);
        ret = clReleaseMemObject(intensity_mem_obj_in);
        ret = clReleaseMemObject(distance_mem_obj_in);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
    }
}

namespace Help {
    static void print() {
        std::cout << "Syntax: ./bilateral <image> <algorithm> [-h] [-o output_file] [-n threads/workgroups] [-s sigma s] [-v sigma v] [--no-memoize]" << std::endl << std::endl;
        std::cout << "Algorithms:" << std::endl;
        std::cout << "- seq - sequential (1 thread)" << std::endl;
        std::cout << "- openmp - openmp (n threads)" << std::endl;
        std::cout << "- opencl - opencl (n workgroups)" << std::endl << std::endl;
        std::cout << "Memoization is enabled by default and should be disabled if you don't have enough RAM." << std::endl;
        std::cout << "Memoization uses approximately 256 * 8 bytes + (3 * SIGMA_S + 1) * (3 * SIGMA_S + 1) * 8 bytes of memory." << std::endl;
        std::cout << "This means, by default, it will use approximately 2.640625 MiB of memory." << std::endl << std::endl;
        std::cout << "Memoization might yield unexpected results if graphics device runs out of local memory." << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::string inputFile;
    std::string outputFile;
    int numThreads = -1;
    enum { SEQUENTIAL, OPENMP, OPENCL } algorithm = SEQUENTIAL;

    size_t pArg = 1; // Positional arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            std::string nextArg;
            if (i < argc - 1) {
                nextArg = argv[++i];
            }
            switch (arg[1]) {
                case 'h':
                    Help::print();
                    return 0;
                case 'o':
                    outputFile = nextArg;
                    if (outputFile.substr(outputFile.find_last_of(".") + 1) != "png") {
                        outputFile += ".png";
                    }
                    break;
                case 's':
                    Gauss::sigma_s = std::stoi(nextArg);
                    if (Gauss::sigma_s <= 0) {
                        std::cout << "Error: Sigma S cannot be equal or less than zero." << std::endl;
                        return 1;
                    }
                    break;
                case 'v':
                    Gauss::sigma_v = std::stoi(nextArg);
                    if (Gauss::sigma_v <= 0) {
                        std::cout << "Error: Sigma V cannot be equal or less than zero." << std::endl;
                        return 1;
                    }
                    break;
                case 'n':
                    numThreads = std::stoi(nextArg);
                    if (numThreads <= 0) {
                        std::cout << "Error: Number of threads cannot be equal or less than zero." << std::endl;
                        return 1;
                    }
                    break;
                case '-':
                    if (arg.compare("--no-memoize") == 0) {
                        Memoization::enabled = false;
                    }
                    break;
                default:
                    std::cout << "Error: Invalid flag passed: " << arg.c_str() << std::endl;
                    return 1;
            }
        } else {
            switch (pArg++) { // Parse positional arguments
                case 1: // Input file
                    inputFile = arg;
                    if (outputFile.empty()) {
                        outputFile = inputFile.substr(0, inputFile.find_last_of(".")) + "-bilateral.png"; // In case -o wasn't passed
                    }
                    break;
                case 2: // Algorithm
                    if (arg.compare("seq") == 0) {
                        algorithm = SEQUENTIAL;
                    } else if (arg.compare("openmp") == 0) {
                        algorithm = OPENMP;
                    } else if (arg.compare("opencl") == 0) {
                        algorithm = OPENCL;
                    } else {
                        // Error maybe? It defaults to SEQUENTIAL anyways
                        std::cout << "Error: Invalid algorithm passed: " << arg << std::endl;
                        return 1;
                    }
                    break;
            }
        }
    }

    if (pArg < 3) {
        std::cout << "Error: Invalid arguments passed!" << std::endl;
        Help::print();
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Input file: " << inputFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << "Algorithm: " << algorithm << std::endl;
    std::cout << "Sigma V: " << Gauss::sigma_v << std::endl;
    std::cout << "Sigma S: " << Gauss::sigma_s << std::endl;
    std::cout << std::endl;

    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, inputFile.c_str(), 0);
    if (!imageBitmap) {
        std::cout << "Error: Image does not exist" << std::endl;
        return 1;
    }
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    int width = FreeImage_GetWidth(imageBitmap32);
    int height = FreeImage_GetHeight(imageBitmap32);
    int pitch = FreeImage_GetPitch(imageBitmap32);

    unsigned char *image_in = (unsigned char *) malloc(height * pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(image_in, imageBitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    FreeImage_Unload(imageBitmap32);
    FreeImage_Unload(imageBitmap);

    unsigned char *image_out = (unsigned char *) malloc(height * pitch * sizeof(unsigned char));

    switch (algorithm) {
        double start, end;
        double memoizationTime;
        case SEQUENTIAL:
            std::cout << "Bilateral filter - sequential" << std::endl;

            start = omp_get_wtime();

            if (Memoization::enabled) {
                std::cout << "Populating memoize arrays..." << std::endl;
                for (int i = 0; i < 256; i++) {
                    Memoization::intensityBetweenColors[i] = Gauss::intensity(i, true);
                }
                Memoization::distanceBetweenPixels = static_cast<double*>(std::calloc((3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1), sizeof(double)));
            }

            BilateralCPU::sequential(image_in, image_out, width, height);

            end = omp_get_wtime();
            printf("Image size %dx%d, CPU (sequential) time: %.6fs\n", width, height, end - start);
            break;
        case OPENMP:
            // Thread usage isn't necessarily guaranteed
            std::cout << "Using (max) " << (numThreads != -1 ? numThreads : omp_get_max_threads()) << " threads." << std::endl;

            std::cout << "Bilateral filter - OpenMP" << std::endl;

            start = omp_get_wtime();

            if (Memoization::enabled) {
                std::cout << "Populating memoize arrays..." << std::endl;
                #pragma omp parallel for
                for (int i = 0; i < 256; i++) {
                    Memoization::intensityBetweenColors[i] = Gauss::intensity(i, true);
                }
                Memoization::distanceBetweenPixels = static_cast<double*>(std::calloc((3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1), sizeof(double)));
            }

            BilateralCPU::openmp(image_in, image_out, width, height, numThreads);

            end = omp_get_wtime();
            printf("Image size %dx%d, CPU (OpenMP) time: %.6fs\n", width, height, end - start);
            break;
        case OPENCL:
            if (numThreads == -1) {
                numThreads = 256;
            }
            std::cout << "Using " << numThreads << " workgroups." << std::endl;

            std::cout << "Bilateral filter - OpenCL" << std::endl;

            // Initialize memoization arrays before setup
            start = omp_get_wtime();

            if (Memoization::enabled) {
                std::cout << "Populating memoize arrays..." << std::endl;
                #pragma omp parallel for
                for (int i = 0; i < 256; i++) {
                    Memoization::intensityBetweenColors[i] = Gauss::intensity(i, true);
                }
                // Memoization::distanceBetweenPixels = std::vector<double>((3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1), -1);
                Memoization::distanceBetweenPixels = static_cast<double *>(std::calloc((3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1), sizeof(double)));

                const int w = ((2 * (3 * Gauss::sigma_s) + 1) / 2);
                #pragma omp parallel for
                for (int i = 0; i < (3 * Gauss::sigma_s + 1) * (3 * Gauss::sigma_s + 1); i++) {
                    int x = i % w;
                    int y = i / w;
                    Memoization::distanceBetweenPixels[i] = Gauss::distance(x, y, w, true);
                }
            }

            end = omp_get_wtime();
            memoizationTime = end - start;

            // We won't count set-up
            std::cout << "Compiling and setting up kernel..." << std::endl;
            BilateralGPU::setup(Memoization::enabled ? "bilateral-memoize.cl" : "bilateral.cl", "bilateral", image_in, width, height, numThreads);
            std::cout << "Running bilateral filtering..." << std::endl;

            start = omp_get_wtime();

            BilateralGPU::run(image_out, width, height);

            end = omp_get_wtime();

            // We won't count clean-up
            BilateralGPU::clean();
            printf("Velikost slike %dx%d, GPU (paralelni) cas: %.6fs\n", width, height, memoizationTime + (end - start));
            break;
        default:
            std::cout << "Bilateral filter - unimplemented" << std::endl;
            break;
    }

    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image_out, width, height, pitch,
        32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
    FreeImage_Save(FIF_PNG, dst, outputFile.c_str(), 0);

    free(Memoization::distanceBetweenPixels);

    return 0;
}
