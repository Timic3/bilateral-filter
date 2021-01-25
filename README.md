# Bilateral filter
> Project for Parallel and Distributed Systems and Algorithms class

## Requirements
- OpenCL
- LibFreeImage

## Build
1. Run `module load CUDA` in order for OpenCL to work
2. Run `g++ bilateral.cpp -O3 -Wall -Wextra -pedantic -lm -lOpenCL -fopenmp -std=c++17 -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o bilateral` to compile the program
3. Run `./bilateral -h` for help

**This repository is still in progress, better build process and documentation is coming  soon.**