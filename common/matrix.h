#pragma once
#include <cuda_runtime.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

void initializeMatrix(Matrix* matrix,int dim) {
    matrix->width = dim;
    matrix->height = dim;
    size_t size = dim * dim * sizeof(float);
    cudaMalloc(&matrix->elements, size);
    float* hostElements = (float*)malloc(size);
    for (int i = 0; i < dim * dim; i++) {
        hostElements[i] = static_cast<float>(i);
    }
    cudaMemcpy(matrix->elements, hostElements, size, cudaMemcpyHostToDevice);
    free(hostElements);
}

void freeMatrix(Matrix* matrix) {
    if (matrix->elements != nullptr) {
        cudaFree(matrix->elements);
        matrix->elements = nullptr;
    }
    matrix->width = 0;
    matrix->height = 0;
}