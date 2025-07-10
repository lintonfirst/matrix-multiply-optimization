#include "common/matrix.h"

int count=1;
void initializeMatrix(Matrix* matrix,int dim) {
    matrix->width = dim;
    matrix->height = dim;
    size_t size = dim * dim * sizeof(float);
    cudaMalloc(&matrix->elements, size);
    float* hostElements = (float*)malloc(size);
    for (int i = 0; i < dim * dim; i++) {
        hostElements[i] = count+i%17;
    }
    cudaMemcpy(matrix->elements, hostElements, size, cudaMemcpyHostToDevice);
    free(hostElements);
    count++;
}

void initializeMatrix_zero(Matrix* matrix,int dim){
    matrix->width = dim;
    matrix->height = dim;
    size_t size = dim * dim * sizeof(float);
    cudaMalloc(&matrix->elements, size);
    cudaMemset(matrix->elements,0,size);
}

void freeMatrix(Matrix* matrix) {
    if (matrix->elements != nullptr) {
        cudaFree(matrix->elements);
        matrix->elements = nullptr;
    }
    matrix->width = 0;
    matrix->height = 0;
}