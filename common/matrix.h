#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

void initializeMatrix(Matrix* matrix,int dim);
void initializeMatrix_zero(Matrix* matrix,int dim);

void freeMatrix(Matrix* matrix);