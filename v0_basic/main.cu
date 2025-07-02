#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>
#include "common/matrix.h"
#include "common/eval.h"
#include "common/check.h"
#include <iostream>


__global__ void matmul(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.height && col < C.width) {
        float value = 0.0f;
        for (int k = 0; k < A.width; ++k) {
            value += A.elements[row * A.width + k] * B.elements[k * B.width + col];
        }
        C.elements[row * C.width + col] = value;
    }
}


int main(int argc,char* argv[]) {
    int dim =1024;
    if(argc>1) {
        dim = atoi(argv[1]);
    }
    assert(dim%128==0);

    Matrix A,B,C;
    initializeMatrix(&A, dim);
    initializeMatrix(&B, dim);
    initializeMatrix(&C, dim);
    Eval eval;

    
    float duration=eval.eval([&]() {
        dim3 grid(dim / 16, dim / 16);
        dim3 block(16, 16);
        matmul<<<grid,block>>>(A, B, C);
    });
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
}