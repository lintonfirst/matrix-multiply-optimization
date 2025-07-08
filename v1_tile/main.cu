#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>
#include "common/matrix.h"
#include "common/eval.h"
#include "common/check.h"
#include <iostream>

#define TILE_SIZE 16

__global__ void matmul(Matrix A, Matrix B, Matrix C) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    float* matA = A.elements;
    float* matB = B.elements;
    float* matC = C.elements;

    int tile_id_row = blockIdx.y;
    int tile_id_col = blockIdx.x;
    int tile_offset_row = threadIdx.y;
    int tile_offset_col = threadIdx.x;
    float sum = 0.0f;
    for(int i=0;i<gridDim.x;i++){
        tile_A[tile_offset_row][tile_offset_col] = matA[(i*TILE_SIZE+tile_offset_col)+A.width*(tile_id_row*TILE_SIZE+tile_offset_row)];
        tile_B[tile_offset_row][tile_offset_col] = matB[(tile_id_col*TILE_SIZE+tile_offset_col)+B.width*(i*TILE_SIZE+tile_offset_row)];
        __syncthreads();
        for(int j=0;j<TILE_SIZE;j++) {
            sum += tile_A[tile_offset_row][j] * tile_B[j][tile_offset_col];
        }
        __syncthreads();
    }
    matC[(tile_id_row*TILE_SIZE+tile_offset_row)*C.width + (tile_id_col*TILE_SIZE+tile_offset_col)] = sum;
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
        cudaMemset(C.elements,0,dim*dim*sizeof(float));
        dim3 grid(dim/TILE_SIZE, dim/TILE_SIZE);
        dim3 block(TILE_SIZE, TILE_SIZE);
        matmul<<<grid,block>>>(A,B,C);
    });
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"isCalculationRight:"<<check(A,B,C)<<std::endl;
}