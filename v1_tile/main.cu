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
    __shared__ float tile_res[TILE_SIZE][TILE_SIZE];

    float* matA = A.elements;
    float* matB = B.elements;
    float* matC = C.elements;

    int tile_id_x = blockIdx.x;
    int tile_id_y = blockIdx.y;
    int tile_offset_x = threadIdx.x;
    int tile_offset_y = threadIdx.y;
    tile_res[tile_offset_x][tile_offset_y] = 0.0f;
    for(int i=0;i<blockDim.x;i++){
        tile_A[tile_offset_x][tile_offset_y] = matA[(i*TILE_SIZE+tile_offset_x)+A.width*(tile_id_y*TILE_SIZE+tile_offset_y)];
        tile_B[tile_offset_x][tile_offset_y] = matB[(tile_id_x*TILE_SIZE+tile_offset_x)+B.width*(i*TILE_SIZE+tile_offset_y)];
        __syncthreads();
        float value = 0.0f;
        for(int i=0;i<TILE_SIZE;i++) {
            value += tile_A[tile_offset_x][i] * tile_B[i][tile_offset_y];
        }
        tile_res[tile_offset_x][tile_offset_y] += value;
        __syncthreads();
    }
    matC[(tile_id_y*TILE_SIZE+tile_offset_y)*C.width + (tile_id_x*TILE_SIZE+tile_offset_x)] = tile_res[tile_offset_x][tile_offset_y];
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
        dim3 grid(dim/TILE_SIZE, dim/TILE_SIZE);
        dim3 block(TILE_SIZE, TILE_SIZE);
        matmul<<<grid,block>>>(A,B,C);
    });
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
}