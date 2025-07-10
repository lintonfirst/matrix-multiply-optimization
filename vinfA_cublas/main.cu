#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>
#include "common/matrix.h"
#include "common/eval.h"
#include "common/check.h"
#include <iostream>
#include <cublas_v2.h>

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

    cublasHandle_t handle;
    cublasCreate(&handle);
    float duration=eval.eval([&]() {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim, dim, dim,
            &alpha,
            B.elements, dim,
            A.elements, dim,
            &beta,
            C.elements, dim
        );
    });
    cublasDestroy(handle);
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"isCalculationRight:"<<check(A,B,C)<<std::endl;

    freeMatrix(&A);
    freeMatrix(&B);
    freeMatrix(&C);
}