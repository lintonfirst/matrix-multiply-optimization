#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>
#include "common/matrix.h"
#include "common/eval.h"
#include "common/check.h"
#include <iostream>
#include "cutlass/gemm/device/gemm.h"


using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix
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
        CutlassGemm::Arguments args({dim , dim, dim},  // Gemm Problem dimensions
                              {A.elements, dim},    // Tensor-ref for source matrix A
                              {B.elements, dim},    // Tensor-ref for source matrix B
                              {C.elements, dim},    // Tensor-ref for source matrix C
                              {C.elements, dim},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {1, 0});
        CutlassGemm gemm_operator;
        cutlass::Status status = gemm_operator(args);
    });
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
}