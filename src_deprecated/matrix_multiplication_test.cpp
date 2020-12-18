//
// Created by b1xian on 2020-10-12.
//
#include <arm_neon.h>
#include <iostream>
#include <time.h>


void matrix_multiply_c(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx=0; i_idx < n; i_idx++) {
        for (int j_idx=0; j_idx < m; j_idx++) {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx < k; k_idx++) {
                C[n*j_idx + i_idx] += A[n*k_idx + i_idx]*B[k*j_idx + k_idx];
            }
        }
    }
}

void matrix_multiply_4x4_neon(float32_t *A, float32_t *B, float32_t *C) {
    // these are the columns A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    // these are the columns C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    A0 = vld1q_f32(A);
    A1 = vld1q_f32(A+4);
    A2 = vld1q_f32(A+8);
    A3 = vld1q_f32(A+12);

    // Zero accumulators for C values
    C0 = vmovq_n_f32(0);
    C1 = vmovq_n_f32(0);
    C2 = vmovq_n_f32(0);
    C3 = vmovq_n_f32(0);

    // Multiply accumulate in 4x1 blocks, i.e. each column in C
    B0 = vld1q_f32(B);
    C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
    C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
    C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
    C0 = vfmaq_laneq_f32(C0, A3, B0, 3);
    vst1q_f32(C, C0);

    B1 = vld1q_f32(B+4);
    C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
    C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
    C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
    C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
    vst1q_f32(C+4, C1);

    B2 = vld1q_f32(B+8);
    C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
    C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
    C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
    C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
    vst1q_f32(C+8, C2);

    B3 = vld1q_f32(B+12);
    C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
    C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
    C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
    C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
    vst1q_f32(C+12, C3);
}

void matrix_multiply_neon(float32_t  *A, float32_t  *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    /*
     * Multiply matrices A and B, store the result in C.
     * It is the user's responsibility to make sure the matrices are compatible.
     */

    int A_idx;
    int B_idx;
    int C_idx;

    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4){
            // zero accumulators before matrix op
            C0=vmovq_n_f32(0);
            C1=vmovq_n_f32(0);
            C2=vmovq_n_f32(0);
            C3=vmovq_n_f32(0);
            for (int k_idx=0; k_idx<k; k_idx+=4){
                // compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // load most current a values in row
                A0=vld1q_f32(A+A_idx);
                A1=vld1q_f32(A+A_idx+n);
                A2=vld1q_f32(A+A_idx+2*n);
                A3=vld1q_f32(A+A_idx+3*n);

                // multiply accumulate 4x1 blocks, i.e. each column C
                B0=vld1q_f32(B+B_idx);
                C0=vfmaq_laneq_f32(C0,A0,B0,0);
                C0=vfmaq_laneq_f32(C0,A1,B0,1);
                C0=vfmaq_laneq_f32(C0,A2,B0,2);
                C0=vfmaq_laneq_f32(C0,A3,B0,3);

                B1=vld1q_f32(B+B_idx+k);
                C1=vfmaq_laneq_f32(C1,A0,B1,0);
                C1=vfmaq_laneq_f32(C1,A1,B1,1);
                C1=vfmaq_laneq_f32(C1,A2,B1,2);
                C1=vfmaq_laneq_f32(C1,A3,B1,3);

                B2=vld1q_f32(B+B_idx+2*k);
                C2=vfmaq_laneq_f32(C2,A0,B2,0);
                C2=vfmaq_laneq_f32(C2,A1,B2,1);
                C2=vfmaq_laneq_f32(C2,A2,B2,2);
                C2=vfmaq_laneq_f32(C2,A3,B2,3);

                B3=vld1q_f32(B+B_idx+3*k);
                C3=vfmaq_laneq_f32(C3,A0,B3,0);
                C3=vfmaq_laneq_f32(C3,A1,B3,1);
                C3=vfmaq_laneq_f32(C3,A2,B3,2);
                C3=vfmaq_laneq_f32(C3,A3,B3,3);
            }
            //Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx, C0);
            vst1q_f32(C+C_idx+n,C1);
            vst1q_f32(C+C_idx+2*n,C2);
            vst1q_f32(C+C_idx+3*n,C3);
        }
    }
}

int main() {

    clock_t start_time = clock();
    uint32_t n = 100000000;
    uint32_t m = 100000000;
    uint32_t k = 100000000;
    float32_t* A;
    float32_t* B;
    float32_t* C;
    A = (float32_t *)malloc(sizeof(float32_t) * n*k);
    B = (float32_t *)malloc(sizeof(float32_t) * k*m);
    C = (float32_t *)malloc(sizeof(float32_t) * n*m);

    for (int i = 0; i < n*k; i++) {
        A[i] = 0.32718543526f;
        B[i] = 0.87946738931f;
    }

    clock_t end_time = clock();
    std::cout << "init: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "ms" << std::endl;


    start_time = clock();
    matrix_multiply_c(A, B, C, n, m, k);
    end_time = clock();
    std::cout << "none_neon_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "ms" << std::endl;

    start_time = clock();
    matrix_multiply_neon(A, B, C, n, m, k);
    end_time = clock();
    std::cout << "neon_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "ms" << std::endl;

//        matrix_multiply_4x4_neon(A, B ,C);
    return 0;
}