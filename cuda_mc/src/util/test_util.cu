/*
 * test_util.cu
 *
 *  Created on: 22-Mar-2009
 *      Author: alee
 */

#include <stdio.h>
#include "gauss.h"
#include <cutil.h>
#include "reduce.h"
#include "scan.h"
#include "mix_gauss.h"
#include "mix_gauss_uniform.h"
#include "MRG.h"
#include "xorshift.h"
#include "rng.h"
#include "output.h"
#include "matrix.h"

__global__ void logtest(int size, float* d_array, int M) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i, j;
    float x;
    for (i = tid; i < size; i += tt) {
        x = i;
        for (j = 0; j < M; j++) {
            x = logf(x);
            x = expf(x);
        }
        d_array[i] = x;
    }
}

void logtestref(int size, float* array, int M) {
    float x;
    for (int i = 0; i < size; i++) {
        x = (float) i;
        for (int j = 0; j < M; j++) {
            x = logf(x);
            x = expf(x);
        }
        array[i] = x;
    }
}

void testLogSpeed(int N, int M) {
    unsigned int hTimer;
    double gtime, ctime;
    cutCreateTimer(&hTimer);
    float* array = (float*) malloc(N * sizeof(float));
    float* d_array;
    cudaMalloc((void**) &d_array, N * sizeof(float));

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    logtest<<<256,64>>>(N, d_array, M);
    cudaThreadSynchronize();
    cutStopTimer(hTimer);
    gtime = cutGetTimerValue(hTimer);
    printf("log test time = %f\n", gtime);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    logtestref(N, array, M);
    cutStopTimer(hTimer);
    ctime = cutGetTimerValue(hTimer);
    printf("ref log test time = %f\n", ctime);

    //    float* h_array = (float*) malloc(N * sizeof(float));
    //    cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < 200; i++) {
    //        printf("%f %f\n", h_array[i], array[i]);
    //    }
    //    free(h_array);

    printf("speedup = %f\n", ctime / gtime);

    free(array);
    cudaFree(d_array);
}

void test_reduce2D(int N, int nb, int nt) {
    const int D = 2;
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* array = (float*) malloc(N * D * sizeof(float));
    float* d_array;
    cudaMalloc((void **) &d_array, N * D * sizeof(float));

    //	populate_rand(array, N * D);

    float c = 1;
    for (int i = 0; i < N * D; i++) {
        if (i % D == 1) {
            c = 1;
        } else {
            c = .5;
        }
        array[i] = c;
    }

    cudaMemcpy(d_array, array, sizeof(float) * N * D, cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    double sumh1 = 0;
    double sumh2 = 0;

    for (int i = 0; i < N; i++) {
        sumh1 += array[i * 2];
        sumh2 += array[i * 2 + 1];
    }

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Host reduce (2D): sum1 = %f, sum2 = %f\n", sumh1, sumh2);
    printf("Time = %f\n\n", time);

    float sum[2];

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    reduce(N, D, d_array, sum, nb, nt);
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("reduce_2D: sum1 = %f, sum2 = %f\n", sum[0], sum[1]);
    printf("Time = %f\n\n", time);

    free(array);
    cudaFree(d_array);
}

void test_reduceMD(int N, int D, int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* array = (float*) malloc(N * D * sizeof(float));
    float* d_array;
    cudaMalloc((void **) &d_array, N * D * sizeof(float));

    //	populate_rand_XS(array, N * D);

    float c = 1;
    for (int i = 0; i < N * D; i++) {
        c = (float) (i % D) + 1;
        array[i] = c;
    }

    cudaMemcpy(d_array, array, sizeof(float) * N * D, cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    //	double* sumh = (double*) malloc(D * sizeof(double));
    float* sumh = (float*) malloc(D * sizeof(float));
    for (int j = 0; j < D; j++) {
        sumh[j] = 0;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            sumh[j] += array[i * D + j];
        }
    }

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Host reduce (MD): (");
    for (int i = 0; i < D; i++) {
        printf("%f,", sumh[i]);
    }
    printf(")\n");
    printf("Time = %f\n\n", time);

    float* sum = (float*) malloc(D * sizeof(float));

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    reduce(N, D, d_array, sum, nb, nt);
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Reduce (MD): (");
    for (int i = 0; i < D; i++) {
        printf("%f,", sum[i]);
    }
    printf(")\n");
    printf("Time = %f\n\n", time);

    free(array);
    cudaFree(d_array);
    free(sum);
    free(sumh);
}

void test_float() {
    float f = (float) 67108868.0f;
    double d = (double) 67108868;
    //	float f = 0;
    //	double d = 0;
    //	for (int i = 0; i < 20000000; i++) {
    //		f = f + 4.0;
    //		d = d + 4.0;
    //	}
    printf("float  = %f\n", f);
    printf("double = %f\n", d);
    //	printf("%f\n", 45000000.0f * 4.0f);
}

void test_reduce(int N, int nb, int nt) {
    //	const int N = 4194304;
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* array = (float*) malloc(N * sizeof(float));
    float* d_array;
    cudaMalloc((void **) &d_array, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        array[i] = 0.5;
    }

    cudaMemcpy(d_array, array, sizeof(float) * N, cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    float sum = 0;
    double sumh = 0;

    for (int i = 0; i < N; i++) {
        sumh += array[i];
    }

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Host reduce: sum = %f\n", sumh);
    printf("Time = %f\n\n", time);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    reduce(N, d_array, sum, nb, nt);
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("reduce: sum = %f\n", sum);
    printf("Time = %f\n\n", time);

    free(array);
    cudaFree(d_array);

}

void test_matrix() {
    const int MA = 3;
    const int NA = 4;
    float A[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    const int MB = 4;
    const int NB = 3;
    float B[12] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };

    const int MC = MA;
    const int NC = NB;

    float C[MC * NC];

    matrix_times(A, B, C, MA, NA, MB, NB);

    matrix_print(C, MC, NC);

    int ME = 3;
    int NE = 3;

    float E[9] = { 1, 2, 2, 5, 7, 3, 4, 8, 7 };
    float EI[9];
    printf("1\n");
    matrix_inverse(E, EI, ME);
    printf("2\n");
    matrix_print(EI, ME, NE);
    printf("3\n");

    printf("|E| = %f\n", matrix_det(E, ME));
    printf("|EI| = %f\n", matrix_det(EI, ME));

    int MF = 4;
    int NF = 4;
    float F[16] = { 1, 8, 6, 3, 2, 7, 4, 0, -5, 4, -8, 3, 2, -6, 9, 3 };
    float FI[16];
    matrix_inverse(F, FI, MF);
    matrix_print(FI, MF, NF);

    printf("|F| = %f\n", matrix_det(F, MF));
    printf("|FI| = %f\n", matrix_det(FI, MF));

    float x[4] = { 3, 5, 2, 7 };
    float r = matrix_xtmx(F, x, 4);
    printf("r = %f\n", r);

}

void test_gaussmv() {
    const int D = 3;
    float x[3] = { 0.5, 1, 2 };
    float mu[3] = { 2, 1, 4 };
    float cov[9] = { 3, 0.5, 3, 0.5, 2, 0, 3, 0, 4 };

    float c2[9];
    float c1;

    compute_c1_c2(cov, D, c1, c2);

    float* h_args = (float*) malloc((1 + D + D * D) * sizeof(float));
    h_args[0] = c1;
    h_args[1] = c2[0];
    h_args[2] = c2[1];
    h_args[3] = c2[2];
    h_args[4] = c2[3];
    h_args[5] = c2[4];
    h_args[6] = c2[5];
    h_args[7] = c2[6];
    h_args[8] = c2[7];
    h_args[9] = c2[8];
    h_args[10] = mu[0];
    h_args[11] = mu[1];
    h_args[12] = mu[2];

    float r = gauss_pdfh(x, c1, c2, mu, D);

    printf("r = %f\n", r);

    r = gauss_pdfh(x, h_args, D);

    printf("r = %f\n", r);

    free(h_args);
}

void test_scan(int N) {
    unsigned int hTimer;
    double ctime, gtime;
    cutCreateTimer(&hTimer);

    float* array = (float*) malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        array[i] = 0.5;
    }
    float* array_out = (float*) malloc(N * sizeof(float));

    float* d_array;
    cudaMalloc((void **) &d_array, N * sizeof(float));
    float* d_array_out;
    cudaMalloc((void **) &d_array_out, N * sizeof(float));

    cudaMemcpy(d_array, array, sizeof(float) * N, cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    scan(N, d_array, d_array_out, 32, 32);
    cutStopTimer(hTimer);
    gtime = cutGetTimerValue(hTimer);
    printf("GPU Time = %f\n\n", gtime);

    cudaThreadSynchronize();

    cudaMemcpy(array_out, d_array_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    //	for (int i = 0; i < N; i++) {
    //		printf("%f ", array_out[i]);
    //	}
    //	printf("\n");

    float* array_out_ref = (float*) malloc(N * sizeof(float));
    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    cudaMemcpy(array, d_array, sizeof(float) * N, cudaMemcpyDeviceToHost);
    scan_ref(N, array, array_out_ref);
    cudaMemcpy(d_array_out, array_out_ref, sizeof(float) * N, cudaMemcpyHostToDevice);
    cutStopTimer(hTimer);
    ctime = cutGetTimerValue(hTimer);
    printf("CPU Time = %f\n", ctime);

    printf("speedup = %f\n", ctime/gtime);

    for (int i = 0; i < N; i++) {
        if (array_out[i] != array_out_ref[i]) {
            printf("FAIL: %d, %f, %f\n", i, array_out[i], array_out_ref[i]);
        }
    }

    free(array);
    free(array_out);
    free(array_out_ref);
    cudaFree(d_array);
    cudaFree(d_array_out);
}

void test_pdfh() {
    seed_rng();

    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    const int k = 4;
    float mus[4] = { -3, 0, 3, 6 };
    float sigma = 0.55f;

    const int N = 100;

    float data_array[N];
    generate_mix_data(k, sigma, mus, data_array, N);

    float guess[4] = { -6, 3, 0, -3 };

    float c1, c2;
    compute_ci1_ci2(sigma, 1.0f / k, c1, c2);

    float r = 0;
    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    for (int i = 0; i < 32768; i++) {
        r = log_mgu_pdfh(data_array, N, guess, k, c1, c2, -10.0f, 10.0f);
    }
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("ref pdfh time = %f\n", time);

    printf("r = %f\n", r);
}

void test_pdfh2() {
    const int k = 4;

    float sigma = 0.55f;

    const int N = 2;
    float data_array[N] = { 0, 0 };
    //  generate_mix_data(k, sigma, mus, data_array, N);

    float guess[k] = { 10, -12, 13, 11 };

    float c1, c2;
    compute_ci1_ci2(sigma, 1.0f / k, c1, c2);

    float r = log_mgu_pdfh(data_array, N, guess, k, c1, c2, -10.0f, 10.0f);
    //  float r2 = log_mgu_pdf(data_array, N, guess, k, c1, c2);

    printf("r = %f\n", r);
}

//void test_MRGseed() {
//    unsigned int hTimer;
//    double time;
//    cutCreateTimer(&hTimer);
//
//    unsigned long seeds[6] = { 2, 1, 2, 3, 4, 5 };
//
//    seed_MRG32k3a(32, 128, seeds);
//
//    int N = 16777216;
//    unsigned int* array1 = (unsigned int*) malloc(N * sizeof(unsigned int));
//    unsigned int* array2 = (unsigned int*) malloc(N * sizeof(unsigned int));
//
//    cutResetTimer(hTimer);
//    cutStartTimer(hTimer);
//    populate_randUI_MRG32k3a(array1, N);
//    cutStopTimer(hTimer);
//    time = cutGetTimerValue(hTimer);
//    printf("no skip time = %f\n", time);
//
//    seed_MRG32k3a(32, 128, seeds);
//
//    cutResetTimer(hTimer);
//    cutStartTimer(hTimer);
//    populate_randUI_MRG32k3a_skip(array2, N);
//    cutStopTimer(hTimer);
//    time = cutGetTimerValue(hTimer);
//    printf("skip time = %f\n", time);
//
//    int k = 4096;
//    for (int i = 0; i < 200; i++) {
//        printf("%d: %u ", i, array1[k - 1 + i * k]);
//        printf("%u\n", array2[i]);
//    }
//
//    free(array1);
//    free(array2);
//
//}

void test_MRG() {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    unsigned long seeds[6] = { 12345UL, 12345UL, 12345UL, 12345UL, 12345UL, 12345UL };

    int nb = 128;
    int nt = 64;
    int tt = nb * nt;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    seed_MRG(nb, nt, seeds);
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("seed time = %f\n", time);

    //    int N = 16777216;
    //        int N = 1048576;
    //        int N = 2097152;
    int N = 131072;
    //    int N = 65536;
    //        int N = 32768;
    //    int N = 8;
    float* array1 = (float*) malloc(N * sizeof(float));
    float* array2 = (float*) malloc(N * sizeof(float));
    float* d_array2;
    cudaMalloc((void**) &d_array2, N * sizeof(float));

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    for (int i = 0; i < 65; i++) {
        populate_rand_MRG_REF(array1, N);
    }
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("ref time = %f\n", time);

    kill_MRG();
    seed_MRG(nb, nt, seeds);

    //    seedXS();

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    populate_rand_MRG_d(d_array2, N);
    //    populate_rand_XS_d(d_array2,N);
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("gpu time = %f\n", time);

    cudaMemcpy(array2, d_array2, N * sizeof(float), cudaMemcpyDeviceToHost);

    //    int k = 4096;
    for (int i = 0; i < 10; i++) {
        //        printf("%d: %f ", i, array1[1048576 + i]);
        //        printf("%f\n", array2[tt*i]);
        printf("%d: %f ", i, array1[i]);
        printf("%f\n", array2[tt * i + 1]);
    }

    cudaFree(d_array2);
    free(array1);
    free(array2);

}

void compilerSmarts();

void test_rng(int N) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);
    seed_rng();
    float* d_array;
    cudaMalloc((void**) &d_array, N * sizeof(float));
    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    populate_rand_d(d_array, N);
    //    compilerSmarts();
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("time = %f\n", time);

    float* array = (float*) malloc(N * sizeof(float));
    cudaMemcpy(array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += array[i] - 0.5f;
    }
    printf("%f\n", sum);

    cudaFree(d_array);
    free(array);
    kill_rng();
}

void test_rng2(int N) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);
    seed_rng(16777216, 32, 128);
    unsigned int* d_array;
    cudaMalloc((void**) &d_array, N * sizeof(unsigned int));
    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    populate_randUI_d(d_array, N);
    populate_randUI_d(d_array, N);
    //    compilerSmarts();
    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("time = %f\n", time);

    unsigned int* array = (unsigned int*) malloc(N * sizeof(unsigned int));
    cudaMemcpy(array, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    printf("%d\n", sum);

    to_file(array, N - 4, "rngdata.txt");

    cudaFree(d_array);
    free(array);
    kill_rng();
}

void compilerSmarts() {
    unsigned long x;
    unsigned long z = 4294967087UL;
    for (int i = 0; i < 1000000; i++) {
        x = 32;
        if (x > z) {
            x = x % z;
        }
    }
    printf("%ld\n", x);
}

void test_matrix2() {
    //    float E[9] = { 0.9673, 0.4522, 0.8797, 0.4522, 0.2890, 0.4882, 0.8797, 0.4882, 1.3795 };
    //    float EI[9];
    const int D = 5;
    float E[D * D] = { 2.1487f, 0.8244f, 0.8244f, 0.3297f, 1.3190f, 0.8244f, 3.3718f, 1.6420f,
            1.6406f, 2.3812f, 0.8244f, 1.6420f, 2.7485f, 1.2692f, 2.1311f, 0.3297f, 1.6406f,
            1.2692f, 1.5613f, 1.4800f, 1.3190f, 2.3812f, 2.1311f, 1.4800f, 3.0657f };
    float EI[D * D];
    //    unsigned int hTimer;
    //    double time;
    //    cutCreateTimer(&hTimer);

    float d;

    printf("%.10f\n", matrix_det(E, D));
    //    printf("%.10f\n", matrix_det2(E, D));
    //    printf("%.10f\n", matrix_det3(E, D));
    //    printf("%.10f\n", matrix_det4(E, D));

    matrix_det_inv_pd(E, d, EI, D);
    //    printf("%.10f\n", d);


    //    cutResetTimer(hTimer);
    //    cutStartTimer(hTimer);
    //    for (int i = 0; i < 100000; i++) {
    //        matrix_inverse(E, EI, 3);
    //    }
    //    cutStopTimer(hTimer);
    //    time = cutGetTimerValue(hTimer);
    //    printf("time = %f\n", time);
    //    matrix_print(EI, 3, 3);
    //
    //    cutResetTimer(hTimer);
    //    cutStartTimer(hTimer);
    //    for (int i = 0; i < 100000; i++) {
    //        matrix_inverse_pd(E, EI, 3);
    //    }
    //    cutStopTimer(hTimer);
    //    time = cutGetTimerValue(hTimer);
    //    printf("time = %f\n", time);
    //    matrix_print(EI, 3, 3);

}

void test_burn() {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    int N = 1073741824;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    seed_XS_REF(N);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("time = %f\n", time);
}

int main(int argc, char **argv) {
    //    test_MRGseed();
    //    test_pdfh();
    //    test_MRG();
    //    int N = 65536;
    //    int N = 131072;
    //    int M = 512;
    //    testLogSpeed(N, M);
    //    test_rng(16777216);
    //    test_rng2(1048576);
    //    test_matrix2();
    //    test_burn();


    //    const int N = 4096;
    //    const int N = 65536;
    const int N = 65536;
    scan_init(N);

    test_scan(N);

    scan_destroy();
}
