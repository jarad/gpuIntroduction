/*
 * matrix.c
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include <stdio.h>
#include "matrix.h"
#ifdef MKL
#include "mkl_cblas.h"
#include "mkl_lapack.h"
#endif

//#undef CBLAS
//#undef LAPACK

#define CBLAS_LIMIT 1
#define SAFE

void matrix_print(float* matrix, int M, int N) {
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", matrix_access(matrix, M, N, i, j));
        }
        printf("\n");
    }
}

void matrix_print(double* matrix, int M, int N) {
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", matrix_access(matrix, M, N, i, j));
        }
        printf("\n");
    }
}

void matrix_print(unsigned int* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%u ", matrix_access(matrix, M, N, i, j));
        }
        printf("\n");
    }
}

void matrix_print(long int* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%ld ", matrix_access(matrix, M, N, i, j));
        }
        printf("\n");
    }
}

void matrix_print(unsigned long int* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lu ", matrix_access(matrix, M, N, i, j));
        }
        printf("\n");
    }
}

void matrix_print(int* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix_access(matrix, M, N, i, j));
        }
        printf("\n");
    }
}

template<class T>
inline void matrix_times_impl(T* matrix1, T* matrix2, T* matrix_out, int M1, int N1, int M2, int N2) {
    T* Bcolj = (T*) malloc(N1 * sizeof(T));
    for (int j = 0; j < N2; j++) {
        for (int k = 0; k < N1; k++) {
            Bcolj[k] = matrix_access(matrix2, M2, N2, k, j);
        }
        for (int i = 0; i < M1; i++) {
            T* Arowi = matrix1 + i * N1;
            T s = 0;
            for (int k = 0; k < N1; k++) {
                s += Arowi[k] * Bcolj[k];
            }
            matrix_set(matrix_out, M1, N2, i, j, s);
        }
    }
    free(Bcolj);
}

template<class T>
inline void matrix_XXt_impl(T* X, T* A, int M, int N) {
    T* r1;
    T* r2;
    T r;
    for (int i = 0; i < M; i++) {
        r1 = X + i * N;
        matrix_set(A, M, M, i, i, vector_xtx(r1, N));
        for (int j = 0; j < i; j++) {
            r2 = X + j * N;
            r = vector_xty(r1, r2, N);
            matrix_set(A, M, M, i, j, r);
        }
    }
}

template<class T>
inline void matrix_XtX_impl(T* X, T* A, int M, int N) {
    T* Xt = (T*) malloc(M * N * sizeof(T));
    matrix_transpose(X, Xt, M, N);
    matrix_XXt_impl(Xt, A, N, M);
    free(Xt);
}

template<class T>
inline int matrix_chol_impl(T* matrix, T* L, int D) {
    int isspd = 1;
    for (int j = 0; j < D; j++) {
        T* Lrowj = L + j * D;
        T d = 0;
        for (int k = 0; k < j; k++) {
            T* Lrowk = L + k * D;
            T s = 0;
            for (int i = 0; i < k; i++) {
                s += Lrowk[i] * Lrowj[i];
            }
            s = (matrix_access(matrix, D, D, j, k) - s) / matrix_access(L, D, D, k, k);
            Lrowj[k] = s;
            d += s * s;
            isspd = isspd
                    & (matrix_access(matrix, D, D, k, j) == matrix_access(matrix, D, D, j, k));
        }
        d = matrix_access(matrix, D, D, j, j) - d;
        isspd = isspd & (d > 0.0);
        matrix_set(L, D, D, j, j, (T) sqrt(max(d, (T) 0.0)));
        for (int k = j + 1; k < D; k++) {
            matrix_set(L, D, D, j, k, (T) 0.0);
        }
    }
    return isspd;
}

template<class T>
inline void matrix_solve_L_impl(T* L, T* b, T* x, int D) {

    for (int i = 0; i < D; i++) {
        x[i] = b[i];
    }

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < i; j++) {
            x[i] -= matrix_access(L, D, D, i, j) * x[j];
        }
        x[i] /= matrix_access(L, D, D, i, i);
    }
}

template<class T>
inline void matrix_solve_Lt_impl(T* L, T* b, T* x, int D) {

    for (int i = 0; i < D; i++) {
        x[i] = b[i];
    }

    for (int i = D - 1; i >= 0; i--) {
        for (int j = i + 1; j < D; j++) {
            x[i] -= matrix_access(L, D, D, j, i) * x[j];
        }
        x[i] /= matrix_access(L, D, D, i, i);
    }
}

template<class T>
inline void matrix_solve_pd_impl(T* L, T* b, T* x, int D) {
    matrix_solve_L(L, b, x, D);
    matrix_solve_Lt(L, x, x, D);
}

void matrix_times(float* matrix1, float* matrix2, float* matrix_out, int M1, int N1, int M2, int N2) {
#ifdef CBLAS
    if (M1 > CBLAS_LIMIT || N1 > CBLAS_LIMIT || N2 > CBLAS_LIMIT) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N2, M2, 1.0, matrix1, N1,
                matrix2, N2, 0.0, matrix_out, N2);
    } else {
        matrix_times_impl(matrix1, matrix2, matrix_out, M1, N1, M2, N2);
    }
#else
    matrix_times_impl(matrix1, matrix2, matrix_out, M1, N1, M2, N2);
#endif
}

void matrix_times(double* matrix1, double* matrix2, double* matrix_out, int M1, int N1, int M2,
        int N2) {
#ifdef CBLAS
    if (M1 > CBLAS_LIMIT || N1 > CBLAS_LIMIT || N2 > CBLAS_LIMIT) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N2, M2, 1.0, matrix1, N1,
                matrix2, N2, 0.0, matrix_out, N2);
    } else {
        matrix_times_impl(matrix1, matrix2, matrix_out, M1, N1, M2, N2);
    }
#else
    matrix_times_impl(matrix1, matrix2, matrix_out, M1, N1, M2, N2);
#endif
}

void matrix_XXt(float* X, float* A, int M, int N) {
#ifdef CBLAS
    if (M > CBLAS_LIMIT || N > CBLAS_LIMIT) {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, M, N, 1.0, X, N, 0.0, A, M);
    } else {
        matrix_XXt_impl(X, A, M, N);
    }
#else
    matrix_XXt_impl(X, A, M, N);
#endif
#ifdef SAFE
    matrix_fill_symm(A, M);
#endif
}

void matrix_XXt(double* X, double* A, int M, int N) {
#ifdef CBLAS
    if (M > CBLAS_LIMIT || N > CBLAS_LIMIT) {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, M, N, 1.0, X, N, 0.0, A, M);
    } else {
        matrix_XXt_impl(X, A, M, N);
    }
#else
    matrix_XXt_impl(X, A, M, N);
#endif
#ifdef SAFE
    matrix_fill_symm(A, M);
#endif
}

void matrix_XtX(float* X, float* A, int M, int N) {
#ifdef CBLAS
    if (M > CBLAS_LIMIT || N > CBLAS_LIMIT) {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasTrans, N, M, 1.0, X, N, 0.0, A, N);
    } else {
        matrix_XtX_impl(X, A, M, N);
    }
#else
    matrix_XtX_impl(X, A, M, N);
#endif
#ifdef SAFE
    matrix_fill_symm(A, N);
#endif
}

void matrix_XtX(double* X, double* A, int M, int N) {
#ifdef CBLAS
    if (M > CBLAS_LIMIT || N > CBLAS_LIMIT) {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, N, M, 1.0, X, N, 0.0, A, N);
    } else {
        matrix_XtX_impl(X, A, M, N);
    }
#else
    matrix_XtX_impl(X, A, M, N);
#endif
#ifdef SAFE
    matrix_fill_symm(A, N);
#endif
}

int matrix_chol(float* matrix, float* L, int D) {
#ifdef LAPACK
    if (D > CBLAS_LIMIT) {
        char uplo[1] = { 'U' }; // reverse bc of column-major LAPACK
        int info;
        matrix_copy_L(matrix, L, D);
        spotrf(uplo, &D, L, &D, &info);
        return info == 0;
    } else {
        return matrix_chol_impl(matrix, L, D);
    }
#else
    return matrix_chol_impl(matrix, L, D);
#endif
}

int matrix_chol(double* matrix, double* L, int D) {
#ifdef LAPACK
    if (D > CBLAS_LIMIT) {
        char uplo[1] = { 'U' }; // reverse bc of column-major LAPACK
        int info;
        matrix_copy_L(matrix, L, D);
        dpotrf(uplo, &D, L, &D, &info);
        return info == 0;
    } else {
        return matrix_chol_impl(matrix, L, D);
    }
#else
    return matrix_chol_impl(matrix, L, D);
#endif
}

void matrix_solve_L(float* L, float* b, float* x, int D) {
#ifdef CBLAS
    if (D > CBLAS_LIMIT) {
        vector_copy(b, x, D);
        cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, D, L, D, x, 1);
    } else {
        matrix_solve_L_impl(L, b, x, D);
    }
#else
    matrix_solve_L_impl(L, b, x, D);
#endif
}

void matrix_solve_L(double* L, double* b, double* x, int D) {
#ifdef CBLAS
    if (D > CBLAS_LIMIT) {
        vector_copy(b, x, D);
        cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, D, L, D, x, 1);
    } else {
        matrix_solve_L_impl(L, b, x, D);
    }
#else
    matrix_solve_L_impl(L, b, x, D);
#endif
}

void matrix_solve_Lt(float* L, float* b, float* x, int D) {
#ifdef CBLAS
    if (D > CBLAS_LIMIT) {
        vector_copy(b, x, D);
        cblas_strsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, D, L, D, x, 1);
    } else {
        matrix_solve_Lt_impl(L, b, x, D);
    }
#else
    matrix_solve_Lt_impl(L, b, x, D);
#endif
}

void matrix_solve_Lt(double* L, double* b, double* x, int D) {
#ifdef CBLAS
    if (D > CBLAS_LIMIT) {
        vector_copy(b, x, D);
        cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, D, L, D, x, 1);
    } else {
        matrix_solve_Lt_impl(L, b, x, D);
    }
#else
    matrix_solve_Lt_impl(L, b, x, D);
#endif
}

void matrix_solve_pd(float* L, float* b, float* x, int D) {
#ifdef CBLAS
    if (D > CBLAS_LIMIT) {
        vector_copy(b, x, D);
        cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, D, L, D, x, 1);
        cblas_strsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, D, L, D, x, 1);
    } else {
        matrix_solve_L_impl(L, b, x, D);
        matrix_solve_Lt_impl(L, x, x, D);
    }
#else
    matrix_solve_L_impl(L, b, x, D);
    matrix_solve_Lt_impl(L, x, x, D);
#endif
}

void matrix_solve_pd(double* L, double* b, double* x, int D) {
#ifdef CBLAS
    if (D > CBLAS_LIMIT) {
        vector_copy(b, x, D);
        cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, D, L, D, x, 1);
        cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, D, L, D, x, 1);
    } else {
        matrix_solve_L_impl(L, b, x, D);
        matrix_solve_Lt_impl(L, x, x, D);
    }
#else
    matrix_solve_L_impl(L, b, x, D);
    matrix_solve_Lt_impl(L, x, x, D);
#endif
}
