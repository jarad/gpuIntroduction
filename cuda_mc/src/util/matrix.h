/*
 * matrix.h
 *
 *  Created on: 11-Feb-2009
 *      Author: alee
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include "func.h"
#include <math.h>

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

void matrix_print(float* matrix, int M, int N);
void matrix_print(double* matrix, int M, int N);
void matrix_print(unsigned int* matrix, int M, int N);
void matrix_print(int* matrix, int M, int N);
void matrix_print(long int* matrix, int M, int N);
void matrix_print(unsigned long int* matrix, int M, int N);

template <class T>
void vector_print(T* vector, int D) {
    matrix_print(vector, 1, D);
}

template<class T>
inline T matrix_access(T* matrix, int M, int N, int i, int j) {
    return matrix[i * N + j];
}

template<class T>
inline void matrix_set(T* matrix, int M, int N, int i, int j, T f) {
    matrix[i * N + j] = f;
}

template<class T>
inline void matrix_zero(T* matrix, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix[i] = 0;
    }
}

template<class T>
inline void matrix_copy(T* matrix, T* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix[i];
    }
}

template<class T>
inline void matrix_getL(T* LU, T* L, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i> j) {
                T LUij = matrix_access(LU, M, N, i, j);
                matrix_set(L, M, N, i, j, LUij);
            } else if (i == j) {
                matrix_set(L, M, N, i, j, 1.0);
            } else {
                matrix_set(L, M, N, i, j, 0.0);
            }
        }
    }
}

template<class T>
inline void matrix_getU(T* LU, T* U, int M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j) {
                T LUij = matrix_access(LU, M, N, i, j);
                matrix_set(U, M, N, i, j, LUij);
            } else {
                matrix_set(U, M, N, i, j, 0.0);
            }
        }
    }
}

template<class T>
inline void matrix_LU(T* matrix, T* LU, int* piv, int& pivsign, int M, int N) {
    matrix_copy(matrix, LU, M, N);

    //  int piv[M];
    for (int i = 0; i < M; i++) {
        piv[i] = i;
    }
    pivsign = 1;
    T* LUrowi;

    T* LUcolj = (T*) malloc(M * sizeof(T));

    // Outer loop.

    for (int j = 0; j < N; j++) {

        // Make a copy of the j-th column to localize references.

        for (int i = 0; i < M; i++) {
            LUcolj[i] = matrix_access(LU, M, N, i, j);
        }

        // Apply previous transformations.

        for (int i = 0; i < M; i++) {
            LUrowi = LU + i * N;

            // Most of the time is spent in the following dot product.

            int kmax = min(i, j);
            T s = 0.0;
            for (int k = 0; k < kmax; k++) {
                s += LUrowi[k] * LUcolj[k];
            }

            LUrowi[j] = LUcolj[i] -= s;
        }

        // Find pivot and exchange if necessary.

        int p = j;
        for (int i = j + 1; i < M; i++) {
            if (fabs(LUcolj[i])> fabs(LUcolj[p])) {
                p = i;
            }
        }
        if (p != j) {
            for (int k = 0; k < N; k++) {
                T t = matrix_access(LU, M, N, p, k);
                matrix_set(LU, M, N, p, k, matrix_access(LU, M, N, j, k));
                matrix_set(LU, M, N, j, k, t);
            }
            int k = piv[p];
            piv[p] = piv[j];
            piv[j] = k;
            pivsign = -pivsign;
        }

        // Compute multipliers.

        if (j < M & matrix_access(LU, M, N, j, j) != 0.0) {
            for (int i = j + 1; i < M; i++) {
                T LUjj = matrix_access(LU, M, N, j, j);
                T LUij = matrix_access(LU, M, N, i, j);
                matrix_set(LU, M, N, i, j, LUij / LUjj);
                //              LU[i][j] /= LU[j][j];
            }
        }
    }

    //  printf("piv: \n");
    //  for (int i = 0; i < M; i++) {
    //      printf("%d ", piv[i]);
    //  }
    //  printf("\n");
    free(LUcolj);

}

template<class T>
inline void matrix_identity(T* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                matrix_set(matrix, N, N, i, j, (T) 1.0);
            } else {
                matrix_set(matrix, N, N, i, j, (T) 0.0);
            }
        }
    }
}

template<class T>
inline void matrix_sub(T* matrix, T* out, int* r, int M, int N, int j0, int j1) {
    //      Matrix X = new Matrix(r.length,j1-j0+1);
    //      double[][] B = X.getArray();
    for (int i = 0; i < M; i++) {
        for (int j = j0; j <= j1; j++) {
            matrix_set(out, M, j1 - j0 + 1, i, j - j0, matrix_access(matrix, M, N, r[i], j));
            //          B[i][j-j0] = A[r[i]][j];
        }
    }
}

template<class T>
inline void matrix_solve(T* A, T* B, T* X, int MA, int NA, int MB, int NB) {
    T* LU = (T*) malloc(MA * NA * sizeof(T));
    //  float LU[MA * NA];
    int* piv = (int*) malloc(MA * sizeof(int));
    //  int piv[MA];
    int pivsign;

    matrix_LU(A, LU, piv, pivsign, MA, NA);

    matrix_sub(B, X, piv, MB, NB, 0, NB - 1);

    // Solve L*Y = B(piv,:)
    for (int k = 0; k < NA; k++) {
        for (int i = k + 1; i < NA; i++) {
            for (int j = 0; j < NB; j++) {
                T r = matrix_access(X, NA, NB, i, j) - matrix_access(X, NA, NB, k, j)
                * matrix_access(LU, MA, NA, i, k);
                matrix_set(X, NA, NB, i, j, r);
                // X[i][j] -= X[k][j] * LU[i][k];
            }
        }
    }

    // Solve U*X = Y;
    for (int k = NA - 1; k >= 0; k--) {
        for (int j = 0; j < NB; j++) {
            T r1 = matrix_access(X, NA, NB, k, j);
            T r2 = matrix_access(LU, MA, NA, k, k);
            matrix_set(X, NA, NB, k, j, r1 / r2);
            // X[k][j] /= LU[k][k];
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < NB; j++) {
                T r1 = matrix_access(X, NA, NB, i, j);
                T r2 = matrix_access(X, NA, NB, k, j);
                T r3 = matrix_access(LU, MA, NA, i, k);
                matrix_set(X, NA, NB, i, j, r1 - r2 * r3);
                // X[i][j] -= X[k][j] * LU[i][k];
            }
        }
    }
    free(piv);
    free(LU);
    //      return Xmat;
}

template<class T>
inline void matrix_solve_pd(T* A, T* B, T* X, int MA, int NA, int MB, int NB) {
    T* L = (T*) malloc(MA * NA * sizeof(T));

    matrix_chol(A, L, MA);

    for (int i = 0; i < MA * NB; i++) {
        X[i] = B[i];
    }

    // Solve L*Y = B;
    for (int k = 0; k < MA; k++) {
        for (int j = 0; j < NB; j++) {
            for (int i = 0; i < k; i++) {
                T r = matrix_access(X, MA, NB, k, j) - matrix_access(X, MA, NB, i, j) * matrix_access(L, MA, NA, k, i);
                matrix_set(X, MA, NB, k, j, r);
            }
            T r = matrix_access(X, MA, NB, k, j) / matrix_access(L, MA, NA, k, k);
            matrix_set(X, MA, NB, k, j, r);
        }
    }

    // Solve L'*X = Y;
    for (int k = MA - 1; k >= 0; k--) {
        for (int j = 0; j < NB; j++) {
            for (int i = k+1; i < MA; i++) {
                T r = matrix_access(X, MA, NB, k, j) - matrix_access(X, MA, NB, i, j) * matrix_access(L, MA, NA, i, k);
                matrix_set(X, MA, NB, k, j, r);
            }
            T r = matrix_access(X, MA, NB, k, j) / matrix_access(L, MA, NA, k, k);
            matrix_set(X, MA, NB, k, j, r);
        }
    }

    free(L);

}

void matrix_solve_L(float* L, float* b, float* x, int D);
void matrix_solve_L(double* L, double* b, double* x, int D);

void matrix_solve_Lt(float* L, float* b, float* x, int D);
void matrix_solve_Lt(double* L, double* b, double* x, int D);

void matrix_solve_pd(float* L, float* b, float* x, int D);
void matrix_solve_pd(double* L, double* b, double* x, int D);

template<class T>
inline void matrix_solve_L(T* L, T* B, T* X, int D, int NB) {
    for (int i = 0; i < D * NB; i++) {
        X[i] = B[i];
    }

    for (int k = 0; k < D; k++) {
        for (int j = 0; j < NB; j++) {
            for (int i = 0; i < k; i++) {
                T r = matrix_access(X, D, NB, k, j) - matrix_access(X, D, NB, i, j)
                * matrix_access(L, D, D, k, i);
                matrix_set(X, D, NB, k, j, r);
            }
            T r = matrix_access(X, D, NB, k, j) / matrix_access(L, D, D, k, k);
            matrix_set(X, D, NB, k, j, r);
        }
    }
}

template<class T>
inline void matrix_solve_Lt(T* L, T* B, T* X, int D, int NB) {
    for (int i = 0; i < D * NB; i++) {
        X[i] = B[i];
    }

    for (int k = D - 1; k >= 0; k--) {
        for (int j = 0; j < NB; j++) {
            for (int i = k + 1; i < D; i++) {
                T r = matrix_access(X, D, NB, k, j) - matrix_access(X, D, NB, i, j)
                * matrix_access(L, D, D, i, k);
                matrix_set(X, D, NB, k, j, r);
            }
            T r = matrix_access(X, D, NB, k, j) / matrix_access(L, D, D, k, k);
            matrix_set(X, D, NB, k, j, r);
        }
    }
}

template<class T>
inline void matrix_solve_pd(T* L, T* B, T* X, int D, int NB) {
    matrix_solve_L(L, B, X, D, NB);
    matrix_solve_Lt(L, X, X, D, NB);
}

template<class T>
inline void matrix_inverse(T* matrix, T* inverse, int N) {
    //  float identity[N * N];
    T* identity = (T*) malloc(N * N * sizeof(T));

    matrix_identity(identity, N);
    matrix_solve(matrix, identity, inverse, N, N, N, N);

    free(identity);
}

template<class T>
inline void matrix_inverse_pd(T* matrix, T* inverse, int N) {
    //  float identity[N * N];
    T* identity = (T*) malloc(N * N * sizeof(T));

    matrix_identity(identity, N);
    matrix_solve_pd(matrix, identity, inverse, N, N, N, N);

    free(identity);
}

template <class T>
void matrix_times_old(T* matrix1, T* matrix2, T* matrix_out, int M1,
        int N1, int M2, int N2) {
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            T acc = 0;
            for (int k = 0; k < N1; k++) {
                acc += matrix_access(matrix1, M1, N1, i, k) * matrix_access(
                        matrix2, M2, N2, k, j);
            }
            matrix_set(matrix_out, M1, N2, i, j, acc);
        }
    }
}

void
        matrix_times(float* matrix1, float* matrix2, float* matrix_out, int M1, int N1, int M2,
                int N2);
void matrix_times(double* matrix1, double* matrix2, double* matrix_out, int M1, int N1, int M2,
        int N2);

void matrix_XXt(float* X, float* A, int M, int N);
void matrix_XXt(double* X, double* A, int M, int N);
void matrix_XtX(float* X, float* A, int M, int N);
void matrix_XtX(double* X, double* A, int M, int N);

int matrix_chol(float* matrix, float* L, int D);
int matrix_chol(double* matrix, double* L, int D);

template<class T>
inline void matrix_copy_L(T* A, T* L, int D) {
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            matrix_set(L, D, D, i, j, matrix_access(A, D, D, i, j));
        }
        for (int j = i + 1; j < D; j++) {
            matrix_set(L, D, D, i, j, (T) 0);
        }
    }
}

template<class T>
inline void matrix_fill_symm(T* A, int D) {
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < i; j++) {
            matrix_set(A, D, D, j, i, matrix_access(A, D, D, i, j));
        }
    }
}

template<class T>
inline void matrix_add(T* matrix1, T* matrix2, T* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] + matrix2[i];
    }
}

template<class T>
inline void matrix_transpose(T* matrix, T* matrix_out, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_set(matrix_out, N, M, j, i, matrix_access(matrix, M, N, i, j));
        }
    }
}

template<class T>
inline void matrix_minus(T* matrix1, T* matrix2, T* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] - matrix2[i];
    }
}

template<class T>
inline void vector_add(T* v1, T* v2, T* v_out, int D) {
    for (int i = 0; i < D; i++) {
        v_out[i] = v1[i] + v2[i];
    }
}

template<class T>
inline void vector_set(T* v1, T* v2, int D) {
    for (int i = 0; i < D; i++) {
        v1[i] = v2[i];
    }
}

template<class T>
inline void vector_zero(T* v, int D) {
    for (int i = 0; i < D; i++) {
        v[i] = 0;
    }
}

template<class T>
inline void matrix_mod(T* matrix, T* matrix_out, int M, int N, unsigned long m) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix[i] % m;
    }
}

template<class T>
inline void matrix_times_mod(T* matrix1, T* matrix2, T* matrix_out, int M1, int N1,
        int M2, int N2, unsigned long m) {
    T* Bcolj = (T*) malloc(N1 * sizeof(T));
    for (int j = 0; j < N2; j++) {
        for (int k = 0; k < N1; k++) {
            Bcolj[k] = matrix_access(matrix2, M2, N2, k, j);
        }
        for (int i = 0; i < M1; i++) {
            T* Arowi = matrix1 + i * N1;
            T s = 0;
            for (int k = 0; k < N1; k++) {
                T r = (Arowi[k] * Bcolj[k]) % m;
                s = (s + r) % m;
                //                s = (s + Arowi[k] * Bcolj[k]) % m;
            }
            matrix_set(matrix_out, M1, N2, i, j, s);
        }
    }
    free(Bcolj);
}

template<class T>
inline T matrix_det(T* matrix, int N) {
    //  float LU[N * N];
    T* LU = (T*) malloc(N * N * sizeof(T));
    //  int piv[N];
    int* piv = (int*) malloc(N * sizeof(int));
    int pivsign;
    matrix_LU(matrix, LU, piv, pivsign, N, N);
    //    T d = (T) pivsign;
    //    for (int j = 0; j < N; j++) {
    //        d *= matrix_access(LU, N, N, j, j);
    //    }
    T d = (T) log(matrix_access(LU, N, N, 0, 0));
    for (int j = 1; j < N; j++) {
        d += (T) log(matrix_access(LU, N, N, j, j));
    }
    d = (T) (pivsign * exp(d));
    free(LU);
    free(piv);
    return d;
}

template<class T>
inline T matrix_xtmx(T* matrix, T* x, int N) {
    T* tmp = (T*) malloc(N * sizeof(T));
    matrix_times(matrix, x, tmp, N, N, N, 1);
    T r;
    matrix_times(x, tmp, &r, 1, N, N, 1);
    free(tmp);
    return r;
}

template<class T>
inline void matrix_add(T* matrix1, T value, T* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] + value;
    }
}

template <class T>
inline void matrix_minus(T* matrix1, T value, T* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] - value;
    }
}

template <class T>
inline T* vector_get(T* array, int D, int j) {
    return array + j * D;
}

template <class T>
inline void vector_swap(T* v1, T* v2, int D) {
    float t;
    for (int i = 0; i < D; i++) {
        t = v1[i];
        v1[i] = v2[i];
        v2[i] = t;
    }
}

template <class T>
inline void matrix_times(T* matrix, T* matrix_out, T k, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix[i] * k;
    }
}

template <class T>
inline void matrix_times_diag(T* matrix1, T* diag, T* matrix_out, int M,
        int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix_set(matrix_out, M, N, i, j, matrix_access(matrix1, M, N, i,
                            j) * diag[j]);
        }
    }
}

template <class T>
inline void vector_exp(T* v, T* v_out, int D) {
    for (int i = 0; i < D; i++) {
        v_out[i] = (T) exp(v[i]);
    }
}

template <class T>
inline void matrix_det_inv_pd(T* cov, T &c1, T* c2, int D) {
    T* L = (T*) malloc(D * D * sizeof(T));
    matrix_chol(cov, L, D);
    T v = (T) log(matrix_access(L, D, D, 0, 0));
    for (int j = 1; j < D; j++) {
        v += (T) log(matrix_access(L, D, D, j, j));
    }
    c1 = (T) exp(v * 2);

    T* identity = (T*) malloc(D * D * sizeof(T));

    matrix_identity(identity, D);
    matrix_solve_pd(L, identity, c2, D, D);

    free(L);
    free(identity);
}

template <class T>
inline void matrix_det_chol_pd(T* cov, T &c1, T* L, int D) {
    matrix_chol(cov, L, D);
    T v = (T) log(matrix_access(L, D, D, 0, 0));
    for (int j = 1; j < D; j++) {
        v += (T) log(matrix_access(L, D, D, j, j));
    }
    c1 = (T) exp(v * 2);
}

template<class T>
inline T vector_xtx(T* x, int N) {
    T r = 0;
    for (int i = 0; i < N; i++) {
        r += x[i] * x[i];
    }
    return r;
}

template<class T>
inline T vector_xty(T* x, T* y, int N) {
    T r = 0;
    for (int i = 0; i < N; i++) {
        r += x[i] * y[i];
    }
    return r;
}

template <class T>
inline void vector_copy(T* x, T* out, int D) {
    for (int i = 0; i < D; i++) {
        out[i] = x[i];
    }
}

#endif /* MATRIX_H_ */
