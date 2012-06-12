/*
 * matrix.ch
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#ifndef MATRIX_CH_
#define MATRIX_CH_

#include <math.h>
#include "func.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

template<class T>
__device__ T d_matrix_access(T* matrix, int M, int N, int i, int j) {
    return matrix[i * N + j];
}

template<class T>
__device__ void d_matrix_set(T* matrix, int M, int N, int i, int j, T f) {
    matrix[i * N + j] = f;
}

__device__ void d_matrix_copy(float* matrix, float* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix[i];
    }
}

__device__ void d_matrix_add(float* matrix1, float* matrix2, float* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] + matrix2[i];
    }
}

__device__ void d_matrix_add(float* matrix1, float value, float* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] + value;
    }
}

__device__ void d_matrix_minus(float* matrix1, float value, float* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] - value;
    }
}

__device__ void d_matrix_minus(float* matrix1, float* matrix2, float* matrix_out, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix1[i] - matrix2[i];
    }
}

__device__ void d_matrix_times_diag(float* matrix1, float* diag, float* matrix_out, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            d_matrix_set(matrix_out, M, N, i, j, d_matrix_access(matrix1, M, N, i, j) * diag[j]);
        }
    }
}

__device__ void d_matrix_times(float* matrix1, float* matrix2, float* matrix_out, int M1, int N1,
        int M2, int N2) {
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            float acc = 0;
            for (int k = 0; k < N1; k++) {
                acc += d_matrix_access(matrix1, M1, N1, i, k) * d_matrix_access(matrix2, M2, N2, k,
                        j);
            }
            d_matrix_set(matrix_out, M1, N2, i, j, acc);
        }
    }
}

template<int N1>
__device__ void d_matrix_times(float* matrix1, float* matrix2, float* matrix_out, int M1, int N2) {
    float Bcolj[N1];
    for (int j = 0; j < N2; j++) {
        for (int k = 0; k < N1; k++) {
            Bcolj[k] = d_matrix_access(matrix2, N1, N2, k, j);
        }
        for (int i = 0; i < M1; i++) {
            float* Arowi = matrix1 + i * N1;
            float s = 0;
            for (int k = 0; k < N1; k++) {
                s += Arowi[k] * Bcolj[k];
            }
            d_matrix_set(matrix_out, M1, N2, i, j, s);
        }
    }

    //  for (int i = 0; i < M1; i++) {
    //      for (int j = 0; j < N2; j++) {
    //          float acc = 0;
    //          for (int k = 0; k < N1; k++) {
    //              acc += d_matrix_access(matrix1, M1, N1, i, k)
    //              * d_matrix_access(matrix2, M2, N2, k, j);
    //          }
    //          d_matrix_set(matrix_out, M1, N2, i, j, acc);
    //      }
    //  }

}

template<class T, int N1>
__device__ void d_matrix_times_mod(T* matrix1, T* matrix2, T* matrix_out, int M1, int N2, T m) {
    T Bcolj[N1];
    for (int j = 0; j < N2; j++) {
        for (int k = 0; k < N1; k++) {
            Bcolj[k] = d_matrix_access(matrix2, N1, N2, k, j);
        }
        for (int i = 0; i < M1; i++) {
            T* Arowi = matrix1 + i * N1;
            T s = 0;
            for (int k = 0; k < N1; k++) {
                T r = (Arowi[k] * Bcolj[k]) % m;
                s = (s + r) % m;
            }
            d_matrix_set(matrix_out, M1, N2, i, j, s);
        }
    }
}

template<class T>
__device__ void d_matrix_times_mod(T* matrix1, T* matrix2, T* matrix_out, int M1, int N1, int M2,
        int N2, T m) {
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            T acc = 0;
            for (int k = 0; k < N1; k++) {
                T t1 = d_matrix_access(matrix1, M1, N1, i, k);
                T t2 = d_matrix_access(matrix2, M2, N2, k, j);
                //                T t3 = (t1 * t2) % m;
                //                acc = (acc + t3) % m;
                acc = (acc + t1 * t2) % m;
            }
            d_matrix_set(matrix_out, M1, N2, i, j, acc);
        }
    }
}

//__device__ void d_matrix_times_mod(unsigned long* matrix1, unsigned long* matrix2,
//        unsigned long* matrix_out, int M1, int N1, int M2, int N2, unsigned long m) {
//    for (int i = 0; i < M1; i++) {
//        for (int j = 0; j < N2; j++) {
//            unsigned long acc = 0;
//            for (int k = 0; k < N1; k++) {
//                unsigned long t1 = d_matrix_access(matrix1, M1, N1, i, k);
//                unsigned long t2 = d_matrix_access(matrix2, M2, N2, k, j);
//                unsigned long t3 = (t1 * t2) % m;
//                acc = (acc + t3) % m;
//            }
//            d_matrix_set(matrix_out, M1, N2, i, j, acc);
//        }
//    }
//}

__device__ void d_vector_add(float* v1, float* v2, float* v_out, int D) {
    for (int i = 0; i < D; i++) {
        v_out[i] = v1[i] + v2[i];
    }
}

__device__ float* d_vector_get(float* array, int D, int j) {
    return array + j * D;
}

__device__ void d_vector_set(float* v1, float* v2, int D) {
    for (int i = 0; i < D; i++) {
        v1[i] = v2[i];
    }
}

template<int N>
__device__ float d_matrix_xtmx(float* matrix, float* x) {
    float tmp[N];
    d_matrix_times<N> (matrix, x, tmp, N, 1);
    float r;
    d_matrix_times(x, tmp, &r, 1, N, N, 1);
    return r;
}

__device__ void d_vector_swap(float* v1, float* v2, int D) {
    float t;
    for (int i = 0; i < D; i++) {
        t = v1[i];
        v1[i] = v2[i];
        v2[i] = t;
    }
}

__device__ void d_matrix_times(float* matrix, float* matrix_out, float k, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix_out[i] = matrix[i] * k;
    }
}

__device__ void d_vector_exp(float* v, float* v_out, int D) {
    for (int i = 0; i < D; i++) {
        v_out[i] = expf(v[i]);
    }
}

__device__ void d_vector_zero(float* v, int D) {
    for (int i = 0; i < D; i++) {
        v[i] = 0;
    }
}

template<class T>
__device__ void d_matrix_identity(T* matrix, int N) {
    T one = 1;
    T zero = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                d_matrix_set(matrix, N, N, i, j, one);
            } else {
                d_matrix_set(matrix, N, N, i, j, zero);
            }
        }
    }
}

template<int M, int N>
__device__ void d_matrix_LU(float* matrix, float* LU, int* piv, int& pivsign) {
    d_matrix_copy(matrix, LU, M, N);

    //  int piv[M];
    for (int i = 0; i < M; i++) {
        piv[i] = i;
    }
    pivsign = 1;
    float* LUrowi;
    //  float LUcolj[M];
    //  float* LUcolj = (float*) malloc(M* sizeof(float));
    float LUcolj[M];

    // Outer loop.

    for (int j = 0; j < N; j++) {

        // Make a copy of the j-th column to localize references.

        for (int i = 0; i < M; i++) {
            LUcolj[i] = d_matrix_access(LU, M, N, i, j);
        }

        // Apply previous transformations.

        for (int i = 0; i < M; i++) {
            LUrowi = LU + i * N;

            // Most of the time is spent in the following dot product.

            int kmax = min(i, j);
            float s = 0.0;
            for (int k = 0; k < kmax; k++) {
                s += LUrowi[k] * LUcolj[k];
            }

            LUrowi[j] = LUcolj[i] -= s;
        }

        // Find pivot and exchange if necessary.

        int p = j;
        for (int i = j + 1; i < M; i++) {
            if (abs(LUcolj[i]) > abs(LUcolj[p])) {
                p = i;
            }
        }
        if (p != j) {
            for (int k = 0; k < N; k++) {
                float t = d_matrix_access(LU, M, N, p, k);
                d_matrix_set(LU, M, N, p, k, d_matrix_access(LU, M, N, j, k));
                d_matrix_set(LU, M, N, j, k, t);
            }
            int k = piv[p];
            piv[p] = piv[j];
            piv[j] = k;
            pivsign = -pivsign;
        }

        // Compute multipliers.

        if (j < M & d_matrix_access(LU, M, N, j, j) != 0.0) {
            for (int i = j + 1; i < M; i++) {
                float LUjj = d_matrix_access(LU, M, N, j, j);
                float LUij = d_matrix_access(LU, M, N, i, j);
                d_matrix_set(LU, M, N, i, j, LUij / LUjj);
                //              LU[i][j] /= LU[j][j];
            }
        }
    }

    //  printf("piv: \n");
    //  for (int i = 0; i < M; i++) {
    //      printf("%d ", piv[i]);
    //  }
    //  printf("\n");
    //  free(LUcolj);

}

__device__ void d_matrix_sub(float* matrix, float* out, int* r, int M, int N, int j0, int j1) {
    //      Matrix X = new Matrix(r.length,j1-j0+1);
    //      double[][] B = X.getArray();
    for (int i = 0; i < M; i++) {
        for (int j = j0; j <= j1; j++) {
            d_matrix_set(out, M, j1 - j0 + 1, i, j - j0, d_matrix_access(matrix, M, N, r[i], j));
            //          B[i][j-j0] = A[r[i]][j];
        }
    }
}

// A*X = B
// takes LU, piv of A as input
__device__ void d_matrix_solve(float* LU, int* piv, float* B, float* X, int MA, int NA, int MB,
        int NB) {

    float r;

    d_matrix_sub(B, X, piv, MB, NB, 0, NB - 1);

    // Solve L*Y = B(piv,:)
    for (int k = 0; k < NA; k++) {
        for (int i = k + 1; i < NA; i++) {
            for (int j = 0; j < NB; j++) {
                r = d_matrix_access(X, NA, NB, i, j) - d_matrix_access(X, NA, NB, k, j)
                        * d_matrix_access(LU, MA, NA, i, k);
                d_matrix_set(X, NA, NB, i, j, r);
                // X[i][j] -= X[k][j] * LU[i][k];
            }
        }
    }

    // Solve U*X = Y;
    for (int k = NA - 1; k >= 0; k--) {
        for (int j = 0; j < NB; j++) {
            r = d_matrix_access(X, NA, NB, k, j) / d_matrix_access(LU, MA, NA, k, k);
            d_matrix_set(X, NA, NB, k, j, r);
            // X[k][j] /= LU[k][k];
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < NB; j++) {
                r = d_matrix_access(X, NA, NB, i, j) - d_matrix_access(X, NA, NB, k, j)
                        * d_matrix_access(LU, MA, NA, i, k);
                d_matrix_set(X, NA, NB, i, j, r);
                // X[i][j] -= X[k][j] * LU[i][k];
            }
        }
    }
}

template<int D>
__device__ void d_matrix_det_inv(float* cov, float &c1, float* c2) {
    float LU[D * D];
    int piv[D];
    int pivsign;
    d_matrix_LU<D, D> (cov, LU, piv, pivsign);
    c1 = (float) pivsign;
    for (int j = 0; j < D; j++) {
        c1 *= d_matrix_access(LU, D, D, j, j);
    }

    float identity[D * D];
    d_matrix_identity(identity, D);
    d_matrix_solve(LU, piv, identity, c2, D, D, D, D);

}

template<int D>
__device__ void d_matrix_chol(float* matrix, float* L) {
    float s, d;
    float* Lrowk;
    float* Lrowj;
    int i, j, k;
    for (j = 0; j < D; j++) {
        Lrowj = L + j * D;
        d = 0;
        for (k = 0; k < j; k++) {
            Lrowk = L + k * D;
            s = 0;
            for (i = 0; i < k; i++) {
                s += Lrowk[i] * Lrowj[i];
            }
            s = (d_matrix_access(matrix, D, D, j, k) - s) / d_matrix_access(L, D, D, k, k);
            Lrowj[k] = s;
            d += s * s;
        }
        d = d_matrix_access(matrix, D, D, j, j) - d;
        d_matrix_set(L, D, D, j, j, sqrtf(max(d, 0.0f)));
        for (int k = j + 1; k < D; k++) {
            d_matrix_set(L, D, D, j, k, 0.0f);
        }
    }
}

__device__ void d_matrix_solve_L(float* L, float* b, float* x, int D) {
    for (int i = 0; i < D; i++) {
        x[i] = b[i];
    }
    //
    //    for (int i = 0; i < D; i++) {
    //        for (int j = 0; j < i; j++) {
    //            x[i] -= d_matrix_access(L, D, D, i, j) * x[j];
    //        }
    //        x[i] /= d_matrix_access(L, D, D, i, i);
    //    }
    float v;
    for (int i = 0; i < D; i++) {
        v = x[i];
        for (int j = 0; j < i; j++) {
            v -= d_matrix_access(L, D, D, i, j) * x[j];
        }
        x[i] = v / d_matrix_access(L, D, D, i, i);
    }
}

__device__ void d_matrix_solve_Lt(float* L, float* b, float* x, int D) {

    for (int i = 0; i < D; i++) {
        x[i] = b[i];
    }
    //
    //    for (int i = D-1; i >= 0; i--) {
    //        for (int j = i + 1; j < D; j++) {
    //            x[i] -= d_matrix_access(L, D, D, j, i) * x[j];
    //        }
    //        x[i] /= d_matrix_access(L, D, D, i, i);
    //    }
    float v;
    for (int i = D - 1; i >= 0; i--) {
        v = x[i];
        for (int j = i + 1; j < D; j++) {
            v -= d_matrix_access(L, D, D, j, i) * x[j];
        }
        x[i] = v / d_matrix_access(L, D, D, i, i);
    }
}

__device__ void d_matrix_solve_pd(float* L, float* b, float* x, int D) {
    d_matrix_solve_L(L, b, x, D);
    d_matrix_solve_Lt(L, x, x, D);
}

__device__ void d_matrix_solve_L(float* L, float* B, float* X, int D, int NB) {
    for (int i = 0; i < D * NB; i++) {
        X[i] = B[i];
    }

    for (int k = 0; k < D; k++) {
        for (int j = 0; j < NB; j++) {
            for (int i = 0; i < k; i++) {
                float r = d_matrix_access(X, D, NB, k, j) - d_matrix_access(X, D, NB, i, j)
                        * d_matrix_access(L, D, D, k, i);
                d_matrix_set(X, D, NB, k, j, r);
            }
            float r = d_matrix_access(X, D, NB, k, j) / d_matrix_access(L, D, D, k, k);
            d_matrix_set(X, D, NB, k, j, r);
        }
    }
}

__device__ void d_matrix_solve_Lt(float* L, float* B, float* X, int D, int NB) {
    for (int i = 0; i < D * NB; i++) {
        X[i] = B[i];
    }

    for (int k = D - 1; k >= 0; k--) {
        for (int j = 0; j < NB; j++) {
            for (int i = k + 1; i < D; i++) {
                float r = d_matrix_access(X, D, NB, k, j) - d_matrix_access(X, D, NB, i, j)
                        * d_matrix_access(L, D, D, i, k);
                d_matrix_set(X, D, NB, k, j, r);
            }
            float r = d_matrix_access(X, D, NB, k, j) / d_matrix_access(L, D, D, k, k);
            d_matrix_set(X, D, NB, k, j, r);
        }
    }
}

__device__ void d_matrix_solve_pd(float* L, float* B, float* X, int D, int NB) {
    d_matrix_solve_L(L, B, X, D, NB);
    d_matrix_solve_Lt(L, X, X, D, NB);
}

//template<int D>
//__device__ void d_matrix_solve_pd(float* L, float* B, float* X, int NB) {
//
//    for (int i = 0; i < D * NB; i++) {
//        X[i] = B[i];
//    }
//
//    // Solve L*Y = B;
//    for (int k = 0; k < D; k++) {
//        for (int j = 0; j < NB; j++) {
//            for (int i = 0; i < k; i++) {
//                float r = d_matrix_access(X, D, NB, k, j) - d_matrix_access(X, D, NB, i, j)
//                        * d_matrix_access(L, D, D, k, i);
//                d_matrix_set(X, D, NB, k, j, r);
//            }
//            float r = d_matrix_access(X, D, NB, k, j) / d_matrix_access(L, D, D, k, k);
//            d_matrix_set(X, D, NB, k, j, r);
//        }
//    }
//
//    // Solve L'*X = Y;
//    for (int k = D - 1; k >= 0; k--) {
//        for (int j = 0; j < NB; j++) {
//            for (int i = k + 1; i < D; i++) {
//                float r = d_matrix_access(X, D, NB, k, j) - d_matrix_access(X, D, NB, i, j)
//                        * d_matrix_access(L, D, D, i, k);
//                d_matrix_set(X, D, NB, k, j, r);
//            }
//            float r = d_matrix_access(X, D, NB, k, j) / d_matrix_access(L, D, D, k, k);
//            d_matrix_set(X, D, NB, k, j, r);
//        }
//    }
//
//}

template<int D>
__device__ void d_matrix_det_inv_pd(float* cov, float &c1, float* c2) {
    float L[D * D];
    d_matrix_chol<D> (cov, L);
    //    c1 = 1;
    //    for (int j = 0; j < D; j++) {
    //        c1 *= d_matrix_access(L, D, D, j, j);
    //    }
    //    c1 *= c1;
    c1 = logf(d_matrix_access(L, D, D, 0, 0));
    for (int j = 1; j < D; j++) {
        c1 += logf(d_matrix_access(L, D, D, j, j));
    }
    c1 = expf(c1 * 2);

    float identity[D * D];
    d_matrix_identity(identity, D);
    d_matrix_solve_pd(L, identity, c2, D);

}

template<int D>
__device__ void d_matrix_det_chol_pd(float* cov, float &c1, float* L) {
    //    T* L = (T*) malloc(D * D * sizeof(T));
    d_matrix_chol<D> (cov, L);
    float v = logf(d_matrix_access(L, D, D, 0, 0));
    for (int j = 1; j < D; j++) {
        v += logf(d_matrix_access(L, D, D, j, j));
    }
    c1 = expf(v * 2);
}

__device__ float d_vector_xty(float* x, float* y, int N) {
    float r = 0;
    for (int i = 0; i < N; i++) {
        r += x[i] * y[i];
    }
    return r;
}

#endif /* MATRIX_CH_ */
