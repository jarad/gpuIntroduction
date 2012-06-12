/*
 * mix_gauss.ch
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#ifndef MIX_GAUSS_CH_
#define MIX_GAUSS_CH_

#include "gauss.ch"

// DIFFERENT WEIGHTS, SIGMAS

// returns p(x | mus, ws, sigmas) = sum(p(x | mu, ws, sigmas))
__device__ float mix_gauss1_pdf(float x, int k, float* mus,
        float* c1s, float* c2s) {
    float result = 0;
    for (int i = 0; i < k; i++) {
        result += gauss1_pdf(x, c1s[i], c2s[i], mus[i]);
    }
    return result;
}

// wrapper for above
__device__ float mix_gauss1_pdf(float x, float* args) {
    int k = (int) args[0];
    float* mus = args + 1;
    float* c1s = args + 1 + k;
    float* c2s = args + 1 + 2 * k;
    return mix_gauss1_pdf(x, k, mus, c1s, c2s);
}

// returns log p(x | mus, ws, sigmas) using log sum exp trick
template <int k>
__device__ float log_mix_gauss1_pdf(float x, float* c1, float* c2,
        float *mus) {
    float vals[k];
    for (int i = 0; i < k; i++) {
        vals[i] = log_gauss1_pdf(x, c1[i], c2[i], mus[i]);
    }
    float maxv = vals[0];
    int i_maxv = 0;
    for (int i = 1; i < k; i++) {
        if (vals[i]> maxv) {
            maxv = vals[i];
            i_maxv = i;
        }
    }
    float result = 1;
    for (int i = 0; i < k; i++) {
        if (i != i_maxv) {
            result += expf(vals[i] - maxv);
        }
    }
    return maxv + logf(result);
}

// template switch
__device__ float log_mix_gauss1_pdf(float x, int k, float* c1, float* c2,
        float *mus) {
    switch (k) {
    case 2:
        return log_mix_gauss1_pdf < 2 > (x, c1, c2, mus);
    case 3:
        return log_mix_gauss1_pdf < 3 > (x, c1, c2, mus);
    case 4:
        return log_mix_gauss1_pdf < 4 > (x, c1, c2, mus);
    case 5:
        return log_mix_gauss1_pdf < 5 > (x, c1, c2, mus);
    case 6:
        return log_mix_gauss1_pdf < 6 > (x, c1, c2, mus);
    default:
        return -FLT_MAX;
    }
}

// wrapper
__device__ float log_mix_gauss1_pdf(float x, float* args) {
    int k = (int) args[0];
    float* mus = args + 1;
    float* c1s = args + 1 + k;
    float* c2s = args + 1 + 2 * k;
    return log_mix_gauss1_pdf(x, k, c1s, c2s, mus);
}

//// returns p(x | mus, ws, sigmas)
//__device__ float mix_gauss1_pdf(float x, int k, float* ws, float* mus,
//      float* sigmas) {
//  float result = 0;
//  for (int i = 0; i < k; i++) {
//      result += gauss1_pdf(x, mus[i], sigmas[i]);
//  }
//  return result;
//}
//
//// returns p(xs | mus, ws, sigmas) = prod(p(x | mus, ws, sigmas))
//__device__ float mix_gauss1_pdf(float* x, int N, int k, float* ws, float* mus,
//      float* sigmas) {
//  float logr = 0;
//  for (int i = 0; i < N; i++) {
//      logr += logf(mix_gauss1_pdf(x[i], k, ws, mus, sigmas));
//  }
//  return expf(logr);
//}

// SHARED WEIGHTS, SIGMAS

// returns p(x | mus)
template <int k>
__device__ float mix_gauss1_mus_pdf(float x, float c1, float c2,
        float *mus) {
    float result = 0;
    for (int i = 0; i < k; i++) {
        result += gauss1_pdf(x, c1, c2, mus[i]);
    }
    return result;
}

template <int k>
__device__ float log_mix_gauss1_mus_pdf(float x, float c1, float c2,
        float *mus) {
    float vals[k];
    for (int i = 0; i < k; i++) {
        vals[i] = log_gauss1_pdf(x, c1, c2, mus[i]);
    }
    float maxv = vals[0];
    int i_maxv = 0;
    for (int i = 1; i < k; i++) {
        if (vals[i]> maxv) {
            maxv = vals[i];
            i_maxv = i;
        }
    }
    float result = 1;
    for (int i = 0; i < k; i++) {
        if (i != i_maxv) {
            result += expf(vals[i] - maxv);
        }
    }
    return maxv + logf(result);
}

//__device__ float log_mix_gauss1_mus_pdf(float x, int k, float c1, float c2,
//      float *mus) {
//  switch (k) {
//  case 2:
//      return log_mix_gauss1_mus_pdf < 2 > (x, c1, c2, mus);
//  case 3:
//      return log_mix_gauss1_mus_pdf < 3 > (x, c1, c2, mus);
//  case 4:
//      return log_mix_gauss1_mus_pdf < 4 > (x, c1, c2, mus);
//  case 5:
//      return log_mix_gauss1_mus_pdf < 5 > (x, c1, c2, mus);
//  case 6:
//      return log_mix_gauss1_mus_pdf < 6 > (x, c1, c2, mus);
//  default:
//      return -FLT_MAX;
//  }
//}

template <int k>
__device__ float log_mix_gauss1_mus_pdf(float* x, int N, float c1,
        float c2, float *mus) {
    float logr = 0;
    for (int i = 0; i < N; i++) {
        logr += log_mix_gauss1_mus_pdf<k>(x[i], c1, c2, mus);
    }
    return logr;
}

template <int k>
__device__ float mix_gauss1_mus_pdf(float* x, int N, float c1, float c2,
        float *mus) {
    float logr = 0;
    for (int i = 0; i < N; i++) {
        logr += logf(mix_gauss1_mus_pdf<k>(x[i], c1, c2, mus));
    }
    return expf(logr);
}

#endif /* MIX_GAUSS_CH_ */
