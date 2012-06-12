/*
 * fsv.inl
 *
 *  Created on: 07-Mar-2009
 *      Author: alee
 */

#include "gauss.ch"
#include "matrix.ch"

template < int Dx, int Dy>
__device__ float fsv_pdf(float* x, float* y, float* args) {
	float temp[Dy * Dx];
	float cov[Dy * Dy];
	// B is Dy x Dx
	float* B = args;
	float* Bt = args + Dx * Dy;
	// Psi is Dy * Dy
	float* Psi = args + 2 * Dx * Dy;
	float H[Dx];
	d_vector_exp(x, H, Dx);
	float mu[Dy];
	d_vector_zero(mu, Dy);

	d_matrix_times_diag(B, H, temp, Dy, Dx);

	d_matrix_times<Dx>(temp, Bt, cov, Dy, Dy);

	d_matrix_add(cov, Psi, cov, Dy, Dy);

	return gauss_pdf<Dy>(y, mu, cov);
}

template < int Dx, int Dy>
__device__ float log_fsv_pdf(float* x, float* y, float* args) {
    float temp[Dy * Dx];
    float cov[Dy * Dy];
    // B is Dy x Dx
    float* B = args;
    float* Bt = args + Dx * Dy;
    // Psi is Dy * Dy
    float* Psi = args + 2 * Dx * Dy;
    float H[Dx];
    d_vector_exp(x, H, Dx);
    float mu[Dy];
    d_vector_zero(mu, Dy);

    d_matrix_times_diag(B, H, temp, Dy, Dx);

    d_matrix_times<Dx>(temp, Bt, cov, Dy, Dy);

    d_matrix_add(cov, Psi, cov, Dy, Dy);

    return log_gauss_pdf<Dy>(y, mu, cov);
}
