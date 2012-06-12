/*
 * kiss.c
 *
 *  Created on: 04-Mar-2009
 *      Author: alee
 */

#include "kiss.h"

unsigned int x_KISS = 123456789, y_KISS = 362436000, z_KISS = 521288629, c_KISS = 7654321; /* Seed variables */

unsigned int KISS() {
	unsigned long long t, a = 698769069ULL;
	x_KISS = 69069 * x_KISS + 12345;
	y_KISS ^= (y_KISS << 13);
	y_KISS ^= (y_KISS >> 17);
	y_KISS ^= (y_KISS << 5);
	t = a * z_KISS + c_KISS;
	c_KISS = (t >> 32);
	return x_KISS + y_KISS + (z_KISS = ((unsigned int) t));
}

void KISS_burn(int N) {
	unsigned long long t, a = 698769069ULL;
	int i;
	for (i = 0; i < N; i++) {
		x_KISS = 69069 * x_KISS + 12345;
		y_KISS ^= (y_KISS << 13);
		y_KISS ^= (y_KISS >> 17);
		y_KISS ^= (y_KISS << 5);
		t = a * z_KISS + c_KISS;
		c_KISS = (t >> 32);
	}
}

void KISS_reset() {
	x_KISS = 123456789;
	y_KISS = 362436000;
	z_KISS = 521288629;
	c_KISS = 7654321;
}
