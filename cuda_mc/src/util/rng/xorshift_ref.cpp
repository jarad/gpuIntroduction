/*
 * xorshift.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "rng_shared.h"
#include "xorshift.h"

unsigned int h_seeds_xorshift[4] = { 123456789, 362436069, 521288629, 88675123 };

void burn_XS_REF(int N) {
    unsigned int x, y, z, w, tmp;
    x = h_seeds_xorshift[0];
    y = h_seeds_xorshift[1];
    z = h_seeds_xorshift[2];
    w = h_seeds_xorshift[3];

    int i;
    for (i = 0; i < N; i++) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));
    }

    h_seeds_xorshift[0] = x;
    h_seeds_xorshift[1] = y;
    h_seeds_xorshift[2] = z;
    h_seeds_xorshift[3] = w;
}

void seed_XS_REF(int N) {
    h_seeds_xorshift[0] = 123456789;
    h_seeds_xorshift[1] = 362436069;
    h_seeds_xorshift[2] = 521288629;
    h_seeds_xorshift[3] = 88675123;

    burn_XS_REF(N);
}

void populate_rand_XS_REF(float* array, int N) {
    unsigned int x, y, z, w, tmp;
    x = h_seeds_xorshift[0];
    y = h_seeds_xorshift[1];
    z = h_seeds_xorshift[2];
    w = h_seeds_xorshift[3];

    int i;
    for (i = 0; i < N; i++) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        array[i] = ((float) w) / 4294967295.0f;
    }

    h_seeds_xorshift[0] = x;
    h_seeds_xorshift[1] = y;
    h_seeds_xorshift[2] = z;
    h_seeds_xorshift[3] = w;
}

void populate_randn_XS_REF(float* array, int N) {
    unsigned int x, y, z, w, tmp;
    x = h_seeds_xorshift[0];
    y = h_seeds_xorshift[1];
    z = h_seeds_xorshift[2];
    w = h_seeds_xorshift[3];

    int i;
    for (i = 0; i < N; i++) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        array[i] = ((float) w) / 4294967295.0f;
    }

    for (i = 0; i < N; i += 2) {
        BoxMuller_REF(array[i], array[i + 1]);
    }

    h_seeds_xorshift[0] = x;
    h_seeds_xorshift[1] = y;
    h_seeds_xorshift[2] = z;
    h_seeds_xorshift[3] = w;
}

void populate_randIK_XS_REF(int* array, int N, int k) {
    unsigned int x, y, z, w, tmp;
    x = h_seeds_xorshift[0];
    y = h_seeds_xorshift[1];
    z = h_seeds_xorshift[2];
    w = h_seeds_xorshift[3];

    int i;
    for (i = 0; i < N; i++) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        array[i] = w % k;
    }

    h_seeds_xorshift[0] = x;
    h_seeds_xorshift[1] = y;
    h_seeds_xorshift[2] = z;
    h_seeds_xorshift[3] = w;
}

void populate_randUI_XS_REF(unsigned int* array, int N) {
    unsigned int x, y, z, w, tmp;
    x = h_seeds_xorshift[0];
    y = h_seeds_xorshift[1];
    z = h_seeds_xorshift[2];
    w = h_seeds_xorshift[3];

    int i;
    for (i = 0; i < N; i++) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        array[i] = w;
    }

    h_seeds_xorshift[0] = x;
    h_seeds_xorshift[1] = y;
    h_seeds_xorshift[2] = z;
    h_seeds_xorshift[3] = w;
}
