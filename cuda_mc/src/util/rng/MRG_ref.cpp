/*
 * MRG.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "MRG.h"

unsigned long h_seeds_MRG[6];

#define a_MRG 1403580UL
#define b_MRG 810728UL
#define c_MRG 527612UL
#define d_MRG 1370589UL

#define m1_MRG 4294967087UL
#define m2_MRG 4294944443UL

void seed_MRG_ref(unsigned long* seeds) {

    for (int i = 0; i < 6; i++) {
        h_seeds_MRG[i] = seeds[i];
    }
}

inline unsigned long mymodh(unsigned long x, unsigned long m) {
    if (x > m) {
        return x % m;
    } else {
        return x;
    }
}

void populate_randUI_MRG_REF(unsigned int* array, int N) {
    unsigned long y1[3] = { h_seeds_MRG[0], h_seeds_MRG[1], h_seeds_MRG[2] };
    unsigned long y2[3] = { h_seeds_MRG[3], h_seeds_MRG[4], h_seeds_MRG[5] };

    //    unsigned long mbmodm1 = m1_MRG - b_MRG;
    //    unsigned long mdmodm2 = m2_MRG - d_MRG;

    unsigned long t1;
    unsigned long t2;
    unsigned long y1_n;
    unsigned long y2_n;
    unsigned long z_n;

    for (int i = 0; i < N; i++) {
        t1 = (a_MRG * y1[1]) % m1_MRG;
        t2 = mymodh(b_MRG * y1[2], m1_MRG);
        if (t2 < t1) {
            y1_n = t1 - t2;
        } else {
            y1_n = t1 + m1_MRG - t2;
        }

        t1 = (c_MRG * y2[0]) % m2_MRG;
        t2 = mymodh(d_MRG * y2[2], m2_MRG);
        if (t2 < t1) {
            y2_n = t1 - t2;
        } else {
            y2_n = t1 + m2_MRG - t2;
        }

        y1[2] = y1[1];
        y1[1] = y1[0];
        y1[0] = y1_n;

        y2[2] = y2[1];
        y2[1] = y2[0];
        y2[0] = y2_n;

        if (y1_n > y2_n) {
            z_n = y1_n - y2_n;
        } else {
            z_n = y1_n + m1_MRG - y2_n;
        }
        if (z_n > 0) {
            array[i] = z_n;
        } else {
            array[i] = m1_MRG;
        }

    }

    h_seeds_MRG[0] = y1[0];
    h_seeds_MRG[1] = y1[1];
    h_seeds_MRG[2] = y1[2];
    h_seeds_MRG[3] = y2[0];
    h_seeds_MRG[4] = y2[1];
    h_seeds_MRG[5] = y2[2];
}

void populate_rand_MRG_REF(float* array, int N) {
    unsigned long y1[3] = { h_seeds_MRG[0], h_seeds_MRG[1], h_seeds_MRG[2] };
    unsigned long y2[3] = { h_seeds_MRG[3], h_seeds_MRG[4], h_seeds_MRG[5] };

    //    unsigned long mbmodm1 = m1_MRG - b_MRG;
    //    unsigned long mdmodm2 = m2_MRG - d_MRG;

    unsigned long t1;
    unsigned long t2;
    unsigned long y1_n;
    unsigned long y2_n;
    unsigned long z_n;

    for (int i = 0; i < N; i++) {
        t1 = (a_MRG * y1[1]) % m1_MRG;
        t2 = mymodh(b_MRG * y1[2], m1_MRG);
        if (t2 < t1) {
            y1_n = t1 - t2;
        } else {
            y1_n = t1 + m1_MRG - t2;
        }

        t1 = (c_MRG * y2[0]) % m2_MRG;
        t2 = mymodh(d_MRG * y2[2], m2_MRG);
        if (t2 < t1) {
            y2_n = t1 - t2;
        } else {
            y2_n = t1 + m2_MRG - t2;
        }

        y1[2] = y1[1];
        y1[1] = y1[0];
        y1[0] = y1_n;

        y2[2] = y2[1];
        y2[1] = y2[0];
        y2[0] = y2_n;

        if (y1_n > y2_n) {
            z_n = y1_n - y2_n;
        } else {
            z_n = y1_n + m1_MRG - y2_n;
        }
        if (z_n > 0) {
            array[i] = ((float) z_n) / (m1_MRG + 1);
        } else {
            array[i] = ((float) m1_MRG) / (m1_MRG + 1);
        }

    }

    h_seeds_MRG[0] = y1[0];
    h_seeds_MRG[1] = y1[1];
    h_seeds_MRG[2] = y1[2];
    h_seeds_MRG[3] = y2[0];
    h_seeds_MRG[4] = y2[1];
    h_seeds_MRG[5] = y2[2];
}
