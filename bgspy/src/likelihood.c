#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// if you're a bit over the max boundary, we just truncate
#define MUTMAX_THRESH 1e-11

#define LOGBW_GET(lB, ii, ll, jj, kk, s) lB[(ii)*s[0] + (ll)*s[1] + (jj)*s[2] + (kk)*s[3]]
#define W_GET(WW, jj, kk, col) WW[col*(jj) + kk]

double interp_logBw(double x, double *w, double *logB, ssize_t nw,
              ssize_t i, ssize_t j, ssize_t k, ssize_t *strides) {
    double min_w = w[0];
    double max_w = w[nw-1];
    double y1, y2;
    double y;
 
    //printf("interpolation bounds: [%.3g, %.3g]\n", min_w, max_w);
    // if mutation is weak below threshold, B = 1 so we return log(1) = 0
    if (x < min_w) return 0;
    if (x > max_w && fabs(x - max_w) > MUTMAX_THRESH) {
        printf("ERROR: x=%g out past max (bounds: [%g, %g], max diff: %g)\n", 
                x, min_w, max_w, fabs(max_w - x));
    }
    if (x > max_w && fabs(x - max_w) < MUTMAX_THRESH) {
        // within the max thresh; truncate to last point
        y = LOGBW_GET(logB, i, nw-1, j, k, strides);
        return y;
    }

    for (int l=0; l < nw-1; l++) {
        if (w[l] <= x && x < w[l+1]) {
            //assert(l-1 >= 0);
            //printf("l = %d\n", l);
            y1 = LOGBW_GET(logB, i, l, j, k, strides);
            y2 = LOGBW_GET(logB, i, l+1,   j, k, strides);
            //printf("y1 = %g, y2 = %g\n", y1, y2);
            y = (y2 - y1) / (w[l+1] - w[l]) * (x - w[l]) + y1;
            //printf("y = %g", y);
            return y;
        }
    }
    printf("ERROR: interpolation failed: x=%g (bounds: [%g, %g])\n", x, min_w, max_w);
    assert(0);
    return NAN;
}

double test(double *test) {
    return test[0];
    printf("%f", test[0]);
    return 1123.8377;
}

double access(double *logB, ssize_t i, ssize_t l, ssize_t j, ssize_t k,
              ssize_t *dim, ssize_t *logB_strides) {
    ssize_t *strides = malloc(4 * sizeof(ssize_t));
    for (int i=0; i<4; i++) strides[i] = logB_strides[i] / sizeof(double);
    double out = LOGBW_GET(logB, i, l, j, k, strides);
    free(strides);
    return out;
}

double negloglik(double *theta,
                 double *nS, double *nD, 
                 double *logB, 
                 double *w,
                 ssize_t *logB_dim, 
                 ssize_t *logB_strides) {

    ssize_t nx = logB_dim[0]; 
    ssize_t nw = logB_dim[1];
    ssize_t nt = logB_dim[2];
    ssize_t nf = logB_dim[3];
    //printf("dims: nx=%d, nw=%d, nt=%d, nf=%d\n", nx, nw, nt, nf);
    double pi0 = theta[0];
    double mu = theta[1];
    ssize_t nW = (nt-1)*nf;
    double *W = calloc(nW, sizeof(double));
    memcpy(W, theta + 2, nW * sizeof(double));
    double *logBw = calloc(nx, sizeof(double));
    double Wjk;
    double ll = 0;
    ssize_t *strides = malloc(4 * sizeof(ssize_t));
    for (int i=0; i<4; i++) strides[i] = logB_strides[i] / sizeof(double);
    //for (int i=0; i<4; i++) printf("-> i=%d, %d", i, strides[i]);

    //for (int i=0; i < 50; i++) printf("   %g\n", logB[i]);

    //for (int i=0; i < 4; i++) printf(" stide %d=%d \n", i, logB_strides[i]);
    //printf("mu: %g\n", mu);

    /* for (int i=0; i < n_theta; i++) { */
    /*     printf("theta(i=%d) -> %g\n", i, theta[i]); */
    /* } */

    /* for (int i=0; i < nW; i++) { */
    /*     printf("W(i=%d) -> %g\n", i, W[i]); */
    /* } */


    double *wsum = calloc(nf, sizeof(double));
    for (int k=0; k < nf; k++) {
        for (int j=0; j < nt-1; j++)
            wsum[k] += W_GET(W, j, k, nf);
        //printf("W[%d]=%g\n", k, wsum[k]);
    }

    for (int i=0; i < nx; i++) {
        for (int j=0; j < nt; j++) {
            for (int k=0; k < nf; k++) {
                if (j == 0) {
                    // this is the fixed class of the simplex
                    Wjk = 1. - wsum[k];
                } else {
                    Wjk = W_GET(W, j-1, k, nf);
                    //if (j == 4) printf("j=%d, k=%d,offset: %d, nW=%d, W=%g W_GET=%g\n", j, k, nf*(j-1) + k, nW, W[(j-1)*nf + k], Wjk);
                }
                //printf("i=%d, j=%d, k=%d | mu=%g, Wjk=%g, mu Wjk=%g\n", i, j, k, 
                //       mu, Wjk, mu * Wjk);
                logBw[i] += interp_logBw(mu*Wjk, w, logB, nw, i, j, k, strides);
            }
        }
        //printf("%g, ", logBw[i]);
        double log_pi = log(pi0) + logBw[i];
        //printf("c log(pi0): %g\n", log(pi0));
        //printf("c nD[%d]=%g, nS[%d]=%g\n", i, nD[i], i, nS[i]);
        //printf("c llm[%d]: %g\n", i, nD[i]*log_pi + nS[i]*log(1 - exp(log_pi)));
        ll += nD[i]*log_pi + nS[i]*log1p(-exp(log_pi));
    }
    free(strides);
    free(wsum);
    free(logBw);
    return -ll;
}

