#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LOGBW_GET(lB, ii, ll, jj, kk, s) lB[ii*s[0] + ll*s[1] + jj*s[2] + kk*s[3]]
#define W_GET(theta, jj, kk, nf) theta[2 + nf*jj + kk]

double interp_logBw(double x, double *w, double *logB, ssize_t nw,
              ssize_t i, ssize_t j, ssize_t k, ssize_t *logB_strides) {
    double min_w = w[0];
    double max_w = w[nw-1];
    printf("interpolation bounds: [%.3g, %.3g]\n", min_w, max_w);
    // if mutation is weak below threshold, B = 1 so we return log(1) = 0
    if (x < min_w) return 0;
    assert(x < max_w);
    double y1, y2;
    for (int l=1; l < nw; l++) {
        if (w[l-1] <= x && x < w[l]) {
            y1 = LOGBW_GET(logB, i, l-1, j, k, logB_strides);
            y2 = LOGBW_GET(logB, i, l, j, k, logB_strides);
            return (y2 - y1) / (w[i] - w[i-1]) * (x - w[i-1]) + y1;
        }
    }
    printf("ERROR: interpolation failed: x=%f (bounds: [%f, %f])\n", x, w[0], w[nw-1]);
    //assert(0);
    return 1.0;
}

double test(double *test) {
    return test[0];
    printf("%f", test[0]);
    return 1123.8377;
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
    printf("dims: nx=%d, nw=%d, nt=%d, nf=%d\n", nx, nw, nt, nf);
    double pi0 = theta[0];
    double *logBw = calloc(nx, sizeof(double));
    double Wjk;
    double ll;
    double mu = theta[1];

    printf("mu: %g\n", mu);

    ssize_t n_theta = 2 + (nt-1)*nf;
    for (int i=0; i < n_theta; i++) {
        printf("theta(i=%d) -> %g\n", i, theta[i]);
    }

    double *wsum = calloc(nf, sizeof(double));
    for (int k=0; k < nf; k++) {
        for (int j=0; j < nt-1; j++)
            wsum[k] += W_GET(theta, j, k, nf);
        printf("W[%d]=%g\n", k, wsum[k]);
    }

    for (int i=0; i < nx; i++) {
        for (int j=0; j < nt; j++) {
            for (int k=0; k < nf; k++) {
                if (j == 0) {
                    // this is the fixed class of the simplex
                    Wjk = 1 - wsum[k];
                } else {
                    Wjk = W_GET(theta, j-1, k, nf);
                }
                printf("mu * Wjk: %g\n", mu * Wjk);
                logBw[i] += interp_logBw(mu*Wjk, w, logB, nw, i, j, k, logB_strides);
            }
        }
        double log_pi = log(pi0) + logBw[i];
        ll += nD[i]*log_pi + nS[i]*log1p(exp(-log_pi));
    }
    return -ll;
}




