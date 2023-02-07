#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// if you're a bit over the max boundary, we just truncate
#define MUTMAX_THRESH 1e-10

#define LOGBW_GET(B, ii, ll, jj, kk, s) B[(ii)*s[0] + (ll)*s[1] + (jj)*s[2] + (kk)*s[3]]
#define W_GET(WW, jj, kk, col) WW[col*(jj) + kk]

void print_Bw(const double *logB, ssize_t i, 
              ssize_t j, ssize_t k, ssize_t nw,
              ssize_t *logB_strides) {
    ssize_t *strides = malloc(4 * sizeof(ssize_t));
    for (ssize_t i=0; i<4; i++) strides[i] = logB_strides[i] / sizeof(double);
 
    printf("w = [");
    for (ssize_t l=0; l<nw; l++) {
        double y = LOGBW_GET(logB, i, l, j, k, strides);
        if (l < nw-1) 
            printf("%g (%ld), ", y, l);
        else
            printf("%g (%ld)", y, l);
    }
    printf("]");
    free(strides);
}

double access(double *logB, ssize_t i, ssize_t l, ssize_t j, ssize_t k,
              ssize_t *logB_strides) {
    ssize_t *strides = malloc(4 * sizeof(ssize_t));
    for (ssize_t i=0; i<4; i++) strides[i] = logB_strides[i] / sizeof(double);
    double out = LOGBW_GET(logB, i, l, j, k, strides);
    free(strides);
    return out;
}

double access2(double *logB, ssize_t i, ssize_t l, ssize_t j, ssize_t k,
              ssize_t *strides) {
    double out = LOGBW_GET(logB, i, l, j, k, strides);
    return out;
}


double interp_logBw(const double x, const double *w, const double *logB, 
                    ssize_t nw, ssize_t i, ssize_t j,
                    ssize_t k, 
                    const ssize_t *logB_strides) {
    double min_w = w[0];
    double max_w = w[nw-1];
    double y1, y2;
    double y;
    ssize_t *strides = malloc(4 * sizeof(ssize_t));
    for (int i=0; i<4; i++) strides[i] = logB_strides[i] / sizeof(double);
 
    //print_Bw(logB, i, j, k, nw, strides); printf("\n");
    //printf("interpolation bounds: [%.3g, %.3g]\n", min_w, max_w);
    if (x <= min_w) {
        // if mutation is below or equal to the lower threshold return 
        // *zero*, which in log-space is B=1, e.g. no reduction (since 
        // mutation is sufficiently weak, no effect)
        //y = LOGBW_GET(logB, i, 0, j, k, strides);
        free(strides);
        return 0;
    }
    if (x >= max_w && fabs(x - max_w) < MUTMAX_THRESH) {
        // within the max thresh; truncate to last point
        y = LOGBW_GET(logB, i, nw-1, j, k, strides);
        free(strides);
        return y;
    }
    if (x > max_w && fabs(x - max_w) > MUTMAX_THRESH) {
        printf("ERROR: x=%g out past max (with tolerance %g) (bounds: [%g, %g], max diff: %g)\n", 
                x, MUTMAX_THRESH, min_w, max_w, fabs(max_w - x));
        free(strides);
        return NAN;
    }
    for (ssize_t l=0; l < nw-1; l++) {
        //printf("nw = %d, w[%d] = %g, w[%d] = %g l = %d\n", nw, l, w[l], l+1, w[l+1], l, i, j, k);
            //y1 = LOGBW_GET(logB, i, l,   j, k, strides);
            //y2 = LOGBW_GET(logB, i, l+1, j, k, strides);
            //printf("x = %g, y1 = %g, y2 = %g\n", x, y1, y2);
            //y = (y2 - y1) / (w[l+1] - w[l]) * (x - w[l]) + y1;
            //printf("x = %g, y = %g\n", x, y);
 
        if ((w[l] <= x) && (x < w[l+1])) {
            //assert(l-1 >= 0);
            //printf("l = %d\n", l);
            //printf("***nw = %d, w[%d] = %g, w[%d] = %g l = %d | i = %d, j = %d, k = %d\n", nw, l, w[l], l+1, w[l+1], l, i, j, k);
            //printf("***nw = %ld, w[%ld] = %g, w[%ld] = %g l = %ld | i = %ld, j = %ld, k = %ld\n", nw, l, w[l], l+1, w[l+1], l, i, j, k);
            //y1 = access2(logB, i, l,   j, k, strides);
            //y2 = access2(logB, i, l+1, j, k, strides);
            y1 = LOGBW_GET(logB, i, l,   j, k, strides);
            y2 = LOGBW_GET(logB, i, l+1, j, k, strides);
            /* if (y1 == 0 | y2 == 0) { */ 
                /* printf("***nw = %d, w[%d] = %g, w[%d] = %g l = %d | i = %d, j = %d, k = %d\n", nw, l, w[l], l+1, w[l+1], l, i, j, k); */
            //printf("x = %g, y1 = %g, y2 = %g\n", x, y1, y2);
            //printf("x = %g, y1 = %g, y2 = %g\n", x, y1, y2);
            y = (y2 - y1) / (w[l+1] - w[l]) * (x - w[l]) + y1;
            //printf("x = %g, y = %g\n", x, y);
            //printf("y = %g", y);
            free(strides);
            return y;
        }
    }
    printf("ERROR: interpolation reached end: x=%g (bounds: [%g, %g], max diff: %g)\n", x, min_w, max_w, fabs(max_w - x));
    free(strides);
    assert(0);
    return NAN;
}

void print_theta(const double *theta, ssize_t n) {
    for (ssize_t i=0; i < n; i++) {
        if (i < n-1) 
            printf("%g, ", theta[i]);
        else
            printf("%g", theta[i]);
    }
}

double negloglik(const double *theta,
                 const double *nS, const double *nD, 
                 const double *logB, 
                 const double *w,
                 const ssize_t *logB_dim, 
                 const ssize_t *logB_strides,
                 const int two_alleles) {

    ssize_t nx = logB_dim[0]; 
    ssize_t nw = logB_dim[1];
    ssize_t nt = logB_dim[2];
    ssize_t nf = logB_dim[3];

    double pi0 = theta[0];
    double mu = theta[1];
    ssize_t nW = nt*nf;
    double *W = calloc(nW, sizeof(double));
    memcpy(W, theta + 2, nW * sizeof(double));
    double logBw_i;
    double Wjk;
    double ll = 0;

    // iterate through all windows, summing B values
    // across features and selection coefficients per feature
    //print_theta(theta, 2+nW);
    for (ssize_t i=0; i < nx; i++) {
        logBw_i = 0.; // initialize start of sum
        for (ssize_t j=0; j < nt; j++) {
            for (ssize_t k=0; k < nf; k++) {
                Wjk = W_GET(W, j, k, nf);
                if (isnan(Wjk)) {
                    printf("ERROR: W_GET(W, j=%ld, k=%ld, nf=%ld) is returning NaN\n", 
                            j, k, nf);
                }
                //if (j == 4) printf("j=%d, k=%d,offset: %d, nW=%d, W=%g W_GET=%g\n", j, k, nf*(j-1) + k, nW, W[(j-1)*nf + k], Wjk);
                //printf("i=%d, j=%d, k=%d | mu=%g, Wjk=%g, mu Wjk=%g\n", i, j, k, 
                //       mu, Wjk, mu * Wjk);
                //
                // get the B value for this feature and sel coef, 
                // by interpolating the B value for this mu x DFE
                double Binc = interp_logBw(mu*Wjk, w, logB, nw, i, j, k, logB_strides);
                if (isnan(Binc)) {
                    printf("NaN Binc! theta=[");
                    print_theta(theta, 2+nW);
                    printf("]");
                    printf("i=%ld, j=%ld, k=%ld | Wjk=%g,", i, j, k, Wjk);
                    free(W);
                    return NAN;
                }
                logBw_i += Binc;
            }
        }

        double log_pi;
        if (two_alleles) {
            double pi = pi0 * exp(logBw_i);
            log_pi = log(pi / (1 + 2*pi));
            /* printf("pi = %f, pi_2a =  %f\n", pi, exp(log_pi)); */
        } else {
            log_pi = log(pi0) + logBw_i;
        }
        //printf("c log(pi0): %g\n", log(pi0));
        //printf("c nD[%d]=%g, nS[%d]=%g\n", i, nD[i], i, nS[i]);
        //printf("c llm[%d]: %g\n", i, nD[i]*log_pi + nS[i]*log(1 - exp(log_pi)));
        
        // compute the final likelihood for this window
        double obs_pi = nD[i] / (nS[i] + nD[i]);
        double pi = exp(log_pi);
        //ll += pow(obs_pi - pi, 2.);
        ll += pow(obs_pi - pi, 2.);
    }
    free(W);
    fflush(stdout);
    return -ll;
}

double negloglik2(const double *theta,
                 const double *nS, const double *nD, 
                 const double *logB, 
                 const double *w,
                 const ssize_t *logB_dim, 
                 const ssize_t *logB_strides,
                 const int two_alleles) {

    ssize_t nx = logB_dim[0]; 
    ssize_t nw = logB_dim[1];
    ssize_t nt = logB_dim[2];
    ssize_t nf = logB_dim[3];

    double pi0 = theta[0];
    double mu = theta[1];
    ssize_t nW = nt*nf;
    double *W = calloc(nW, sizeof(double));
    memcpy(W, theta + 2, nW * sizeof(double));
    double logBw_i;
    double Wjk;
    double ll = 0;
    double log_pi;

    // variables we need for interpolation in this loop
    double y1, y2;
    double min_w = w[0];
    double max_w = w[nw-1];
    ssize_t *strides = malloc(4 * sizeof(ssize_t));
    for (int i=0; i<4; i++) strides[i] = logB_strides[i] / sizeof(double);
 
    // Iterate through all windows, summing B values across features and
    // selection coefficients per feature
    //
    // Note that for each w=μW for a feature/selection combination has the same
    // index, so this can be found once for a combination and re-used
   
    // array (simulating matrix) of the w indices for interpolation
    int *w_idx = (int *)malloc(nt * nf * sizeof(int));

    //int offset = i * nf + j;
    // Now, get the right indices *once* for this theta.
    int offset;
    for (ssize_t j=0; j < nt; j++) {
        for (ssize_t k=0; k < nf; k++) {
            Wjk = W_GET(W, j, k, nf);
            offset = j*nf + k;
            double x = mu*Wjk;
            if (x <= min_w) {
                w_idx[offset] = -1;
                break;
            } else if (x >= max_w) {
                if (x > max_w && fabs(x - max_w) > MUTMAX_THRESH) {
                    printf("ERROR: x=%g out past max (with tolerance %g) (bounds: [%g, %g], max diff: %g)\n", 
                            x, MUTMAX_THRESH, min_w, max_w, fabs(max_w - x));
                    free(strides);
                    return NAN;
                }
                w_idx[offset] = -2;
                break;
            } else {
                for (ssize_t l=0; l < nw-1; l++) {
                    // this is a O(n) search, we could use bisect to make this
                    // faster
                    if ((w[l] <= x) && (x < w[l+1])) {
                        w_idx[offset] = l;
                        break;
                    }
                }
            }
        }
    }

    double log_pi0 = log(pi0);
    for (ssize_t i=0; i < nx; i++) {
        logBw_i = 0.; // initialize start of sum
        for (ssize_t j=0; j < nt; j++) {
            for (ssize_t k=0; k < nf; k++) {
                Wjk = W_GET(W, j, k, nf);
                double x = mu*Wjk;

                offset = j*nf + k;
                int wi = w_idx[offset];
                if (wi == -1) {
                    // we're past the lowest mutation rate; assume that
                    // mutation and the DFE are so weak that there's no
                    // reduction, e.g. log(B) = 0, so do nothing
                } else if (wi == -2) {
                    // we're past the end, use the endpoint B
                    // TODO check for far out out of bounds
                    logBw_i += LOGBW_GET(logB, i, nw-1, j, k, strides);
                } else {
                    //assert(w_idx >= 0);
                    //assert(w_idx < nw);
                    y1 = LOGBW_GET(logB, i, wi,   j, k, strides);
                    y2 = LOGBW_GET(logB, i, wi+1, j, k, strides);
                    logBw_i += (y2 - y1) / (w[wi+1] - w[wi]) * (x - w[wi]) + y1;
                }
                //double Binc = interp_logBw(x, w, logB, nw, i, j, k, logB_strides);
                //printf("new: %g, old: %g\n", logBw_i, Binc);
            }
        }
        log_pi = log_pi0 + logBw_i;
        ll += nD[i]*log_pi + nS[i]*log1p(-exp(log_pi));
    }

    free(w_idx);
    free(W);
    free(strides);
    fflush(stdout);
    return -ll;
}

