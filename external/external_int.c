#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
typedef signed long long pyint;
typedef signed long long longint;
typedef double longfloat;
const int floatsize = sizeof(longfloat)*8;

typedef struct triple1 {
    longint val;
    double ral;
    int i;
} triple1;

int compare1(const void* a, const void* b) {
    triple1 t1 = * ((triple1*) a);
    triple1 t2 = * ((triple1*) b);

    if (t1.val < t2.val) return -1;
    if (t1.val > t2.val) return 1;
    if (t1.ral < t2.ral) return -1;
    if (t1.ral > t2.ral) return 1;
    return t1.i - t2.i;
}

typedef struct triple2 {
    longfloat val;
    int i;
    int j;
} triple2;

int compare2(const void* a, const void* b) {
    triple2 t1 = * ((triple2*) a);
    triple2 t2 = * ((triple2*) b);

    if (t1.val < t2.val) return -1;
    if (t1.val > t2.val) return 1;
    int i_diff = t2.i - t1.i;
    if (i_diff != 0) return i_diff;
    return t1.j - t2.j;
}

void sort1(triple1* jOrder, const pyint* limitlist, const longint* v, const double* r, int n_limitlist) {
    for (int j=0; j<n_limitlist; j++) {
        int i = (int) limitlist[j];
        triple1 t;
        t.val = -v[i];
        t.ral = -r[i];
        t.i = i;
        jOrder[j] = t;
    }
    qsort(jOrder, n_limitlist, sizeof(triple1), compare1);
}

void sort2(
        int n, triple2* tau, int* tau_size, int n_limitlist, int n_optionallist,
        const double *r, const longint* v, const longint* vr,
        const pyint* limitlist, const pyint* optionallist
) {
    int pos = 0;
    for (int i0=0; i0<n_limitlist; i0++) {
        int i = (int) limitlist[i0];
        for (int j0=i0+1; j0<n_limitlist; j0++) {
            int j = (int) limitlist[j0];
            longint denom = v[i] - v[j];
            if (denom == 0) continue;
            longint num = vr[i] - vr[j];

            triple2 t;
            t.val = (longfloat)num/(longfloat)denom;
            t.i = i;
            t.j = j;
            if (denom < 0) {
                t.i = j;
                t.j = i;
            }
            tau[pos++] = t;
        }
    }

    for (int i0=0; i0<n_optionallist; i0++) {
        int i = (int) optionallist[i0];
        triple2 t;
        t.val = r[i];
        t.i = i;
        t.j = n;
        tau[pos++] = t;
    }

    *tau_size = pos;
    qsort(tau, pos, sizeof(triple2), compare2);
}

void update(
        double* optZ, bool* optM,
        const bool* M_k, int n, int k,
        const longint* vsum, const longint* vrsum, const double* cost, double a
) {
    double z = a * log((double)vsum[k]) + (longfloat)vrsum[k]/(longfloat)vsum[k] - cost[k];
    if (z > *optZ) {
        *optZ = z;
        for (int i=0; i<n; i++) optM[i] = M_k[i];
    }
}

void iter1(
        double* optZ, bool* optM,
        longint* vsum, longint* vrsum,
        bool* M, const int n,
        const triple1* jOrder, const int n_jOrder,
        int* o, const int kmin,
        const longint* v, const longint* vr,
        const double a, const double* cost
) {
    for (int i=0; i<n_jOrder; i++) {
        int j = jOrder[i].i;
        o[j] = i + 1;
        int k = kmin + i + 1;
        vsum[k] = vsum[k-1] + v[j];
        vrsum[k] = vrsum[k-1] + vr[j];
        int base = k*n;
        for (int p=0; p<n; p++) M[base+p] = M[base+p-n];
        M[base+j] = 1;
        update(optZ, optM, &M[base], n, k, vsum, vrsum, cost, a);
    }
}

void iter2(
        double* optZ, bool* optM,
        longint* vsum, longint* vrsum,
        bool* M, const int n,
        int* o, const triple2* tau, const int tau_size,
        const longint* v, const longint* vr,
        const double a, const double* cost,
        const int kmin, const int kmax
) {

    int counter = 0;
    for (int t=0; t<tau_size; t++) {
        counter++;

        triple2 triple = tau[t];
        int i = triple.i;
        int j = triple.j;

        if (j == n) {
            for (int k=kmin; k<=kmax; k++) {
                int base = n*k;
                if (M[base+i] == 0) continue;

                M[base+i] = 0;
                vsum[k] -= v[i];
                vrsum[k] -= vr[i];
                update(optZ, optM, &M[base], n, k, vsum, vrsum, cost, a);
            }
        }
        else if (o[i] < o[j]) {
            int oi = o[i];
            int oj = o[j];
            o[j] = oi;
            o[i] = oj;
            int k = oi + kmin;
            int base = n*k;
            if (M[base+i] == 0) continue;

            M[base+i] = 0;
            M[base+j] = 1;
            vsum[k] += v[j] - v[i];
            vrsum[k] += vr[j] - vr[i];
            update(optZ, optM, &M[base], n, k, vsum, vrsum, cost, a);
        }
    }
}

void iter(
        double* optZ, bool* optM,
        const int n,
        const double* v, const double *r,
        const double a, const double* cost,
        const int kmin, const int kmax,
        const pyint* limitlist, const pyint* optionallist,
        int n_limitlist, int n_optionallist
) {
    longint num = 1; num <<= floatsize/2;
    longfloat factor = 1.0;
    for (int i=0; i<n; i++) if (v[i] > factor) factor = v[i];
    factor = (longfloat) num/factor;

    bool* M = (bool*) malloc(sizeof(bool)*(kmax + 1)*n);
    longint* vint = (longint*) malloc(sizeof(longint)*n);
    longint* vr = (longint*) malloc(sizeof(longint)*n);
    longint* vsum = (longint*) malloc(sizeof(longint)*(kmax + 1));
    longint* vrsum = (longint*) malloc(sizeof(longint)*(kmax + 1));
    int* o = (int*) malloc(sizeof(int)*n);

    longint vs = 0;
    longint vrs = 0;
    int base = kmin*n;
    for (int i=0; i<n; i++) {
        vint[i] = (longint)(v[i]*factor);
        vr[i] = (longint)(v[i]*r[i]*factor);
        M[base+i] = optM[i];

        if (optM[i] == 0) continue;
        vs += vint[i];
        vrs += vr[i];
    }
    vsum[kmin] = vs;
    vrsum[kmin] = vrs;
    *optZ = -INFINITY;

    triple1* jOrder = (triple1*) malloc(sizeof(triple1) * n_limitlist);
    sort1(jOrder, limitlist, vint, r, n_limitlist);

    iter1(optZ, optM, vsum, vrsum, M, n, jOrder, n_limitlist, o, kmin, vint, vr, a, cost);

    int tau_size = n*(n-1)/2 + n_optionallist;
    triple2* tau = (triple2*) malloc(sizeof(triple2)*tau_size);
    sort2(n, tau, &tau_size, n_limitlist, n_optionallist, r, vint, vr, limitlist, optionallist);

    iter2(optZ, optM, vsum, vrsum, M, n, o, tau, tau_size, vint, vr, a, cost, kmin, kmax);
    *optZ = *optZ - log((double) factor);

    free(M);
    free(vr);
    free(vsum);
    free(vrsum);
    free(o);
    free(jOrder);
    free(tau);
}
