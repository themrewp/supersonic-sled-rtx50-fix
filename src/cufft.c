/*
 * CPU replacement for cufft32_30_11.dll: just the 2D real<->complex
 * transforms the Supersonic Sled smoke fluid uses (128x128).
 * Matches cuFFT semantics: unnormalized, R2C output is nx * (ny/2+1)
 * complex values; in-place transforms use padded real rows.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define API __declspec(dllexport) __stdcall
typedef struct { float re, im; } cpx;
enum { CUFFT_R2C = 0x2a, CUFFT_C2R = 0x2c, CUFFT_C2C = 0x29 };

typedef struct { int nx, ny, type; cpx *work, *col; } Plan;
#define MAX_PLANS 64
static Plan *g_plans[MAX_PLANS];

/* in-place iterative radix-2 FFT; sign -1 forward, +1 inverse; falls back to DFT for non powers of two */
static void fft1d(cpx *a, int n, int stride, int sign, cpx *tmp)
{
    for (int i = 0; i < n; i++) tmp[i] = a[i * stride];
    if ((n & (n - 1)) == 0) {
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { cpx t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = sign * 2.0 * 3.14159265358979323846 / len;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < len / 2; k++) {
                    float wr = (float)cos(ang * k), wi = (float)sin(ang * k);
                    cpx u = tmp[i + k], v = tmp[i + k + len / 2];
                    cpx vw = { v.re * wr - v.im * wi, v.re * wi + v.im * wr };
                    tmp[i + k].re = u.re + vw.re; tmp[i + k].im = u.im + vw.im;
                    tmp[i + k + len / 2].re = u.re - vw.re; tmp[i + k + len / 2].im = u.im - vw.im;
                }
        }
        for (int i = 0; i < n; i++) a[i * stride] = tmp[i];
    } else {
        for (int k = 0; k < n; k++) {
            double sr = 0, si = 0;
            for (int j = 0; j < n; j++) {
                double ang = sign * 2.0 * 3.14159265358979323846 * j * k / n;
                sr += tmp[j].re * cos(ang) - tmp[j].im * sin(ang);
                si += tmp[j].re * sin(ang) + tmp[j].im * cos(ang);
            }
            a[k * stride].re = (float)sr; a[k * stride].im = (float)si;
        }
    }
}

API int cufftPlan2d(int *plan, int nx, int ny, int type)
{
    for (int i = 1; i < MAX_PLANS; i++)
        if (!g_plans[i]) {
            Plan *p = calloc(1, sizeof(Plan));
            p->nx = nx; p->ny = ny; p->type = type;
            p->work = calloc((size_t)nx * ny, sizeof(cpx));
            p->col = calloc((size_t)(nx > ny ? nx : ny), sizeof(cpx));
            g_plans[i] = p;
            *plan = i;
            return 0;
        }
    return 2; /* CUFFT_ALLOC_FAILED */
}

API int cufftDestroy(int plan)
{
    if (plan > 0 && plan < MAX_PLANS && g_plans[plan]) {
        free(g_plans[plan]->work); free(g_plans[plan]->col); free(g_plans[plan]);
        g_plans[plan] = NULL;
    }
    return 0;
}

/* data layout: nx rows (slowest), ny columns (fastest) */
API int cufftExecR2C(int plan, float *in, cpx *out)
{
    Plan *p = g_plans[plan];
    int nx = p->nx, ny = p->ny, nh = ny / 2 + 1;
    int rstride = ((void *)in == (void *)out) ? 2 * nh : ny;
    cpx *w = p->work;
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) { w[x * ny + y].re = in[x * rstride + y]; w[x * ny + y].im = 0; }
        fft1d(w + x * ny, ny, 1, -1, p->col);
    }
    for (int y = 0; y < nh; y++) fft1d(w + y, nx, ny, -1, p->col);
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < nh; y++) out[x * nh + y] = w[x * ny + y];
    return 0;
}

API int cufftExecC2R(int plan, cpx *in, float *out)
{
    Plan *p = g_plans[plan];
    int nx = p->nx, ny = p->ny, nh = ny / 2 + 1;
    int rstride = ((void *)in == (void *)out) ? 2 * nh : ny;
    cpx *w = p->work;
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < nh; y++) w[x * ny + y] = in[x * nh + y];
    for (int y = 0; y < nh; y++) fft1d(w + y, nx, ny, +1, p->col);
    for (int x = 0; x < nx; x++) {
        cpx *row = w + x * ny;
        for (int y = nh; y < ny; y++) { row[y].re = row[ny - y].re; row[y].im = -row[ny - y].im; }
        fft1d(row, ny, 1, +1, p->col);
        for (int y = 0; y < ny; y++) out[x * rstride + y] = row[y].re;
    }
    return 0;
}
