/*
 * Minimal nvcuda.dll for SupersonicSled.exe, which imports two driver API
 * calls directly. Placed next to the exe so it wins over the system copy
 * (which has no 32-bit CUDA device on RTX 50-series anyway).
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define API __declspec(dllexport) __stdcall

void shim_set_d3d_device(void *dev);  /* from cudart32_30_11.dll */
typedef void (*SetDev)(void *);

API int cuD3D11CtxCreate(void **ctx, int *dev, unsigned flags, void *d3d)
{
    (void)flags;
    static int dummy;
    if (ctx) *ctx = &dummy;
    if (dev) *dev = 0;
    HMODULE m = GetModuleHandleA("cudart32_30_11.dll");
    SetDev f = m ? (SetDev)GetProcAddress(m, "shim_set_d3d_device") : NULL;
    if (f) f(d3d);
    return 0;
}
API int cuCtxDestroy(void *ctx) { (void)ctx; return 0; }
/* anything else (e.g. PhysX probing for a GPU) sees "no device" */
API int cuInit(unsigned flags) { (void)flags; return 100; }
API int cuDeviceGetCount(int *n) { if (n) *n = 0; return 0; }
API int cuDriverGetVersion(int *v) { if (v) *v = 3000; return 0; }
