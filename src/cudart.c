/*
 * CPU replacement for cudart32_30_11.dll (CUDA 3.0 runtime, 32-bit).
 *
 * RTX 50-series GPUs cannot run 32-bit CUDA at all, so this DLL implements
 * the subset of the CUDA runtime that SupersonicSled.exe imports on the CPU:
 * "device" memory is ordinary host memory (the process is 32-bit, so device
 * pointers and host pointers are the same size), kernels are the PTX from the
 * exe translated to C by tools/ptx2c.py, and D3D11 interop is done with
 * staging copies through the game's own D3D11 device.
 */
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "kernels.h"

#define API __declspec(dllexport) __stdcall

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 11, cudaErrorInvalidDeviceFunction = 8,
       cudaErrorMemoryAllocation = 2, cudaErrorInvalidResourceHandle = 33 };

typedef struct { unsigned x, y, z; } dim3;
typedef struct { int x, y, z, w, f; } cudaChannelFormatDesc;
enum { cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat };

typedef struct {
    int normalized;
    int filterMode;          /* 0 point, 1 linear */
    int addressMode[3];      /* 0 wrap, 1 clamp, 2 mirror, 3 border */
    cudaChannelFormatDesc channelDesc;
    int __cudaReserved[16];
} textureReference;

typedef struct cudaArray {
    cudaChannelFormatDesc desc;
    int width, height, elem;  /* elem = bytes per texel */
    uint8_t *data;
    int owns;
} cudaArray;

struct TexSlot {
    const char *name;
    const textureReference *ref;  /* host-side textureReference (game sets modes) */
    const cudaArray *arr;
};

/* CUDA 3.0 cudaDeviceProp, 32-bit layout */
typedef struct {
    char name[256];
    size_t totalGlobalMem, sharedMemPerBlock;
    int regsPerBlock, warpSize;
    size_t memPitch;
    int maxThreadsPerBlock, maxThreadsDim[3], maxGridSize[3], clockRate;
    size_t totalConstMem;
    int major, minor;
    size_t textureAlignment;
    int deviceOverlap, multiProcessorCount, kernelExecTimeoutEnabled, integrated,
        canMapHostMemory, computeMode, maxTexture1D, maxTexture2D[2], maxTexture3D[3],
        maxTexture2DArray[3];
    size_t surfaceAlignment;
    int concurrentKernels, ECCEnabled;
} cudaDeviceProp;

/* ------------------------------------------------------------------ log */
static FILE *g_log;
static int g_verbose = 0;
static CRITICAL_SECTION g_cs;

static void logf_(const char *fmt, ...)
{
    if (!g_log) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fflush(g_log);
}
#define LOG(...) logf_(__VA_ARGS__)
#define VLOG(...) do { if (g_verbose) logf_(__VA_ARGS__); } while (0)

/* ---------------------------------------------------- symbols / textures */
typedef struct { const void *host; const char *name; void *dev; int size; } VarReg;
static VarReg g_vars[64];
static int g_nvars;

typedef struct { const void *host; const KernelInfo *k; } FuncReg;
static FuncReg g_funcs[64];
static int g_nfuncs;

static TexSlot g_tex[16];
static int g_ntex;

static TexSlot *tex_for_ref(const textureReference *ref)
{
    for (int i = 0; i < g_ntex; i++)
        if (g_tex[i].ref == ref) return &g_tex[i];
    return NULL;
}

/* ------------------------------------------------------- texture fetch */
static inline void texel(const cudaArray *a, int x, int y, float o[4])
{
    const uint8_t *p = a->data + ((size_t)y * a->width + x) * a->elem;
    const cudaChannelFormatDesc *d = &a->desc;
    int bits[4] = { d->x, d->y, d->z, d->w };
    o[0] = o[1] = o[2] = o[3] = 0.0f;
    for (int c = 0; c < 4 && bits[c]; c++) {
        if (d->f == cudaChannelFormatKindFloat) {
            if (bits[c] == 32) { memcpy(&o[c], p, 4); p += 4; }
            else { p += bits[c] / 8; } /* half floats unused by this demo */
        } else if (bits[c] == 8) {
            o[c] = d->f == cudaChannelFormatKindSigned ? (int8_t)*p / 127.0f : *p / 255.0f; p += 1;
        } else if (bits[c] == 16) {
            uint16_t v; memcpy(&v, p, 2);
            o[c] = d->f == cudaChannelFormatKindSigned ? (int16_t)v / 32767.0f : v / 65535.0f; p += 2;
        } else {
            uint32_t v; memcpy(&v, p, 4);
            o[c] = d->f == cudaChannelFormatKindSigned ? (float)(int32_t)v : (float)v; p += 4;
        }
    }
}

static inline int addr_coord(int i, int n, int mode, int normalized)
{
    if (normalized && mode == 0) { i %= n; if (i < 0) i += n; return i; }
    if (normalized && mode == 2) {
        int period = 2 * n; i %= period; if (i < 0) i += period;
        return i < n ? i : period - 1 - i;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);  /* clamp (and border approximated) */
}

void tex_fetch2d(const TexSlot *s, float x, float y, float out[4])
{
    const cudaArray *a = s ? s->arr : NULL;
    if (!a || !a->data) { out[0] = out[1] = out[2] = out[3] = 0.0f; return; }
    const textureReference *r = s->ref;
    int norm = r->normalized, mx = r->addressMode[0], my = r->addressMode[1];
    if (norm) { x *= a->width; y *= a->height; }
    if (r->filterMode == 1) {
        x -= 0.5f; y -= 0.5f;
        float fx = floorf(x), fy = floorf(y);
        float ax = x - fx, ay = y - fy;
        int x0 = (int)fx, y0 = (int)fy;
        int xa = addr_coord(x0, a->width, mx, norm), xb = addr_coord(x0 + 1, a->width, mx, norm);
        int ya = addr_coord(y0, a->height, my, norm), yb = addr_coord(y0 + 1, a->height, my, norm);
        float t00[4], t10[4], t01[4], t11[4];
        texel(a, xa, ya, t00); texel(a, xb, ya, t10);
        texel(a, xa, yb, t01); texel(a, xb, yb, t11);
        for (int c = 0; c < 4; c++)
            out[c] = (1 - ay) * ((1 - ax) * t00[c] + ax * t10[c]) + ay * ((1 - ax) * t01[c] + ax * t11[c]);
    } else {
        int xi = addr_coord((int)floorf(x), a->width, mx, norm);
        int yi = addr_coord((int)floorf(y), a->height, my, norm);
        texel(a, xi, yi, out);
    }
}

/* ------------------------------------------------------- thread pool */
#define MAX_WORKERS 32
static int g_nworkers;
static HANDLE g_workers[MAX_WORKERS];
static HANDLE g_start_ev[MAX_WORKERS], g_done_ev[MAX_WORKERS];

static struct {
    const KernelInfo *k;
    uint8_t params[512];
    dim3 grid, block;
    volatile LONG next_block;
} g_job;

static void run_blocks(void)
{
    const uint32_t nblocks = g_job.grid.x * g_job.grid.y;
    ThreadIdx t;
    t.ntid_x = g_job.block.x; t.ntid_y = g_job.block.y;
    t.nctaid_x = g_job.grid.x; t.nctaid_y = g_job.grid.y;
    for (;;) {
        LONG b = InterlockedIncrement(&g_job.next_block) - 1;
        if ((uint32_t)b >= nblocks) break;
        t.ctaid_x = (uint32_t)b % g_job.grid.x;
        t.ctaid_y = (uint32_t)b / g_job.grid.x;
        for (uint32_t z = 0; z < g_job.block.z; z++)
            for (t.tid_y = 0; t.tid_y < g_job.block.y; t.tid_y++)
                for (t.tid_x = 0; t.tid_x < g_job.block.x; t.tid_x++)
                    g_job.k->fn(g_job.params, &t);
    }
}

static DWORD WINAPI worker_main(LPVOID arg)
{
    int i = (int)(intptr_t)arg;
    for (;;) {
        WaitForSingleObject(g_start_ev[i], INFINITE);
        run_blocks();
        SetEvent(g_done_ev[i]);
    }
    return 0;
}

static void pool_init(void)
{
    if (g_nworkers) return;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int)si.dwNumberOfProcessors - 2;  /* leave room for the game + PhysX */
    if (n < 1) n = 1;
    if (n > MAX_WORKERS) n = MAX_WORKERS;
    for (int i = 0; i < n; i++) {
        g_start_ev[i] = CreateEventA(NULL, FALSE, FALSE, NULL);
        g_done_ev[i] = CreateEventA(NULL, FALSE, FALSE, NULL);
        g_workers[i] = CreateThread(NULL, 256 * 1024, worker_main, (LPVOID)(intptr_t)i, 0, NULL);
    }
    g_nworkers = n;
    LOG("thread pool: %d workers\n", n);
}

/* --------------------------------------------------------- launch state */
static dim3 g_cfg_grid, g_cfg_block;
static uint8_t g_args[512];
static int g_args_size;
static cudaError_t g_last_error;

static const KernelInfo *find_kernel_by_name(const char *name)
{
    for (const KernelInfo *k = g_kernels; k->name; k++)
        if (!strcmp(k->name, name)) return k;
    return NULL;
}

/* ===================================================================== */
/*                        registration (host stubs)                       */
/* ===================================================================== */
API void **__cudaRegisterFatBinary(void *fatCubin)
{
    static int dummy[64];
    (void)fatCubin;
    return (void **)dummy;
}

API void __cudaUnregisterFatBinary(void **h) { (void)h; }

API void __cudaRegisterFunction(void **h, const char *hostFun, char *deviceFun, const char *deviceName,
                                int thread_limit, void *tid, void *bid, void *bDim, void *gDim, int *wSize)
{
    (void)h; (void)deviceFun; (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
    const KernelInfo *k = find_kernel_by_name(deviceName);
    LOG("register function %s -> %s\n", deviceName, k ? "ok" : "MISSING");
    if (g_nfuncs < 64) { g_funcs[g_nfuncs].host = hostFun; g_funcs[g_nfuncs].k = k; g_nfuncs++; }
}

API void __cudaRegisterVar(void **h, char *hostVar, char *deviceAddress, const char *deviceName,
                           int ext, int size, int constant, int global)
{
    (void)h; (void)deviceAddress; (void)ext; (void)constant; (void)global;
    void *mem = _aligned_malloc(size > 0 ? size : 16, 16);
    memset(mem, 0, size > 0 ? size : 16);
    for (int i = 0; g_symbol_names[i]; i++)
        if (!strcmp(g_symbol_names[i], deviceName)) *g_symbol_slots[i] = mem;
    LOG("register var %s size %d\n", deviceName, size);
    if (g_nvars < 64) { g_vars[g_nvars].host = hostVar; g_vars[g_nvars].name = deviceName;
                        g_vars[g_nvars].dev = mem; g_vars[g_nvars].size = size; g_nvars++; }
}

API void __cudaRegisterTexture(void **h, const textureReference *hostVar, const void **deviceAddress,
                               const char *deviceName, int dim, int norm, int ext)
{
    (void)h; (void)deviceAddress; (void)dim; (void)norm; (void)ext;
    if (g_ntex >= 16) return;
    TexSlot *s = &g_tex[g_ntex++];
    s->name = deviceName; s->ref = hostVar; s->arr = NULL;
    for (int i = 0; g_texture_names[i]; i++)
        if (!strcmp(g_texture_names[i], deviceName)) *g_texture_slots[i] = s;
    LOG("register texture %s dim %d\n", deviceName, dim);
}

/* ===================================================================== */
/*                               device                                   */
/* ===================================================================== */
API cudaError_t cudaGetDeviceCount(int *n) { if (n) *n = 1; return cudaSuccess; }
API cudaError_t cudaSetDevice(int d) { (void)d; pool_init(); return cudaSuccess; }

API cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int dev)
{
    (void)dev;
    memset(p, 0, sizeof(*p));
    strcpy(p->name, "CPU CUDA (Supersonic Sled shim)");
    p->totalGlobalMem = 1024u * 1024u * 1024u;
    p->sharedMemPerBlock = 48 * 1024;
    p->regsPerBlock = 32768;
    p->warpSize = 32;
    p->memPitch = 0x7fffffff;
    p->maxThreadsPerBlock = 1024;
    p->maxThreadsDim[0] = 1024; p->maxThreadsDim[1] = 1024; p->maxThreadsDim[2] = 64;
    p->maxGridSize[0] = 65535; p->maxGridSize[1] = 65535; p->maxGridSize[2] = 1;
    p->clockRate = 1400000;
    p->totalConstMem = 65536;
    p->major = 2; p->minor = 0;
    p->textureAlignment = 512;
    p->deviceOverlap = 1;
    p->multiProcessorCount = 15;
    p->canMapHostMemory = 1;
    p->maxTexture1D = 65536;
    p->maxTexture2D[0] = 65536; p->maxTexture2D[1] = 65535;
    p->maxTexture3D[0] = p->maxTexture3D[1] = p->maxTexture3D[2] = 2048;
    p->surfaceAlignment = 512;
    p->concurrentKernels = 1;
    LOG("cudaGetDeviceProperties\n");
    return cudaSuccess;
}

API cudaError_t cudaGetLastError(void) { cudaError_t e = g_last_error; g_last_error = 0; return e; }

API const char *cudaGetErrorString(cudaError_t e)
{
    static char buf[64];
    if (e == 0) return "no error";
    sprintf(buf, "CPU CUDA shim error %d", e);
    return buf;
}

API cudaError_t cudaStreamCreate(void **s) { static int dummy; if (s) *s = &dummy; return cudaSuccess; }
API cudaError_t cudaStreamDestroy(void *s) { (void)s; return cudaSuccess; }
API cudaError_t cudaStreamSynchronize(void *s) { (void)s; return cudaSuccess; }

/* ===================================================================== */
/*                               memory                                   */
/* ===================================================================== */
static void *dev_alloc(size_t n)
{
    void *p = _aligned_malloc(n ? n : 16, 256);
    if (p) memset(p, 0, n);
    return p;
}

API cudaError_t cudaMalloc(void **p, size_t n)
{
    *p = dev_alloc(n);
    VLOG("cudaMalloc %u -> %p\n", (unsigned)n, *p);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
API cudaError_t cudaMallocHost(void **p, size_t n) { return cudaMalloc(p, n); }
API cudaError_t cudaFree(void *p) { if (p) _aligned_free(p); return cudaSuccess; }
API cudaError_t cudaFreeHost(void *p) { return cudaFree(p); }

API cudaError_t cudaMallocPitch(void **p, size_t *pitch, size_t width, size_t height)
{
    size_t pt = (width + 255) & ~(size_t)255;
    *pitch = pt;
    return cudaMalloc(p, pt * height);
}

API cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, int kind)
{
    (void)kind;
    if (n && dst != src) memmove(dst, src, n);
    return cudaSuccess;
}
API cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, int kind, void *stream)
{
    (void)stream;
    return cudaMemcpy(dst, src, n, kind);
}
API cudaError_t cudaMemset(void *p, int v, size_t n) { memset(p, v, n); return cudaSuccess; }
API cudaError_t cudaMemset2D(void *p, size_t pitch, int v, size_t w, size_t h)
{
    for (size_t y = 0; y < h; y++) memset((uint8_t *)p + y * pitch, v, w);
    return cudaSuccess;
}

static VarReg *find_var(const void *symbol)
{
    for (int i = 0; i < g_nvars; i++)
        if (g_vars[i].host == symbol) return &g_vars[i];
    for (int i = 0; i < g_nvars; i++)  /* CUDA 3.0 also accepts the symbol name */
        if (!IsBadStringPtrA((const char *)symbol, 256) && !strcmp(g_vars[i].name, (const char *)symbol))
            return &g_vars[i];
    return NULL;
}

API cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset,
                                        int kind, void *stream)
{
    (void)kind; (void)stream;
    VarReg *v = find_var(symbol);
    if (!v) { LOG("cudaMemcpyToSymbolAsync: unknown symbol %p\n", symbol); return cudaErrorInvalidValue; }
    if (offset + count > (size_t)v->size) count = v->size > (int)offset ? v->size - offset : 0;
    memcpy((uint8_t *)v->dev + offset, src, count);
    return cudaSuccess;
}

/* ---------------------------------------------------------------- arrays */
API cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, int f)
{
    cudaChannelFormatDesc d = { x, y, z, w, f };
    return d;
}

static int desc_bytes(const cudaChannelFormatDesc *d) { return (d->x + d->y + d->z + d->w) / 8; }

/* CUDA 3.0 has no flags argument (stack size 16) */
API cudaError_t cudaMallocArray(cudaArray **out, const cudaChannelFormatDesc *desc, size_t w, size_t h)
{
    cudaArray *a = calloc(1, sizeof(*a));
    a->desc = *desc;
    a->width = (int)w;
    a->height = h ? (int)h : 1;
    a->elem = desc_bytes(desc);
    a->data = dev_alloc((size_t)a->width * a->height * a->elem);
    a->owns = 1;
    *out = a;
    LOG("cudaMallocArray %dx%d elem %d (%d,%d,%d,%d f%d)\n", a->width, a->height, a->elem,
        desc->x, desc->y, desc->z, desc->w, desc->f);
    return cudaSuccess;
}

API cudaError_t cudaFreeArray(cudaArray *a)
{
    if (!a) return cudaSuccess;
    if (a->owns) _aligned_free(a->data);
    free(a);
    return cudaSuccess;
}

API cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *d, const cudaArray *a)
{
    *d = a->desc;
    return cudaSuccess;
}

API cudaError_t cudaMemcpy2DToArray(cudaArray *dst, size_t wOff, size_t hOff, const void *src, size_t spitch,
                                    size_t width, size_t height, int kind)
{
    (void)kind;
    size_t dpitch = (size_t)dst->width * dst->elem;
    for (size_t y = 0; y < height; y++)
        memcpy(dst->data + (hOff + y) * dpitch + wOff, (const uint8_t *)src + y * spitch, width);
    return cudaSuccess;
}
API cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOff, size_t hOff, const void *src,
                                         size_t spitch, size_t width, size_t height, int kind, void *stream)
{
    (void)stream;
    return cudaMemcpy2DToArray(dst, wOff, hOff, src, spitch, width, height, kind);
}
API cudaError_t cudaMemcpyToArray(cudaArray *dst, size_t wOff, size_t hOff, const void *src, size_t count,
                                  int kind)
{
    (void)kind;
    size_t dpitch = (size_t)dst->width * dst->elem;
    memcpy(dst->data + hOff * dpitch + wOff, src, count);
    return cudaSuccess;
}

API cudaError_t cudaBindTextureToArray(const textureReference *tex, const cudaArray *a,
                                       const cudaChannelFormatDesc *desc)
{
    (void)desc;
    TexSlot *s = tex_for_ref(tex);
    if (!s) { LOG("bind: unknown texture %p\n", tex); return cudaErrorInvalidValue; }
    s->arr = a;
    VLOG("bind %s -> %dx%d norm %d filter %d addr %d,%d\n", s->name, a->width, a->height,
         tex->normalized, tex->filterMode, tex->addressMode[0], tex->addressMode[1]);
    return cudaSuccess;
}

API cudaError_t cudaUnbindTexture(const textureReference *tex)
{
    TexSlot *s = tex_for_ref(tex);
    if (s) s->arr = NULL;
    return cudaSuccess;
}

/* ===================================================================== */
/*                                launch                                  */
/* ===================================================================== */
API cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shmem, void *stream)
{
    (void)shmem; (void)stream;
    g_cfg_grid = grid;
    g_cfg_block = block;
    g_args_size = 0;
    return cudaSuccess;
}

API cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    if (offset + size > sizeof(g_args)) return cudaErrorInvalidValue;
    memcpy(g_args + offset, arg, size);
    if ((int)(offset + size) > g_args_size) g_args_size = (int)(offset + size);
    return cudaSuccess;
}

static LONG g_launch_count[64];

API cudaError_t cudaLaunch(const char *entry)
{
    const KernelInfo *k = NULL;
    int idx = -1;
    for (int i = 0; i < g_nfuncs; i++)
        if (g_funcs[i].host == entry) { k = g_funcs[i].k; idx = i; break; }
    if (!k) {
        LOG("cudaLaunch: unknown kernel %p\n", entry);
        g_last_error = cudaErrorInvalidDeviceFunction;
        return g_last_error;
    }
    if (g_cfg_block.z == 0) g_cfg_block.z = 1;
    if (g_cfg_grid.x == 0 || g_cfg_grid.y == 0 || g_cfg_block.x == 0 || g_cfg_block.y == 0) return cudaSuccess;
    if (g_launch_count[idx]++ == 0)
        LOG("first launch %s grid %ux%u block %ux%ux%u args %d (ptx %d)\n", k->name, g_cfg_grid.x,
            g_cfg_grid.y, g_cfg_block.x, g_cfg_block.y, g_cfg_block.z, g_args_size, k->param_size);

    pool_init();
    EnterCriticalSection(&g_cs);
    g_job.k = k;
    memcpy(g_job.params, g_args, sizeof(g_job.params));
    g_job.grid = g_cfg_grid;
    g_job.block = g_cfg_block;
    g_job.next_block = 0;
    uint32_t nblocks = g_cfg_grid.x * g_cfg_grid.y;
    uint32_t threads = nblocks * g_cfg_block.x * g_cfg_block.y * g_cfg_block.z;
    int use = threads < 2048 ? 0 : g_nworkers;
    for (int i = 0; i < use; i++) SetEvent(g_start_ev[i]);
    run_blocks();
    for (int i = 0; i < use; i++) WaitForSingleObject(g_done_ev[i], INFINITE);
    LeaveCriticalSection(&g_cs);
    return cudaSuccess;
}

/* ===================================================================== */
/*                          Direct3D 11 interop                           */
/* ===================================================================== */
static ID3D11Device *g_dev;

typedef struct {
    ID3D11Resource *res;
    D3D11_RESOURCE_DIMENSION dim;
    unsigned map_flags;          /* 0 none, 1 read only, 2 write discard */
    void *shadow;                /* buffer contents while mapped */
    size_t size;
    cudaArray arr;               /* texture contents while mapped */
    int mapped;
} GfxRes;

__declspec(dllexport) void shim_set_d3d_device(void *dev)
{
    if (dev && !g_dev) {
        g_dev = (ID3D11Device *)dev;
        ID3D11Device_AddRef(g_dev);
        LOG("D3D11 device %p\n", dev);
    }
}

API cudaError_t cudaD3D11SetDirect3DDevice(ID3D11Device *dev)
{
    shim_set_d3d_device(dev);
    return cudaSuccess;
}

API cudaError_t cudaGraphicsD3D11RegisterResource(GfxRes **out, ID3D11Resource *res, unsigned flags)
{
    (void)flags;
    GfxRes *g = calloc(1, sizeof(*g));
    g->res = res;
    ID3D11Resource_AddRef(res);
    ID3D11Resource_GetType(res, &g->dim);
    if (!g_dev) ID3D11Resource_GetDevice(res, &g_dev);
    *out = g;
    LOG("register D3D resource %p type %d\n", res, g->dim);
    return cudaSuccess;
}

API cudaError_t cudaGraphicsUnregisterResource(GfxRes *g)
{
    if (!g) return cudaSuccess;
    ID3D11Resource_Release(g->res);
    if (g->shadow) _aligned_free(g->shadow);
    if (g->arr.data) _aligned_free(g->arr.data);
    free(g);
    return cudaSuccess;
}

API cudaError_t cudaGraphicsResourceSetMapFlags(GfxRes *g, unsigned flags)
{
    g->map_flags = flags;
    return cudaSuccess;
}

static int dxgi_to_desc(DXGI_FORMAT f, cudaChannelFormatDesc *d)
{
    cudaChannelFormatDesc z = { 0, 0, 0, 0, cudaChannelFormatKindFloat };
    *d = z;
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: d->x = d->y = d->z = d->w = 32; return 16;
    case DXGI_FORMAT_R32G32_FLOAT: d->x = d->y = 32; return 8;
    case DXGI_FORMAT_R32_FLOAT: d->x = 32; return 4;
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        d->x = d->y = d->z = d->w = 8; d->f = cudaChannelFormatKindUnsigned; return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: d->x = d->y = d->z = d->w = 16; return 8;
    default: LOG("unhandled DXGI format %d\n", f); d->x = 32; return 4;
    }
}

/* copy a GPU resource into CPU memory via a staging resource */
static void *readback(GfxRes *g, size_t *size_out, UINT *row_pitch, D3D11_TEXTURE2D_DESC *td_out)
{
    ID3D11DeviceContext *ctx;
    ID3D11Device_GetImmediateContext(g_dev, &ctx);
    ID3D11Resource *stg = NULL;
    void *out = NULL;
    if (g->dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
        D3D11_BUFFER_DESC bd;
        ID3D11Buffer_GetDesc((ID3D11Buffer *)g->res, &bd);
        bd.Usage = D3D11_USAGE_STAGING; bd.BindFlags = 0; bd.MiscFlags = 0;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Device_CreateBuffer(g_dev, &bd, NULL, (ID3D11Buffer **)&stg);
        *size_out = bd.ByteWidth;
    } else {
        D3D11_TEXTURE2D_DESC td;
        ID3D11Texture2D_GetDesc((ID3D11Texture2D *)g->res, &td);
        *td_out = td;
        td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.MiscFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MipLevels = 1; td.ArraySize = 1;
        td.SampleDesc.Count = 1; td.SampleDesc.Quality = 0;
        ID3D11Device_CreateTexture2D(g_dev, &td, NULL, (ID3D11Texture2D **)&stg);
    }
    if (stg) {
        if (g->dim == D3D11_RESOURCE_DIMENSION_BUFFER)
            ID3D11DeviceContext_CopyResource(ctx, stg, g->res);
        else
            ID3D11DeviceContext_CopySubresourceRegion(ctx, stg, 0, 0, 0, 0, g->res, 0, NULL);
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(ID3D11DeviceContext_Map(ctx, stg, 0, D3D11_MAP_READ, 0, &m))) {
            if (g->dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
                out = _aligned_malloc(*size_out, 256);
                memcpy(out, m.pData, *size_out);
            } else {
                *row_pitch = m.RowPitch;
                *size_out = (size_t)m.RowPitch * td_out->Height;
                out = _aligned_malloc(*size_out, 256);
                memcpy(out, m.pData, *size_out);
            }
            ID3D11DeviceContext_Unmap(ctx, stg, 0);
        }
        ID3D11Resource_Release(stg);
    }
    ID3D11DeviceContext_Release(ctx);
    return out;
}

API cudaError_t cudaGraphicsMapResources(int count, GfxRes **res, void *stream)
{
    (void)stream;
    for (int i = 0; i < count; i++) {
        GfxRes *g = res[i];
        g->mapped = 1;
        if (g->dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
            D3D11_BUFFER_DESC bd;
            ID3D11Buffer_GetDesc((ID3D11Buffer *)g->res, &bd);
            if (!g->shadow || g->size != bd.ByteWidth) {
                if (g->shadow) _aligned_free(g->shadow);
                g->size = bd.ByteWidth;
                g->shadow = dev_alloc(g->size);
            }
            /* keep the previous contents unless the game said it overwrites everything */
            if (g->map_flags != 2 && !(bd.Usage == D3D11_USAGE_DYNAMIC)) {
                size_t n; UINT rp; D3D11_TEXTURE2D_DESC td;
                void *data = readback(g, &n, &rp, &td);
                if (data) { memcpy(g->shadow, data, n < g->size ? n : g->size); _aligned_free(data); }
            }
        } else if (g->dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
            size_t n; UINT rp = 0; D3D11_TEXTURE2D_DESC td;
            void *data = readback(g, &n, &rp, &td);
            if (g->arr.data) _aligned_free(g->arr.data);
            g->arr.elem = dxgi_to_desc(td.Format, &g->arr.desc);
            g->arr.width = td.Width;
            g->arr.height = td.Height;
            g->arr.data = dev_alloc((size_t)td.Width * td.Height * g->arr.elem);
            if (data) {
                for (UINT y = 0; y < td.Height; y++)
                    memcpy(g->arr.data + (size_t)y * td.Width * g->arr.elem, (uint8_t *)data + (size_t)y * rp,
                           (size_t)td.Width * g->arr.elem);
                _aligned_free(data);
            }
            VLOG("mapped texture %ux%u fmt %d\n", td.Width, td.Height, td.Format);
        }
    }
    return cudaSuccess;
}

API cudaError_t cudaGraphicsUnmapResources(int count, GfxRes **res, void *stream)
{
    (void)stream;
    ID3D11DeviceContext *ctx;
    ID3D11Device_GetImmediateContext(g_dev, &ctx);
    for (int i = 0; i < count; i++) {
        GfxRes *g = res[i];
        g->mapped = 0;
        if (g->map_flags == 1) continue;  /* read only */
        if (g->dim == D3D11_RESOURCE_DIMENSION_BUFFER && g->shadow) {
            D3D11_BUFFER_DESC bd;
            ID3D11Buffer_GetDesc((ID3D11Buffer *)g->res, &bd);
            if (bd.Usage == D3D11_USAGE_DYNAMIC) {
                D3D11_MAPPED_SUBRESOURCE m;
                if (SUCCEEDED(ID3D11DeviceContext_Map(ctx, g->res, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                    memcpy(m.pData, g->shadow, g->size);
                    ID3D11DeviceContext_Unmap(ctx, g->res, 0);
                }
            } else {
                ID3D11DeviceContext_UpdateSubresource(ctx, g->res, 0, NULL, g->shadow, 0, 0);
            }
        } else if (g->dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D && g->arr.data) {
            ID3D11DeviceContext_UpdateSubresource(ctx, g->res, 0, NULL, g->arr.data,
                                                  g->arr.width * g->arr.elem, 0);
        }
    }
    ID3D11DeviceContext_Release(ctx);
    return cudaSuccess;
}

API cudaError_t cudaGraphicsResourceGetMappedPointer(void **ptr, size_t *size, GfxRes *g)
{
    *ptr = g->shadow;
    if (size) *size = g->size;
    return g->shadow ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

API cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray **arr, GfxRes *g, unsigned idx, unsigned level)
{
    (void)idx; (void)level;
    *arr = &g->arr;
    return cudaSuccess;
}

/* ===================================================================== */
BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved)
{
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        InitializeCriticalSection(&g_cs);
        char path[MAX_PATH];
        GetModuleFileNameA(inst, path, MAX_PATH);
        char *slash = strrchr(path, '\\');
        if (slash) slash[1] = 0;
        strcat(path, "cpu_cuda_log.txt");
        g_log = fopen(path, "w");
        g_verbose = GetEnvironmentVariableA("SLED_CUDA_VERBOSE", NULL, 0) > 0;
        LOG("CPU CUDA shim loaded\n");
    }
    return TRUE;
}
