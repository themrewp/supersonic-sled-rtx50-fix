/* Shared declarations for the CPU-translated CUDA kernels. */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t tid_x, tid_y, ntid_x, ntid_y;
    uint32_t ctaid_x, ctaid_y, nctaid_x, nctaid_y;
} ThreadIdx;

typedef struct TexSlot TexSlot;
void tex_fetch2d(const TexSlot *slot, float x, float y, float out[4]);

typedef void (*KernelFn)(const uint8_t *params, const ThreadIdx *t);
typedef struct {
    const char *name;
    KernelFn fn;
    int param_size;
    int nparams;
    int offs[48];
} KernelInfo;

extern const KernelInfo g_kernels[];
extern const char *const g_symbol_names[];
extern void **const g_symbol_slots[];
extern const char *const g_texture_names[];
extern TexSlot **const g_texture_slots[];

static inline float F(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t UF(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline double D(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static inline uint64_t UD(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static inline float sat01(float f) { return f != f ? 0.0f : (f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f)); }

/* PTX float->int conversions saturate and map NaN to 0 */
static inline int32_t cvt_f2s32(float f) {
    if (f != f) return 0;
    if (f >= 2147483647.0f) return 2147483647;
    if (f <= -2147483648.0f) return (int32_t)0x80000000u;
    return (int32_t)f;
}
static inline uint32_t cvt_f2u32(float f) {
    if (f != f || f <= 0.0f) return 0;
    if (f >= 4294967295.0f) return 0xFFFFFFFFu;
    return (uint32_t)f;
}
static inline int16_t cvt_f2s16(float f) { int32_t v = cvt_f2s32(f); return (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v)); }
static inline uint16_t cvt_f2u16(float f) { uint32_t v = cvt_f2u32(f); return (uint16_t)(v > 65535 ? 65535 : v); }
