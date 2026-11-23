/*
 * PlayKit frame multiplier (pass-through prototype)
 *
 * For every input frame pair, emit `multi` output frames.  The prototype
 * duplicates the current frame and assigns evenly spaced PTS values.  It is
 * meant to validate D3D11 frame handling, PTS interpolation, queueing and EOF
 * before the real DLSS-G / aji zero-copy path is plugged in.
 */

#include <math.h>
#include <stdio.h>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>

#include "common/common.h"
#include "common/msg.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/frame.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "video/hwdec.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/mp_image_pool.h"
#include "video/sws_utils.h"

static double pk_now_ms(void)
{
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return 1000.0 * (double)now.QuadPart / (double)freq.QuadPart;
}

struct opts {
    int multi;
    bool log;
    int outfmt;
    char *dll;
    char *proxy_dir;
    int matrix;
    bool full_range;
};

#define OPT_BASE_STRUCT struct opts

static const m_option_t vf_opts_fields[] = {
    {"multi", OPT_INT(multi)},
    {"log", OPT_BOOL(log)},
    {"outfmt", OPT_IMAGEFORMAT(outfmt)},
    {"dll", OPT_STRING(dll)},
    {"proxy_dir", OPT_STRING(proxy_dir)},
    {"matrix", OPT_INT(matrix)},
    {"full_range", OPT_BOOL(full_range)},
    {0}
};

struct pk_bridge;

struct priv {
    struct opts *opts;
    struct mp_sws_context *sws;

    struct mp_image *prev;
    struct mp_image *queue[6];
    int queue_len;
    int queue_pos;

    bool eof_pending;
    bool logged;
    bool bridge_failed;
    int out_count;

    struct pk_bridge *bridge;
};

static void clear_queue(struct mp_filter *f)
{
    struct priv *p = f->priv;

    for (int i = 0; i < p->queue_len; i++)
        mp_image_unrefp(&p->queue[i]);
    p->queue_len = 0;
    p->queue_pos = 0;
}

static void reset(struct mp_filter *f)
{
    struct priv *p = f->priv;

    clear_queue(f);
    p->eof_pending = false;
    p->logged = false;
    p->bridge_failed = false;
    p->out_count = 0;
    mp_image_unrefp(&p->prev);
}

struct pk_stats {
    double wall_ms;
    double evaluate_ms;
    double gpu_ms;
    int suppressed;
};

struct pk_out_view {
    ID3D11Texture2D *tex;
    UINT sub;
    ID3D11VideoProcessorOutputView *view;
};

struct pk_bridge {
    HMODULE proxy;
    HMODULE dll;
    void *(*create)(uint32_t, int32_t, uint32_t, uint32_t, int, int, int,
                    const wchar_t *, char *, size_t);
    int (*process)(void *, void *, void *, void **, uint32_t, uint32_t,
                   struct pk_stats *, char *, size_t);
    void (*destroy)(void *);
    void *handle;

    ID3D11Device *dev;
    ID3D11DeviceContext *ctx;
    ID3D11VideoDevice *video_dev;
    ID3D11VideoContext *video_ctx;
    ID3D11VideoProcessorEnumerator *vp_enum;
    ID3D11VideoProcessor *vp;
    ID3D11Texture2D *in_tex[2];
    HANDLE in_handle[2];
    ID3D11Texture2D *out_tex[5];
    HANDLE out_handle[5];
    int out_count;
    ID3D11Query *done;
    // Video-processor views are driver objects; creating them per frame is
    // costly. Inputs are fixed (5 output textures); outputs are cached by the
    // hw-pool texture and subresource.
    ID3D11VideoProcessorInputView *in_view[5];
    struct pk_out_view out_view[16];
    int out_view_count;

    AVBufferRef *device_ref;
    AVBufferRef *hw_pool;
    int subfmt;
    int width;
    int height;
};

static HANDLE pk_share_handle(ID3D11Texture2D *tex)
{
    IDXGIResource1 *res = NULL;
    if (!tex || FAILED(ID3D11Texture2D_QueryInterface(tex, &IID_IDXGIResource1,
                                                       (void **)&res)))
        return NULL;
    HANDLE handle = NULL;
    HRESULT hr = res->lpVtbl->CreateSharedHandle(
        res, NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        NULL, &handle);
    res->lpVtbl->Release(res);
    return SUCCEEDED(hr) ? handle : NULL;
}

static ID3D11Texture2D *pk_create_tex(ID3D11Device *dev, int w, int h,
                                      DXGI_FORMAT fmt, UINT bind)
{
    D3D11_TEXTURE2D_DESC desc = {0};
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = fmt;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bind;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                     D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

    ID3D11Texture2D *tex = NULL;
    if (FAILED(ID3D11Device_CreateTexture2D(dev, &desc, NULL, &tex)))
        return NULL;
    return tex;
}

static void pk_bridge_destroy(struct pk_bridge *b)
{
    if (!b)
        return;
    if (b->handle && b->destroy)
        b->destroy(b->handle);
    for (int i = 0; i < 5; i++) {
        if (b->in_view[i])
            ID3D11VideoProcessorInputView_Release(b->in_view[i]);
    }
    for (int i = 0; i < b->out_view_count; i++) {
        if (b->out_view[i].view)
            ID3D11VideoProcessorOutputView_Release(b->out_view[i].view);
    }
    for (int i = 0; i < 2; i++) {
        if (b->in_handle[i])
            CloseHandle(b->in_handle[i]);
        if (b->in_tex[i])
            ID3D11Texture2D_Release(b->in_tex[i]);
    }
    for (int i = 0; i < b->out_count; i++) {
        if (b->out_handle[i])
            CloseHandle(b->out_handle[i]);
        if (b->out_tex[i])
            ID3D11Texture2D_Release(b->out_tex[i]);
    }
    if (b->vp)
        ID3D11VideoProcessor_Release(b->vp);
    if (b->vp_enum)
        ID3D11VideoProcessorEnumerator_Release(b->vp_enum);
    if (b->video_ctx)
        ID3D11VideoContext_Release(b->video_ctx);
    if (b->video_dev)
        ID3D11VideoDevice_Release(b->video_dev);
    if (b->done)
        ID3D11Query_Release(b->done);
    if (b->ctx)
        ID3D11DeviceContext_Release(b->ctx);
    if (b->dev)
        ID3D11Device_Release(b->dev);
    if (b->hw_pool)
        av_buffer_unref(&b->hw_pool);
    if (b->device_ref)
        av_buffer_unref(&b->device_ref);
    if (b->dll)
        FreeLibrary(b->dll);
    if (b->proxy)
        FreeLibrary(b->proxy);
    talloc_free(b);
}

static bool pk_bridge_init(struct mp_filter *f, struct mp_image *img)
{
    struct priv *p = f->priv;
    struct mp_stream_info *info = mp_filter_find_stream_info(f);
    if (!info || !info->hwdec_devs) {
        MP_ERR(f, "playkit: no hwdec device information\n");
        return false;
    }

    struct hwdec_imgfmt_request req = {
        .imgfmt = IMGFMT_D3D11,
        .probing = false,
    };
    hwdec_devices_request_for_img_fmt(info->hwdec_devs, &req);
    struct mp_hwdec_ctx *hwctx = hwdec_devices_get_by_imgfmt_and_type(
        info->hwdec_devs, IMGFMT_D3D11, AV_HWDEVICE_TYPE_D3D11VA);
    if (!hwctx || !hwctx->av_device_ref) {
        MP_ERR(f, "playkit: no D3D11VA device\n");
        return false;
    }

    AVBufferRef *device_ref = av_buffer_ref(hwctx->av_device_ref);
    AVHWDeviceContext *avhwctx = (void *)device_ref->data;
    AVD3D11VADeviceContext *d3dctx = avhwctx->hwctx;

    DXGI_FORMAT dxgi_fmt;
    int bits;
    if (img->params.hw_subfmt == IMGFMT_NV12) {
        dxgi_fmt = DXGI_FORMAT_NV12;
        bits = 8;
    } else if (img->params.hw_subfmt == IMGFMT_P010) {
        dxgi_fmt = DXGI_FORMAT_P010;
        bits = 10;
    } else {
        MP_ERR(f, "playkit: unsupported D3D11 subformat %s\n",
               mp_imgfmt_to_name(img->params.hw_subfmt));
        av_buffer_unref(&device_ref);
        return false;
    }

    struct pk_bridge *b = talloc_zero(p, struct pk_bridge);
    b->dev = d3dctx->device;
    ID3D11Device_AddRef(b->dev);
    b->ctx = d3dctx->device_context;
    ID3D11DeviceContext_AddRef(b->ctx);

    if (FAILED(ID3D11Device_QueryInterface(b->dev, &IID_ID3D11VideoDevice,
                                            (void **)&b->video_dev)) ||
        FAILED(ID3D11DeviceContext_QueryInterface(b->ctx, &IID_ID3D11VideoContext,
                                                   (void **)&b->video_ctx))) {
        MP_ERR(f, "playkit: D3D11 video device unavailable\n");
        goto fail;
    }

    D3D11_VIDEO_PROCESSOR_CONTENT_DESC vpdesc = {
        .InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
        .InputWidth = (UINT)img->w,
        .InputHeight = (UINT)img->h,
        .OutputWidth = (UINT)img->w,
        .OutputHeight = (UINT)img->h,
        .Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
    };
    if (FAILED(ID3D11VideoDevice_CreateVideoProcessorEnumerator(
            b->video_dev, &vpdesc, &b->vp_enum)) ||
        FAILED(ID3D11VideoDevice_CreateVideoProcessor(
            b->video_dev, b->vp_enum, 0, &b->vp))) {
        MP_ERR(f, "playkit: cannot create D3D11 video processor\n");
        goto fail;
    }

    UINT fmt_flags = 0;
    if (FAILED(ID3D11VideoProcessorEnumerator_CheckVideoProcessorFormat(
            b->vp_enum, DXGI_FORMAT_R8G8B8A8_UNORM, &fmt_flags)) ||
        !(fmt_flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT) ||
        FAILED(ID3D11VideoProcessorEnumerator_CheckVideoProcessorFormat(
            b->vp_enum, dxgi_fmt, &fmt_flags)) ||
        !(fmt_flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT)) {
        MP_ERR(f, "playkit: video processor lacks RGBA->%s support\n",
               mp_imgfmt_to_name(b->subfmt));
        goto fail;
    }

    b->device_ref = device_ref;
    b->subfmt = img->params.hw_subfmt;
    b->width = img->w;
    b->height = img->h;
    b->out_count = p->opts->multi - 1;
    if (b->out_count > 5)
        b->out_count = 5;

    D3D11_QUERY_DESC qd = { .Query = D3D11_QUERY_EVENT };
    if (FAILED(ID3D11Device_CreateQuery(b->dev, &qd, &b->done)))
        goto fail;

    for (int i = 0; i < 2; i++) {
        b->in_tex[i] = pk_create_tex(b->dev, b->width, b->height, dxgi_fmt,
                                     D3D11_BIND_RENDER_TARGET |
                                     D3D11_BIND_SHADER_RESOURCE);
        if (!b->in_tex[i])
            goto fail;
        b->in_handle[i] = pk_share_handle(b->in_tex[i]);
        if (!b->in_handle[i])
            goto fail;
    }
    for (int i = 0; i < b->out_count; i++) {
        b->out_tex[i] = pk_create_tex(b->dev, b->width, b->height,
                                      DXGI_FORMAT_R8G8B8A8_UNORM,
                                      D3D11_BIND_RENDER_TARGET |
                                      D3D11_BIND_UNORDERED_ACCESS |
                                      D3D11_BIND_SHADER_RESOURCE);
        if (!b->out_tex[i])
            goto fail;
        b->out_handle[i] = pk_share_handle(b->out_tex[i]);
        if (!b->out_handle[i])
            goto fail;
    }

    if (!mp_update_av_hw_frames_pool(&b->hw_pool, b->device_ref, IMGFMT_D3D11,
                                     b->subfmt, b->width, b->height,
                                     false)) {
        MP_ERR(f, "playkit: cannot create %s D3D11 frame pool\n",
               mp_imgfmt_to_name(b->subfmt));
        goto fail;
    }

    IDXGIDevice *dxgi_dev = NULL;
    IDXGIAdapter *adapter = NULL;
    DXGI_ADAPTER_DESC adesc = {0};
    if (FAILED(ID3D11Device_QueryInterface(b->dev, &IID_IDXGIDevice,
                                            (void **)&dxgi_dev)))
        goto fail;
    HRESULT hr = IDXGIDevice_GetAdapter(dxgi_dev, &adapter);
    IDXGIDevice_Release(dxgi_dev);
    if (FAILED(hr))
        goto fail;
    hr = IDXGIAdapter_GetDesc(adapter, &adesc);
    IDXGIAdapter_Release(adapter);
    if (FAILED(hr))
        goto fail;

    MP_INFO(f, "playkit: D3D11 adapter LUID=%08lx%08lx name=%ls\n",
            (unsigned long)adesc.AdapterLuid.HighPart,
            (unsigned long)adesc.AdapterLuid.LowPart,
            adesc.Description);

    char dll_path[MAX_PATH];
    snprintf(dll_path, sizeof(dll_path), "%s",
             p->opts->dll ? p->opts->dll : "vdlssg.dll");

    /* The version.dll proxy must be resident before vdlssg.dll initializes
     * NGX; otherwise vdlssg can bind the system 2x runtime directly. */
    char proxy_dir[MAX_PATH] = {0};
    if (p->opts->proxy_dir && p->opts->proxy_dir[0]) {
        snprintf(proxy_dir, sizeof(proxy_dir), "%s", p->opts->proxy_dir);
    } else {
        const char *sep = NULL;
        for (const char *q = dll_path; *q; q++) {
            if (*q == '/' || *q == '\\')
                sep = q;
        }
        if (sep) {
            size_t n = (size_t)(sep - dll_path);
            if (n >= sizeof(proxy_dir))
                n = sizeof(proxy_dir) - 1;
            memcpy(proxy_dir, dll_path, n);
            proxy_dir[n] = '\0';
        } else {
            snprintf(proxy_dir, sizeof(proxy_dir), ".");
        }
    }

    /* NGX resolves the runtime through the paths passed to Init.  Relative
     * paths make it load the system 2x runtime instead of the bundled one,
     * so pin both the bridge DLL and the proxy directory to absolute paths
     * before anything is loaded. */
    char abs_dll[MAX_PATH];
    char abs_proxy_dir[MAX_PATH];
    if (!GetFullPathNameA(dll_path, MAX_PATH, abs_dll, NULL) ||
        !GetFullPathNameA(proxy_dir, MAX_PATH, abs_proxy_dir, NULL)) {
        MP_ERR(f, "playkit: cannot resolve bridge paths (error %lu)\n",
               GetLastError());
        goto fail;
    }

    char proxy_path[MAX_PATH];
    snprintf(proxy_path, sizeof(proxy_path), "%s/version.dll", abs_proxy_dir);
    wchar_t proxy_w[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, proxy_path, -1, proxy_w, MAX_PATH);
    b->proxy = LoadLibraryW(proxy_w);
    if (!b->proxy) {
        MP_ERR(f, "playkit: cannot load %s (error %lu)\n", proxy_path,
               GetLastError());
        goto fail;
    }

    wchar_t dll_w[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, abs_dll, -1, dll_w, MAX_PATH);
    b->dll = LoadLibraryExW(dll_w, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!b->dll) {
        MP_ERR(f, "playkit: cannot load %s (error %lu)\n", dll_path,
               GetLastError());
        goto fail;
    }
    b->create = (void *)GetProcAddress(b->dll, "vdlssg_bridge_create");
    b->process = (void *)GetProcAddress(b->dll, "vdlssg_bridge_process");
    b->destroy = (void *)GetProcAddress(b->dll, "vdlssg_bridge_destroy");
    if (!b->create || !b->process || !b->destroy) {
        MP_ERR(f, "playkit: %s does not export the vdlssg bridge\n", dll_path);
        goto fail;
    }

    wchar_t proxy_dir_w[MAX_PATH] = L"";
    MultiByteToWideChar(CP_UTF8, 0, abs_proxy_dir, -1, proxy_dir_w, MAX_PATH);

    char err[512] = {0};
    b->handle = b->create(adesc.AdapterLuid.LowPart,
                          (int32_t)adesc.AdapterLuid.HighPart,
                          b->width, b->height, bits, p->opts->matrix,
                          p->opts->full_range ? 1 : 0,
                          proxy_dir_w[0] ? proxy_dir_w : NULL, err,
                          sizeof(err));
    if (!b->handle) {
        MP_ERR(f, "playkit: vdlssg_bridge_create failed: %s\n", err);
        goto fail;
    }

    p->bridge = b;
    MP_INFO(f, "playkit: vdlssg bridge ready %dx%d %s bits=%d multi=%d\n",
            b->width, b->height, mp_imgfmt_to_name(b->subfmt), bits,
            p->opts->multi);
    return true;

fail:
    pk_bridge_destroy(b);
    return false;
}

static bool pk_wait(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    ID3D11DeviceContext_End(b->ctx, (ID3D11Asynchronous *)b->done);
    while (ID3D11DeviceContext_GetData(b->ctx, (ID3D11Asynchronous *)b->done,
                                       NULL, 0, 0) == S_FALSE)
        Sleep(0);
    if (ID3D11DeviceContext_GetData(b->ctx, (ID3D11Asynchronous *)b->done,
                                    NULL, 0, 0) != S_OK) {
        MP_ERR(f, "playkit: D3D11 copy did not complete\n");
        return false;
    }
    return true;
}

static uint64_t pk_hash_texture(struct pk_bridge *b, ID3D11Texture2D *tex,
                                UINT sub);

static bool pk_generate(struct mp_filter *f, struct mp_image *prev,
                        struct mp_image *cur)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    struct mp_image *src[2] = { prev, cur };
    const double t0 = pk_now_ms();

    for (int i = 0; i < 2; i++) {
        if (!src[i] || !src[i]->planes[0])
            return false;
        UINT src_sub = (UINT)(intptr_t)src[i]->planes[1];
        ID3D11DeviceContext_CopySubresourceRegion(
            b->ctx, (ID3D11Resource *)b->in_tex[i], 0, 0, 0, 0,
            (ID3D11Resource *)src[i]->planes[0], src_sub, NULL);
    }
    // The shared textures are consumed by a separate D3D12 device.  GetData() alone only proves
    // the D3D11 queue retired; Flush makes the writes visible to the other API before D3D12 reads
    // them through the NT handle.
    ID3D11DeviceContext_Flush(b->ctx);
    if (!pk_wait(f))
        return false;
    const double t1 = pk_now_ms();

    if (p->opts->log) {
        MP_INFO(f, "playkit: in[0] hash=%016llx in[1] hash=%016llx\n",
                (unsigned long long)pk_hash_texture(b, b->in_tex[0], 0),
                (unsigned long long)pk_hash_texture(b, b->in_tex[1], 0));
    }

    struct pk_stats stats = {0};
    char err[512] = {0};
    /* DLSS-G's multiFrameCount is the number of generated frames, not the
     * total including the real current frame; 6x is five generated frames. */
    int rc = b->process(b->handle, b->in_handle[0], b->in_handle[1],
                        (void **)b->out_handle, b->out_count,
                        p->opts->multi - 1, &stats, err, sizeof(err));
    if (rc != 0) {
        MP_ERR(f, "playkit: vdlssg_bridge_process failed (%d): %s\n", rc, err);
        return false;
    }
    const double t2 = pk_now_ms();
    if (p->opts->log || getenv("PLAYKIT_TIMING")) {
        MP_INFO(f, "playkit: timing copy_wait=%.2f process=%.2f bridge wall=%.2f eval=%.2f gpu=%.2f\n",
                t1 - t0, t2 - t1, stats.wall_ms, stats.evaluate_ms,
                stats.gpu_ms);
    }
    return true;
}

static uint64_t pk_hash_texture(struct pk_bridge *b, ID3D11Texture2D *tex,
                                UINT sub)
{
    D3D11_TEXTURE2D_DESC d = {0};
    ID3D11Texture2D_GetDesc(tex, &d);

    D3D11_TEXTURE2D_DESC sd = d;
    sd.ArraySize = 1;
    sd.MipLevels = 1;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.BindFlags = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags = 0;

    ID3D11Texture2D *staging = NULL;
    if (FAILED(ID3D11Device_CreateTexture2D(b->dev, &sd, NULL, &staging)))
        return 0;

    ID3D11DeviceContext_CopySubresourceRegion(
        b->ctx, (ID3D11Resource *)staging, 0, 0, 0, 0,
        (ID3D11Resource *)tex, sub, NULL);

    uint64_t hash = 1469598103934665603ULL;
    D3D11_MAPPED_SUBRESOURCE map;
    HRESULT map_hr = ID3D11DeviceContext_Map(b->ctx, (ID3D11Resource *)staging,
                                             0, D3D11_MAP_READ, 0, &map);
    if (SUCCEEDED(map_hr)) {
        const uint8_t *p = map.pData;
        size_t rows = d.Format == DXGI_FORMAT_R8G8B8A8_UNORM
                          ? d.Height
                          : (size_t)d.Height * 3 / 2;
        size_t bytes = (size_t)map.RowPitch * rows;
        for (size_t i = 0; i < bytes; i++) {
            hash ^= p[i];
            hash *= 1099511628211ULL;
        }
        ID3D11DeviceContext_Unmap(b->ctx, (ID3D11Resource *)staging, 0);
    } else {
        fprintf(stderr, "playkit: hash map failed hr=0x%08lx\n", (unsigned long)map_hr);
        hash = 0;
    }

    ID3D11Texture2D_Release(staging);
    return hash;
}

static void pk_dump_rgba(struct pk_bridge *b, ID3D11Texture2D *tex,
                         const char *path)
{
    D3D11_TEXTURE2D_DESC d = {0};
    ID3D11Texture2D_GetDesc(tex, &d);
    D3D11_TEXTURE2D_DESC sd = d;
    sd.ArraySize = 1;
    sd.MipLevels = 1;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.BindFlags = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags = 0;

    ID3D11Texture2D *staging = NULL;
    if (FAILED(ID3D11Device_CreateTexture2D(b->dev, &sd, NULL, &staging)))
        return;
    ID3D11DeviceContext_CopySubresourceRegion(
        b->ctx, (ID3D11Resource *)staging, 0, 0, 0, 0,
        (ID3D11Resource *)tex, 0, NULL);

    D3D11_MAPPED_SUBRESOURCE map;
    if (SUCCEEDED(ID3D11DeviceContext_Map(b->ctx, (ID3D11Resource *)staging,
                                          0, D3D11_MAP_READ, 0, &map))) {
        FILE *fp = fopen(path, "wb");
        if (fp) {
            fprintf(fp, "P6\n%u %u\n255\n", d.Width, d.Height);
            for (UINT y = 0; y < d.Height; y++) {
                const uint8_t *row = (const uint8_t *)map.pData + (size_t)y * map.RowPitch;
                for (UINT x = 0; x < d.Width; x++)
                    fwrite(row + (size_t)x * 4, 1, 3, fp);
            }
            fclose(fp);
        }
        ID3D11DeviceContext_Unmap(b->ctx, (ID3D11Resource *)staging, 0);
    }
    ID3D11Texture2D_Release(staging);
}

static struct mp_image *pk_frame(struct mp_filter *f, int index, double pts,
                                 double nominal_fps)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    if (index < 0 || index >= b->out_count)
        return NULL;

    const double pt0 = pk_now_ms();
    AVFrame *av = av_frame_alloc();
    if (!av)
        return NULL;
    if (av_hwframe_get_buffer(b->hw_pool, av, 0) < 0) {
        av_frame_free(&av);
        return NULL;
    }
    const double pt1 = pk_now_ms();

    ID3D11Texture2D *dst = (ID3D11Texture2D *)av->data[0];
    UINT dst_sub = (UINT)(intptr_t)av->data[1];

    if (p->opts->log) {
        uint64_t src_hash = pk_hash_texture(b, b->out_tex[index], 0);
        MP_INFO(f, "playkit: src[%d] hash=%016llx dst=%p sub=%u\n",
                index, (unsigned long long)src_hash, (void *)dst,
                (unsigned)dst_sub);
        if (index >= 0 && index < 5) {
            static int dumped;
            if (dumped < 5) {
                char path[128];
                snprintf(path, sizeof(path),
                         "E:/code/dlssg-vfi/work/out%d.ppm", index);
                pk_dump_rgba(b, b->out_tex[index], path);
                dumped++;
            }
        }
    }

    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC ivd = {
        .FourCC = 0,
        .ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D,
    };
    ID3D11VideoProcessorInputView *in_view = b->in_view[index];
    HRESULT hr = S_OK;
    if (!in_view) {
        hr = ID3D11VideoDevice_CreateVideoProcessorInputView(
            b->video_dev, (ID3D11Resource *)b->out_tex[index], b->vp_enum,
            &ivd, &in_view);
        if (FAILED(hr)) {
            MP_ERR(f, "playkit: CreateVideoProcessorInputView failed 0x%lx\n",
                   (unsigned long)hr);
            av_frame_free(&av);
            return NULL;
        }
        b->in_view[index] = in_view;
    }

    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC ovd = {0};
    if (dst_sub) {
        ovd.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2DARRAY;
        ovd.Texture2DArray.FirstArraySlice = dst_sub;
        ovd.Texture2DArray.ArraySize = 1;
    } else {
        ovd.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    }
    ID3D11VideoProcessorOutputView *out_view = NULL;
    bool out_view_cached = false;
    for (int i = 0; i < b->out_view_count; i++) {
        if (b->out_view[i].tex == dst && b->out_view[i].sub == dst_sub) {
            out_view = b->out_view[i].view;
            out_view_cached = true;
            break;
        }
    }
    if (!out_view) {
        hr = ID3D11VideoDevice_CreateVideoProcessorOutputView(
            b->video_dev, (ID3D11Resource *)dst, b->vp_enum, &ovd, &out_view);
        if (FAILED(hr)) {
            MP_ERR(f, "playkit: CreateVideoProcessorOutputView failed 0x%lx\n",
                   (unsigned long)hr);
            av_frame_free(&av);
            return NULL;
        }
        if (b->out_view_count < (int)MP_ARRAY_SIZE(b->out_view)) {
            b->out_view[b->out_view_count].tex = dst;
            b->out_view[b->out_view_count].sub = dst_sub;
            b->out_view[b->out_view_count].view = out_view;
            b->out_view_count++;
            out_view_cached = true;
        }
    }

    ID3D11VideoContext_VideoProcessorSetStreamFrameFormat(
        b->video_ctx, b->vp, 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
    D3D11_VIDEO_PROCESSOR_STREAM stream = {
        .Enable = TRUE,
        .OutputIndex = 0,
        .InputFrameOrField = 0,
        .pInputSurface = in_view,
    };
    hr = ID3D11VideoContext_VideoProcessorBlt(
        b->video_ctx, b->vp, out_view, 0, 1, &stream);
    if (!out_view_cached)
        ID3D11VideoProcessorOutputView_Release(out_view);
    const double pt2 = pk_now_ms();
    if (FAILED(hr)) {
        MP_ERR(f, "playkit: VideoProcessorBlt failed 0x%lx\n",
               (unsigned long)hr);
        av_frame_free(&av);
        return NULL;
    }
    if (getenv("PLAYKIT_TIMING")) {
        MP_INFO(f, "playkit: frame[%d] get=%.2f blt=%.2f total=%.2f dst=%p sub=%u\n",
                index, pt1 - pt0, pt2 - pt1, pt2 - pt0, (void *)dst,
                (unsigned)dst_sub);
    }

    if (p->opts->log) {
        uint64_t hash = pk_hash_texture(b, dst, dst_sub);
        MP_INFO(f, "playkit: out[%d] hash=%016llx\n", index,
                (unsigned long long)hash);
    }

    struct mp_image *img = mp_image_from_av_frame(av);
    av_frame_free(&av);
    if (!img)
        return NULL;
    img->pts = pts;
    img->nominal_fps = nominal_fps;
    return img;
}

static void destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    reset(f);
    pk_bridge_destroy(p->bridge);
    p->bridge = NULL;
}

static void write_video(struct mp_filter *f, struct mp_image *img)
{
    struct priv *p = f->priv;

    if (p->opts->log)
        MP_INFO(f, "playkit: out #%d pts=%.6f\n", ++p->out_count, img->pts);
    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, img));
}

/*
 * Convert a software frame to the requested output format before it is
 * duplicated. Hardware frames are passed through unchanged: turning a D3D11
 * texture into another format needs a video processor, which belongs to the
 * future D3D11/D3D12 bridge rather than to the multiplier itself.
 */
static struct mp_image *convert_outfmt(struct mp_filter *f, struct mp_image *src)
{
    struct priv *p = f->priv;
    const int outfmt = p->opts->outfmt;

    if (!outfmt || src->imgfmt == outfmt)
        return src;

    if (src->imgfmt == IMGFMT_D3D11) {
        MP_ERR(f, "playkit: outfmt=%s cannot convert hardware %s; "
                  "put a format filter after playkit instead\n",
               mp_imgfmt_to_name(outfmt),
               src->params.hw_subfmt ? mp_imgfmt_to_name(src->params.hw_subfmt) : "d3d11");
        return NULL;
    }

    struct mp_image *dst = mp_image_alloc(outfmt, src->w, src->h);
    if (!dst)
        return NULL;

    if (p->opts->log) {
        MP_INFO(f, "playkit: convert %s -> %s\n",
                mp_imgfmt_to_name(src->imgfmt), mp_imgfmt_to_name(outfmt));
    }

    dst->params = src->params;
    dst->params.imgfmt = outfmt;
    dst->params.hw_subfmt = 0;
    mp_image_params_guess_csp(&dst->params);
    mp_image_copy_attributes(dst, src);

    if (mp_sws_scale(p->sws, dst, src) < 0) {
        MP_ERR(f, "playkit: cannot convert %s to %s\n",
               mp_imgfmt_to_name(src->imgfmt), mp_imgfmt_to_name(outfmt));
        mp_image_unrefp(&dst);
        return NULL;
    }

    mp_image_unrefp(&src);
    return dst;
}

static void process(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct mp_pin *in = f->ppins[0];
    struct mp_pin *out = f->ppins[1];

    if (!mp_pin_in_needs_data(out))
        return;

    /* Drain the queue before asking for more input. */
    if (p->queue_pos < p->queue_len) {
        struct mp_image *img = p->queue[p->queue_pos];
        p->queue[p->queue_pos] = NULL;
        p->queue_pos++;
        if (p->queue_pos == p->queue_len) {
            p->queue_pos = 0;
            p->queue_len = 0;
        }
        write_video(f, img);
        return;
    }

    /* The queue is empty; a pending EOF can be forwarded now. */
    if (p->eof_pending) {
        p->eof_pending = false;
        mp_pin_in_write(out, MP_EOF_FRAME);
        return;
    }

    if (!mp_pin_out_request_data(in))
        return;

    struct mp_frame frame = mp_pin_out_read(in);
    if (frame.type == MP_FRAME_NONE)
        return;

    if (mp_frame_is_signaling(frame)) {
        if (p->queue_pos < p->queue_len) {
            /* Keep the queued frames; write the first one now so the filter
             * keeps being scheduled, then forward EOF after the queue drains. */
            p->eof_pending = true;
            mp_frame_unref(&frame);

            struct mp_image *img = p->queue[p->queue_pos];
            p->queue[p->queue_pos] = NULL;
            p->queue_pos++;
            write_video(f, img);
        } else {
            mp_pin_in_write(out, frame);
        }
        return;
    }

    if (frame.type != MP_FRAME_VIDEO) {
        mp_frame_unref(&frame);
        mp_filter_internal_mark_failed(f);
        return;
    }

    struct mp_image *cur = frame.data;
    const int multi = p->opts->multi;

    if (p->opts->log && !p->logged) {
        MP_INFO(f, "playkit: %dx%d imgfmt=%s hw_subfmt=%s pts=%.6f multi=%d\n",
                cur->w, cur->h,
                mp_imgfmt_to_name(cur->imgfmt),
                cur->params.hw_subfmt ? mp_imgfmt_to_name(cur->params.hw_subfmt)
                                      : "none",
                cur->pts, multi);
        p->logged = true;
    }

    if (p->opts->outfmt && cur->imgfmt != p->opts->outfmt) {
        struct mp_image *conv = convert_outfmt(f, cur);
        if (!conv) {
            mp_frame_unref(&frame);
            mp_filter_internal_mark_failed(f);
            return;
        }
        cur = conv;
    }

    /* First frame: pass it through and use it as the previous frame. */
    if (!p->prev) {
        p->prev = mp_image_new_ref(cur);
        write_video(f, cur);
        return;
    }

    double step = 0.0;
    if (cur->pts != MP_NOPTS_VALUE && p->prev->pts != MP_NOPTS_VALUE)
        step = (cur->pts - p->prev->pts) / multi;

    if (p->opts->log)
        MP_INFO(f, "playkit: in pts=%.6f prev=%.6f multi=%d step=%.6f\n",
                cur->pts, p->prev->pts, multi, step);

    double out_fps = cur->nominal_fps;
    if (!isnan(out_fps) && out_fps > 0)
        out_fps *= multi;

    /*
     * D3D11 input: hand the two textures to vdlssg and let it generate the
     * intermediate frames on the GPU.  The filter only copies the shared
     * output textures into mpv's own hw frames.
     */
    if (cur->imgfmt == IMGFMT_D3D11 && multi > 1) {
        if (!p->bridge && !p->bridge_failed) {
            if (!pk_bridge_init(f, cur))
                p->bridge_failed = true;
        }
        if (p->bridge && pk_generate(f, p->prev, cur)) {
            for (int k = 0; k < p->bridge->out_count; k++) {
                struct mp_image *img = pk_frame(f, k,
                    p->prev->pts + (k + 1) * step, out_fps);
                if (!img) {
                    MP_ERR(f, "playkit: cannot wrap generated frame %d\n", k);
                    mp_filter_internal_mark_failed(f);
                    mp_frame_unref(&frame);
                    return;
                }
                p->queue[p->queue_len++] = img;
            }
            cur->nominal_fps = out_fps;
            p->queue[p->queue_len++] = cur;
            mp_image_unrefp(&p->prev);
            p->prev = mp_image_new_ref(cur);

            struct mp_image *first = p->queue[0];
            p->queue[0] = NULL;
            p->queue_pos = 1;
            write_video(f, first);
            return;
        }
        if (!p->bridge) {
            MP_WARN(f, "playkit: vdlssg bridge unavailable; duplicating frames\n");
        }
    }

    /*
     * Queue the intermediate frames followed by the current frame.  The
     * current frame's PTS is already the end of the interval, so only the
     * intermediate frames need new timestamps.
     */
    for (int k = 1; k < multi; k++) {
        struct mp_image *img = mp_image_new_ref(cur);
        img->pts = p->prev->pts + k * step;
        if (!isnan(cur->nominal_fps) && cur->nominal_fps > 0)
            img->nominal_fps = cur->nominal_fps * multi;
        p->queue[p->queue_len++] = img;
    }

    if (!isnan(cur->nominal_fps) && cur->nominal_fps > 0)
        cur->nominal_fps *= multi;
    p->queue[p->queue_len++] = cur; /* take ownership of the input frame */

    mp_image_unrefp(&p->prev);
    p->prev = mp_image_new_ref(cur);

    struct mp_image *first = p->queue[0];
    p->queue[0] = NULL;
    p->queue_pos = 1;
    write_video(f, first);
}

static const struct mp_filter_info vf_playkit_filter = {
    .name = "playkit",
    .process = process,
    .reset = reset,
    .destroy = destroy,
    .priv_size = sizeof(struct priv),
};

static struct mp_filter *vf_playkit_create(struct mp_filter *parent,
                                          void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &vf_playkit_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *p = f->priv;
    p->opts = talloc_steal(p, options);
    if (p->opts->multi < 2)
        p->opts->multi = 2;
    if (p->opts->multi > 6)
        p->opts->multi = 6;
    p->queue_len = 0;
    p->queue_pos = 0;
    p->eof_pending = false;

    if (p->opts->outfmt) {
        p->sws = mp_sws_alloc(p);
        if (!p->sws) {
            talloc_free(f);
            return NULL;
        }
        mp_sws_enable_cmdline_opts(p->sws, f->global);
    }

    return f;
}

const struct mp_user_filter_entry vf_playkit = {
    .desc = {
        .description = "PlayKit frame multiplier (pass-through prototype)",
        .name = "playkit",
        .priv_size = sizeof(OPT_BASE_STRUCT),
        .priv_defaults = &(const OPT_BASE_STRUCT) {
            .multi = 6,
            .log = 0,
        },
        .options = vf_opts_fields,
    },
    .create = vf_playkit_create,
};
