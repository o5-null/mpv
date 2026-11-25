/*
 * PlayKit frame multiplier (D3D11 zero-copy DLSS-G)
 *
 * For every input frame pair, emit `multi` output frames.  The generated
 * frames come from the vdlssg bridge: both input textures and every output
 * texture are D3D11 shared NT handles that the D3D12 side opens directly, so
 * no plane is copied through the CPU.  When the bridge is unavailable (no
 * vdlssg.dll, no D3D11 hwdec, unsupported format) the filter degrades to
 * duplicating the current frame, which still validates PTS interpolation,
 * queueing and EOF handling.
 */

#include <math.h>
#include <stdio.h>
#include <windows.h>
#include <d3d11.h>
#include <d3d11_1.h>
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
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "video/hwdec.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/mp_image_pool.h"
#include "video/csputils.h"
#include "video/sws_utils.h"

/* 每个槽位独立的一组输入/输出纹理，引擎一次调用只碰自己那一组。槽位数就是
 * 流水深度：同一个时刻最多有 slot 对帧在 D3D12/NGX 里计算，filter 线程只做
 * 提交和取回，不再每对阻塞一次。 */
#define PK_MAX_OUT 5    /* multi-1，6× 是 5 张生成帧 */
#define PK_MAX_SLOT 4

enum {
    PK_SLOT_FREE = 0,   /* 空，可以接收新的一对 */
    PK_SLOT_BUSY,       /* 已交给工作线程，引擎在算 */
    PK_SLOT_READY,      /* 结果就绪，等 filter 线程取回 */
    PK_SLOT_ERROR,
};

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
    int pipe;
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
    {"pipe", OPT_INT(pipe)},
    {0}
};

struct pk_bridge;

struct priv {
    struct opts *opts;
    struct mp_sws_context *sws;

    struct mp_image *prev;
    struct mp_image *queue[PK_MAX_SLOT * 6 + 2];
    int queue_len;
    int queue_pos;

    bool eof_pending;
    bool logged;
    bool params_logged;
    bool bridge_failed;
    bool cs_logged;
    int out_count;
    int seq;

    struct pk_bridge *bridge;

    /* 工作线程：槽位的提交/取回都经过 job 队列，filter 线程不再等引擎。 */
    mp_thread worker[PK_MAX_SLOT];
    mp_mutex lock;
    mp_cond cond;
    bool thread_stop;
    bool threads_started;
    int job_index[PK_MAX_SLOT];
    int job_head;
    int job_count;
    /* 提交顺序，取回结果时按它来，保证输出帧顺序不变。 */
    int order[PK_MAX_SLOT];
    int order_len;

    /* 停摆看门狗（PLAYKIT_TRACE 时才启动）：流水线 3 秒没进展就把现场打出来，
     * 用来区分「引擎卡死」和「唤醒丢失」这两种冻结。 */
    mp_thread watchdog;
    mp_cond wd_cond;
    bool watchdog_started;
    double t_progress;
    bool worker_in_engine;
    int worker_slot;

    /* PLAYKIT_TIMING 的分解计时：输入端等待、调度空档、整对周期。 */
    double t_req;
    double t_idle;
    double t_cycle;
    double t_pair_end;
};

static void pk_threads_stop(struct mp_filter *f);

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

    pk_threads_stop(f);
    clear_queue(f);
    p->eof_pending = false;
    p->logged = false;
    p->params_logged = false;
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

struct pk_slot {
    ID3D11Texture2D *in_tex[2];
    HANDLE in_handle[2];
    ID3D11Texture2D *out_tex[PK_MAX_OUT];
    HANDLE out_handle[PK_MAX_OUT];
    int out_count;
    /* 输入拷贝的退休点：工作线程只等这个，再去调引擎。 */
    ID3D11Query *copy_done;
    /* 生成帧拷进 hw 池之后的退休点，槽位复用前必须等它。 */
    ID3D11Query *blit_done;
    bool blit_pending;
    /* 输出纹理的 VP 输入视图，创建一次复用。 */
    ID3D11VideoProcessorInputView *in_view[PK_MAX_OUT];
    int state;
    /* 这一对的输出时间戳信息，由 filter 线程填写、取回时使用。 */
    double step;
    double out_fps;
    double pts_prev;
    struct mp_image *cur;
    int seq;
    /* 统计 */
    double copy_ms;
    double process_ms;
    struct pk_stats stats;
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
    ID3D11VideoContext1 *video_ctx1;
    ID3D11VideoProcessorEnumerator *vp_enum;
    ID3D11VideoProcessor *vp;
    /* 流水线槽位，`slot_count` 即同时可在引擎里的对数。 */
    struct pk_slot slot[PK_MAX_SLOT];
    int slot_count;
    /* 诊断读回（hash / luma stat）专用的退休点。 */
    ID3D11Query *misc_done;
    // Video-processor views are driver objects; creating them per frame is
    // costly. Inputs are per slot (fixed set of output textures); outputs are
    // cached by the hw-pool texture and subresource.
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

/* 便携包一般从自己的目录启动，但文件关联、快捷方式（自带"起始位置"）会让
 * mpv 的工作目录变成别处；Win32 按工作目录解析相对路径，包内的 DLL 就会
 * 找不到。下面两个辅助函数在相对路径按工作目录不存在时，回退到 mpv.exe
 * 所在目录再试一次。 */
static bool pk_is_absolute(const char *path)
{
    if (!path || !path[0])
        return false;
    if (path[0] == '/' || path[0] == '\\')
        return true;
    return path[1] == ':' &&
           ((path[0] >= 'A' && path[0] <= 'Z') ||
            (path[0] >= 'a' && path[0] <= 'z'));
}

static bool pk_exe_relative_exists(const char *rel, char *out, size_t out_size)
{
    wchar_t exe[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, exe, MAX_PATH);
    if (!len || len >= MAX_PATH)
        return false;

    wchar_t *name = wcsrchr(exe, L'\\');
    if (!name)
        return false;
    *name = L'\0';

    wchar_t rel_w[MAX_PATH];
    if (!MultiByteToWideChar(CP_UTF8, 0, rel, -1, rel_w, MAX_PATH))
        return false;
    for (wchar_t *q = rel_w; *q; q++) {
        if (*q == L'/')
            *q = L'\\';
    }

    wchar_t full[MAX_PATH];
    int n = swprintf(full, MAX_PATH, L"%s\\%s", exe, rel_w);
    if (n <= 0 || n >= MAX_PATH)
        return false;
    if (GetFileAttributesW(full) == INVALID_FILE_ATTRIBUTES)
        return false;

    return WideCharToMultiByte(CP_UTF8, 0, full, -1, out, (int)out_size,
                               NULL, NULL) > 0;
}

/* 绝对路径原样保留（NGX 要绝对路径）；相对路径先按工作目录，再按 exe 目录。 */
static void pk_abs_path(const char *path, char *out, size_t out_size)
{
    if (pk_is_absolute(path)) {
        snprintf(out, out_size, "%s", path);
        return;
    }

    char cwd[MAX_PATH];
    if (GetFullPathNameA(path, MAX_PATH, cwd, NULL) &&
        GetFileAttributesA(cwd) != INVALID_FILE_ATTRIBUTES) {
        snprintf(out, out_size, "%s", cwd);
        return;
    }

    if (pk_exe_relative_exists(path, out, out_size))
        return;

    if (!GetFullPathNameA(path, MAX_PATH, cwd, NULL))
        cwd[0] = '\0';
    snprintf(out, out_size, "%s", cwd);
}

static void pk_bridge_destroy(struct pk_bridge *b)
{
    if (!b)
        return;
    if (b->handle && b->destroy)
        b->destroy(b->handle);
    for (int i = 0; i < b->out_view_count; i++) {
        if (b->out_view[i].view)
            ID3D11VideoProcessorOutputView_Release(b->out_view[i].view);
    }
    for (int s = 0; s < b->slot_count; s++) {
        struct pk_slot *sl = &b->slot[s];
        for (int i = 0; i < PK_MAX_OUT; i++) {
            if (sl->in_view[i])
                ID3D11VideoProcessorInputView_Release(sl->in_view[i]);
        }
        for (int i = 0; i < 2; i++) {
            if (sl->in_handle[i])
                CloseHandle(sl->in_handle[i]);
            if (sl->in_tex[i])
                ID3D11Texture2D_Release(sl->in_tex[i]);
        }
        for (int i = 0; i < sl->out_count; i++) {
            if (sl->out_handle[i])
                CloseHandle(sl->out_handle[i]);
            if (sl->out_tex[i])
                ID3D11Texture2D_Release(sl->out_tex[i]);
        }
        if (sl->copy_done)
            ID3D11Query_Release(sl->copy_done);
        if (sl->blit_done)
            ID3D11Query_Release(sl->blit_done);
        mp_image_unrefp(&sl->cur);
    }
    if (b->misc_done)
        ID3D11Query_Release(b->misc_done);
    if (b->vp)
        ID3D11VideoProcessor_Release(b->vp);
    if (b->vp_enum)
        ID3D11VideoProcessorEnumerator_Release(b->vp_enum);
    if (b->video_ctx1)
        ID3D11VideoContext1_Release(b->video_ctx1);
    if (b->video_ctx)
        ID3D11VideoContext_Release(b->video_ctx);
    if (b->video_dev)
        ID3D11VideoDevice_Release(b->video_dev);
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

// Same fill-in the VO does (vf_d3d11vpp does it before deriving a DXGI colour space):
// an untagged clip has repr.sys/levels unset, and guessing makes the filter, the bridge
// and the VO agree on the matrix/range the pixels are really in.
static void pk_guessed_params(const struct mp_image_params *in,
                              struct mp_image_params *out)
{
    *out = *in;
    mp_image_params_guess_csp(out);
}

// Matrix id as used by the bridge (AVColorSpace numbering, see vs-dlssg yuv.h).
static int pk_matrix_from_params(const struct mp_image_params *par, int h)
{
    switch (par->repr.sys) {
    case PL_COLOR_SYSTEM_BT_601:
        // BT.601 splits into 470BG/SMPTE170M purely by resolution.
        return h > 576 ? 6 : 5;
    case PL_COLOR_SYSTEM_SMPTE_240M: return 7;
    case PL_COLOR_SYSTEM_BT_2020_NC: return 9;
    case PL_COLOR_SYSTEM_BT_2020_C:  return 10;
    case PL_COLOR_SYSTEM_BT_709:     return 1;
    default:                         return 1;
    }
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
    const int out_count = MPCLAMP(p->opts->multi - 1, 1, PK_MAX_OUT);

    /* 流水深度：pipe 选项优先，默认两对同时在引擎里（一组 4K 槽位约 160MB）。 */
    int slots = p->opts->pipe > 0 ? p->opts->pipe : 2;
    slots = MPCLAMP(slots, 1, PK_MAX_SLOT);
    b->slot_count = slots;

    D3D11_QUERY_DESC qd = { .Query = D3D11_QUERY_EVENT };
    if (FAILED(ID3D11Device_CreateQuery(b->dev, &qd, &b->misc_done)))
        goto fail;
    for (int s = 0; s < b->slot_count; s++) {
        struct pk_slot *sl = &b->slot[s];
        sl->out_count = out_count;
        sl->state = PK_SLOT_FREE;
        if (FAILED(ID3D11Device_CreateQuery(b->dev, &qd, &sl->copy_done)) ||
            FAILED(ID3D11Device_CreateQuery(b->dev, &qd, &sl->blit_done)))
            goto fail;

        for (int i = 0; i < 2; i++) {
            sl->in_tex[i] = pk_create_tex(b->dev, b->width, b->height, dxgi_fmt,
                                          D3D11_BIND_RENDER_TARGET |
                                          D3D11_BIND_SHADER_RESOURCE);
            if (!sl->in_tex[i])
                goto fail;
            sl->in_handle[i] = pk_share_handle(sl->in_tex[i]);
            if (!sl->in_handle[i])
                goto fail;
        }
        for (int i = 0; i < out_count; i++) {
            sl->out_tex[i] = pk_create_tex(b->dev, b->width, b->height,
                                           DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D11_BIND_RENDER_TARGET |
                                           D3D11_BIND_UNORDERED_ACCESS |
                                           D3D11_BIND_SHADER_RESOURCE);
            if (!sl->out_tex[i])
                goto fail;
            sl->out_handle[i] = pk_share_handle(sl->out_tex[i]);
            if (!sl->out_handle[i])
                goto fail;
        }
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
     * before anything is loaded.  Relative options resolve against the
     * working directory first and the mpv.exe directory second, so the
     * portable layout works even when the player is started elsewhere. */
    char abs_dll[MAX_PATH];
    char abs_proxy_dir[MAX_PATH];
    pk_abs_path(dll_path, abs_dll, sizeof(abs_dll));
    pk_abs_path(proxy_dir, abs_proxy_dir, sizeof(abs_proxy_dir));
    if (!abs_dll[0] || !abs_proxy_dir[0]) {
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
    // The bridge converts NV12/P010 <-> RGB with one fixed matrix/range pair, so it has
    // to match the clip.  The options only override it: otherwise take the values the
    // decoded frames carry (BT.709 limited when the container says nothing).
    struct mp_image_params par;
    pk_guessed_params(&img->params, &par);
    int matrix = p->opts->matrix;
    if (matrix <= 0)
        matrix = pk_matrix_from_params(&par, img->h);
    int full = p->opts->full_range ||
               par.repr.levels == PL_COLOR_LEVELS_FULL;
    MP_INFO(f, "playkit: bridge colours matrix=%d full=%d (sys=%d levels=%d)\n",
            matrix, full, (int)par.repr.sys, (int)par.repr.levels);
    b->handle = b->create(adesc.AdapterLuid.LowPart,
                          (int32_t)adesc.AdapterLuid.HighPart,
                          b->width, b->height, bits, matrix, full ? 1 : 0,
                          proxy_dir_w[0] ? proxy_dir_w : NULL, err,
                          sizeof(err));
    if (!b->handle) {
        MP_ERR(f, "playkit: vdlssg_bridge_create failed: %s\n", err);
        goto fail;
    }

    p->bridge = b;
    MP_INFO(f, "playkit: vdlssg bridge ready %dx%d %s bits=%d multi=%d pipe=%d\n",
            b->width, b->height, mp_imgfmt_to_name(b->subfmt), bits,
            p->opts->multi, b->slot_count);
    return true;

fail:
    pk_bridge_destroy(b);
    return false;
}

static bool pk_slot_wait_copy(struct mp_filter *f, struct pk_slot *s)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    ID3D11Asynchronous *q = (ID3D11Asynchronous *)s->copy_done;

    while (ID3D11DeviceContext_GetData(b->ctx, q, NULL, 0, 0) == S_FALSE)
        Sleep(0);
    if (ID3D11DeviceContext_GetData(b->ctx, q, NULL, 0, 0) != S_OK) {
        MP_ERR(f, "playkit: D3D11 copy did not complete\n");
        return false;
    }
    return true;
}

/*
 * VideoProcessorBlt 是排进 D3D11 队列的，取回帧时它还没跑完；同一个槽位下一轮
 * 会覆写这些输出纹理，所以复用槽位前要等 blit 退休。
 */
static void pk_slot_wait_blit(struct mp_filter *f, struct pk_slot *s)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    ID3D11Asynchronous *q = (ID3D11Asynchronous *)s->blit_done;

    if (!s->blit_pending)
        return;
    while (ID3D11DeviceContext_GetData(b->ctx, q, NULL, 0, 0) == S_FALSE)
        Sleep(0);
    s->blit_pending = false;
}

/* 诊断读回（PLAYKIT_DSTSTAT / log 的 hash）用：等排队中的拷贝退休。 */
static void pk_wait_misc(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    ID3D11Asynchronous *q = (ID3D11Asynchronous *)b->misc_done;

    ID3D11DeviceContext_End(b->ctx, q);
    while (ID3D11DeviceContext_GetData(b->ctx, q, NULL, 0, 0) == S_FALSE)
        Sleep(0);
}

/* 工作线程：等输入拷贝退休 → 调引擎 → 标 READY → 唤醒 filter 线程。 */
static MP_THREAD_VOID pk_worker(void *arg)
{
    struct mp_filter *f = arg;
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;

    mp_thread_set_name("playkit-gen");

    mp_mutex_lock(&p->lock);
    while (!p->thread_stop) {
        if (!p->job_count) {
            mp_cond_wait(&p->cond, &p->lock);
            continue;
        }
        const int idx = p->job_index[p->job_head];
        if (getenv("PLAYKIT_TRACE"))
            MP_INFO(f, "playkit: trace worker job slot=%d\n", idx);
        p->job_head = (p->job_head + 1) % PK_MAX_SLOT;
        p->job_count--;
        mp_mutex_unlock(&p->lock);

        struct pk_slot *s = &b->slot[idx];
        int state = PK_SLOT_ERROR;
        double t0 = pk_now_ms();
        if (getenv("PLAYKIT_TRACE"))
            MP_INFO(f, "playkit: trace worker wait copy slot=%d\n", idx);
        if (pk_slot_wait_copy(f, s)) {
            if (getenv("PLAYKIT_TRACE"))
                MP_INFO(f, "playkit: trace worker copy done slot=%d\n", idx);
            s->copy_ms = pk_now_ms() - t0;
            char err[512] = {0};
            struct pk_stats stats = {0};
            t0 = pk_now_ms();
            p->worker_in_engine = true;
            p->worker_slot = idx;
            int rc = b->process(b->handle, s->in_handle[0], s->in_handle[1],
                                (void **)s->out_handle, s->out_count,
                                p->opts->multi - 1, &stats, err, sizeof(err));
            p->worker_in_engine = false;
            s->process_ms = pk_now_ms() - t0;
            if (getenv("PLAYKIT_TRACE"))
                MP_INFO(f, "playkit: trace worker process done slot=%d rc=%d\n", idx, rc);
            s->stats = stats;
            if (rc != 0) {
                MP_ERR(f, "playkit: vdlssg_bridge_process failed (%d): %s\n",
                       rc, err);
            } else {
                state = PK_SLOT_READY;
            }
        }

        mp_mutex_lock(&p->lock);
        s->state = state;
        mp_mutex_unlock(&p->lock);
        mp_filter_wakeup(f);

        /* 回到循环顶部必须重新持锁：下面是 mp_cond_wait 的进入条件。 */
        mp_mutex_lock(&p->lock);
    }
    mp_mutex_unlock(&p->lock);

    MP_THREAD_RETURN();
}

static MP_THREAD_VOID pk_watchdog(void *arg)
{
    struct mp_filter *f = arg;
    struct priv *p = f->priv;

    mp_thread_set_name("playkit-watch");

    mp_mutex_lock(&p->lock);
    while (!p->thread_stop) {
        mp_cond_timedwait(&p->wd_cond, &p->lock, MP_TIME_MS_TO_NS(500));
        if (p->thread_stop)
            break;
        if (!p->job_count && !p->order_len)
            continue;
        const double stalled = pk_now_ms() - p->t_progress;
        if (stalled < 3000)
            continue;
        MP_ERR(f, "playkit: WATCHDOG stall %.0fms jobs=%d order=%d engine=%d slot=%d\n",
               stalled, p->job_count, p->order_len,
               (int)p->worker_in_engine, p->worker_slot);
        for (int i = 0; p->bridge && i < p->bridge->slot_count; i++) {
            MP_ERR(f, "playkit: WATCHDOG slot %d state=%d blit_pending=%d\n",
                   i, p->bridge->slot[i].state,
                   (int)p->bridge->slot[i].blit_pending);
        }
        /* 再看门狗自己也要能让出 CPU，下一轮继续观察。 */
        p->t_progress = pk_now_ms();
    }
    mp_mutex_unlock(&p->lock);

    MP_THREAD_RETURN();
}

static void pk_threads_start(struct mp_filter *f)
{
    struct priv *p = f->priv;

    if (p->threads_started || !p->bridge)
        return;
    p->thread_stop = false;
    p->job_head = p->job_count = 0;
    p->order_len = 0;
    for (int i = 0; i < p->bridge->slot_count; i++) {
        if (mp_thread_create(&p->worker[i], pk_worker, f))
            MP_ERR(f, "playkit: cannot start worker thread %d\n", i);
        else
            p->threads_started = true;
    }
    if (getenv("PLAYKIT_TRACE") && p->threads_started && !p->watchdog_started) {
        p->t_progress = pk_now_ms();
        if (mp_thread_create(&p->watchdog, pk_watchdog, f)) {
            MP_ERR(f, "playkit: cannot start watchdog thread\n");
        } else {
            p->watchdog_started = true;
        }
    }
}

/* 关线程前让在飞的槽位跑完，之后才能释放纹理。 */
static void pk_threads_stop(struct mp_filter *f)
{
    struct priv *p = f->priv;

    if (!p->threads_started)
        return;
    mp_mutex_lock(&p->lock);
    p->thread_stop = true;
    mp_cond_broadcast(&p->cond);
    mp_mutex_unlock(&p->lock);
    for (int i = 0; i < p->bridge->slot_count; i++)
        mp_thread_join(p->worker[i]);
    if (p->watchdog_started) {
        mp_thread_join(p->watchdog);
        p->watchdog_started = false;
    }
    /* 排队但没跑的 job 直接丢掉，槽位状态必须复位，否则再也不会被复用。 */
    for (int i = 0; i < p->bridge->slot_count; i++) {
        mp_image_unrefp(&p->bridge->slot[i].cur);
        p->bridge->slot[i].state = PK_SLOT_FREE;
    }
    p->threads_started = false;
    p->thread_stop = false;
    p->job_head = p->job_count = 0;
    p->order_len = 0;
}

static uint64_t pk_hash_texture(struct pk_bridge *b, ID3D11Texture2D *tex,
                                UINT sub);
static void pk_luma_stat(struct mp_filter *f, ID3D11Texture2D *tex, UINT sub,
                         int index, const char *what);

// The video processor is never told what the pixels mean, so it falls back to its
// defaults: full range RGB in, BT.601 limited range YCbCr out.  The generated frames
// are then tagged with the source parameters, so the VO never sees the same encoding
// as the real frames - black crush and clipped whites on every generated frame.
// Match the output to the parameters the frame carries instead.
//
// The legacy D3D11_VIDEO_PROCESSOR_COLOR_SPACE.Nominal_Range is inert on the NVIDIA
// driver for the output stage: measured byte-identical output with 0 (full) and 1
// (studio), so a full range source still had its generated frames squeezed into
// 16-235 while the real frames kept 0-250 - a brightness pulse at source rate.
// ID3D11VideoContext1's DXGI colour space API does switch both range and matrix.
static ID3D11VideoContext1 *pk_video_ctx1(struct pk_bridge *b)
{
    if (b->video_ctx1)
        return b->video_ctx1;
    ID3D11VideoContext1 *ctx1 = NULL;
    if (FAILED(ID3D11VideoContext_QueryInterface(b->video_ctx,
                                                 &IID_ID3D11VideoContext1,
                                                 (void **)&ctx1)))
        return NULL;
    b->video_ctx1 = ctx1;
    return ctx1;
}

static void pk_update_colorspace(struct mp_filter *f, struct mp_image *ref)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    if (!b || !b->video_ctx || !b->vp)
        return;

    // 0 = BT.601, 1 = BT.709, 2 = BT.2020
    int matrix = 1;
    bool full = false;
    if (ref) {
        struct mp_image_params par;
        pk_guessed_params(&ref->params, &par);
        switch (par.repr.sys) {
        case PL_COLOR_SYSTEM_BT_601:     matrix = 0; break;
        case PL_COLOR_SYSTEM_BT_2020_NC:
        case PL_COLOR_SYSTEM_BT_2020_C:  matrix = 2; break;
        default:                         matrix = 1; break;
        }
        // Anything that is not explicitly full (auto/unknown included) is handled as
        // studio range, which is what the container tags mean in practice.
        full = par.repr.levels == PL_COLOR_LEVELS_FULL;
    }
    // Diagnostic override: PLAYKIT_OUTRANGE=full|limited.
    const char *ov = getenv("PLAYKIT_OUTRANGE");
    if (ov && !strcmp(ov, "full"))
        full = true;
    else if (ov && !strcmp(ov, "limited"))
        full = false;

    ID3D11VideoContext1 *ctx1 = getenv("PLAYKIT_LEGACY_CS") ? NULL
                                                            : pk_video_ctx1(b);
    if (ctx1) {
        // The engine hands us full range RGB (RGBA8 shared texture, 0-255).
        DXGI_COLOR_SPACE_TYPE in_cs = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        DXGI_COLOR_SPACE_TYPE out_cs;
        if (!full) {
            out_cs = matrix == 0 ? DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601
                   : matrix == 2 ? DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020
                                 : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709;
        } else {
            out_cs = matrix == 0 ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601
                   : matrix == 2 ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020
                                 : DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709;
        }
        ID3D11VideoContext1_VideoProcessorSetStreamColorSpace1(ctx1, b->vp, 0,
                                                               in_cs);
        ID3D11VideoContext1_VideoProcessorSetOutputColorSpace1(ctx1, b->vp,
                                                               out_cs);
        if (getenv("PLAYKIT_DSTSTAT") && !p->cs_logged) {
            p->cs_logged = true;
            MP_INFO(f, "playkit: colorspace cs1 in=0x%x out=0x%x\n",
                    (unsigned)in_cs, (unsigned)out_cs);
        }
        return;
    }

    D3D11_VIDEO_PROCESSOR_COLOR_SPACE in = {0};
    in.Usage = 0;              // playback
    in.RGB_Range = 0;          // the engine outputs full range RGBA
    in.YCbCr_Matrix = 1;       // BT.709 primaries
    in.Nominal_Range = 0;      // 0-255

    D3D11_VIDEO_PROCESSOR_COLOR_SPACE out = {0};
    out.Usage = 0;
    out.RGB_Range = 0;
    out.YCbCr_Matrix = matrix == 0 ? 0 : 1;
    out.Nominal_Range = full ? 0 : 1;

    ID3D11VideoContext_VideoProcessorSetStreamColorSpace(b->video_ctx, b->vp, 0,
                                                         &in);
    ID3D11VideoContext_VideoProcessorSetOutputColorSpace(b->video_ctx, b->vp,
                                                         &out);
    if (getenv("PLAYKIT_DSTSTAT") && !p->cs_logged) {
        p->cs_logged = true;
        MP_INFO(f, "playkit: colorspace legacy in=%s/%d out=%s/%d\n",
                in.YCbCr_Matrix ? "709" : "601", in.Nominal_Range,
                out.YCbCr_Matrix ? "709" : "601", out.Nominal_Range);
    }
}

static struct mp_image *pk_frame(struct mp_filter *f, struct pk_slot *s,
                                 int index, double pts, double nominal_fps);

/*
 * 把一对帧交给空闲槽位：拷贝两张输入纹理、记下这一对的时间戳信息，排进 job
 * 队列。没有空闲槽位时返回 false，调用者稍后再试。
 */
static bool pk_submit(struct mp_filter *f, struct mp_image *prev,
                      struct mp_image *cur, double step, double out_fps)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    struct mp_image *src[2] = { prev, cur };

    if (!prev || !cur || !prev->planes[0] || !cur->planes[0])
        return false;

    int idx = -1;
    mp_mutex_lock(&p->lock);
    for (int i = 0; i < b->slot_count; i++) {
        if (b->slot[i].state == PK_SLOT_FREE) {
            idx = i;
            b->slot[i].state = PK_SLOT_BUSY;
            break;
        }
    }
    mp_mutex_unlock(&p->lock);
    if (idx < 0)
        return false;

    struct pk_slot *s = &b->slot[idx];
    /* 上一轮的 blit 可能还在跑，等它退休再覆写输出纹理。 */
    pk_slot_wait_blit(f, s);

    const double t0 = pk_now_ms();
    for (int i = 0; i < 2; i++) {
        UINT src_sub = (UINT)(intptr_t)src[i]->planes[1];
        ID3D11DeviceContext_CopySubresourceRegion(
            b->ctx, (ID3D11Resource *)s->in_tex[i], 0, 0, 0, 0,
            (ID3D11Resource *)src[i]->planes[0], src_sub, NULL);
    }
    if (getenv("PLAYKIT_DSTSTAT")) {
        pk_luma_stat(f, s->in_tex[0], 0, 0, "src");
        pk_luma_stat(f, s->in_tex[1], 0, 1, "src");
    }
    if (p->opts->log) {
        MP_INFO(f, "playkit: in[0] hash=%016llx in[1] hash=%016llx\n",
                (unsigned long long)pk_hash_texture(b, s->in_tex[0], 0),
                (unsigned long long)pk_hash_texture(b, s->in_tex[1], 0));
    }
    // The shared textures are consumed by a separate D3D12 device.  GetData() alone only proves
    // the D3D11 queue retired; Flush makes the writes visible to the other API before D3D12 reads
    // them through the NT handle.
    ID3D11DeviceContext_Flush(b->ctx);
    ID3D11DeviceContext_End(b->ctx, (ID3D11Asynchronous *)s->copy_done);
    s->copy_ms = pk_now_ms() - t0;

    s->step = step;
    s->out_fps = out_fps;
    s->pts_prev = prev->pts;
    s->seq = p->seq++;
    mp_image_unrefp(&s->cur);
    s->cur = mp_image_new_ref(cur);

    mp_mutex_lock(&p->lock);
    p->order[p->order_len++] = idx;
    p->job_index[(p->job_head + p->job_count) % PK_MAX_SLOT] = idx;
    p->job_count++;
    p->t_progress = pk_now_ms();
    mp_cond_broadcast(&p->cond);
    mp_mutex_unlock(&p->lock);
    return true;
}

/*
 * 按提交顺序取回已完成的对：把生成帧包成 mpv 的 hw 帧塞进队列。返回取回的
 * 对数；-1 表示引擎报错（filter 已标记失败）。
 */
static int pk_reap(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    int reaped = 0;

    if (getenv("PLAYKIT_TRACE"))
        MP_INFO(f, "playkit: trace reap enter order=%d\n", p->order_len);
    while (p->order_len > 0) {
        const int idx = p->order[0];
        struct pk_slot *s = &b->slot[idx];
        int state;
        mp_mutex_lock(&p->lock);
        state = s->state;
        mp_mutex_unlock(&p->lock);
        if (getenv("PLAYKIT_TRACE"))
            MP_INFO(f, "playkit: trace reap slot=%d state=%d\n", idx, state);
        if (state != PK_SLOT_READY && state != PK_SLOT_ERROR)
            break;

        p->order_len--;
        for (int i = 0; i < p->order_len; i++)
            p->order[i] = p->order[i + 1];

        if (state == PK_SLOT_ERROR) {
            mp_image_unrefp(&s->cur);
            mp_mutex_lock(&p->lock);
            s->state = PK_SLOT_FREE;
            mp_mutex_unlock(&p->lock);
            mp_filter_internal_mark_failed(f);
            return -1;
        }

        if (p->opts->log || getenv("PLAYKIT_TIMING")) {
            MP_INFO(f, "playkit: timing copy_wait=%.2f copy=%.2f process=%.2f bridge wall=%.2f eval=%.2f gpu=%.2f req=%.2f idle=%.2f cycle=%.2f slot=%d\n",
                    s->copy_ms, s->copy_ms, s->process_ms, s->stats.wall_ms,
                    s->stats.evaluate_ms, s->stats.gpu_ms, p->t_req, p->t_idle,
                    p->t_cycle, idx);
        }

        for (int k = 0; k < s->out_count; k++) {
            struct mp_image *img = pk_frame(f, s, k,
                s->pts_prev + (k + 1) * s->step, s->out_fps);
            if (!img) {
                MP_ERR(f, "playkit: cannot wrap generated frame %d\n", k);
                mp_image_unrefp(&s->cur);
                mp_mutex_lock(&p->lock);
                s->state = PK_SLOT_FREE;
                mp_mutex_unlock(&p->lock);
                mp_filter_internal_mark_failed(f);
                return -1;
            }
            if (!p->params_logged) {
                p->params_logged = true;
                if (getenv("PLAYKIT_TIMING"))
                    MP_INFO(f, "playkit: params gen=[%s] src=[%s]\n",
                            mp_image_params_to_str(&img->params),
                            mp_image_params_to_str(&s->cur->params));
            }
            /*
             * Frames from the hw frame pool carry the container's unset
             * colour metadata, so their params differ from the decoded
             * source.  The VO compares params per frame and rebuilds its
             * render pass when they change, which cost ~10 ms every source
             * frame here.  Take the source params instead.
             */
            img->params = s->cur->params;
            p->queue[p->queue_len++] = img;
        }
        /* 这一对的真实当前帧紧跟在生成帧后面。 */
        s->cur->nominal_fps = s->out_fps;
        p->queue[p->queue_len++] = s->cur;
        s->cur = NULL;

        /* blit 还排在 D3D11 队列里，槽位复用前要等它退休。 */
        ID3D11DeviceContext_End(b->ctx, (ID3D11Asynchronous *)s->blit_done);
        s->blit_pending = true;

        mp_mutex_lock(&p->lock);
        s->state = PK_SLOT_FREE;
        p->t_progress = pk_now_ms();
        mp_mutex_unlock(&p->lock);
        reaped++;
    }
    return reaped;
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

// Diagnostic only (PLAYKIT_DSTSTAT): mean/min/max of the NV12 luma plane, read back
// through staging so we see exactly what the VO will sample for this frame.
static void pk_luma_stat(struct mp_filter *f, ID3D11Texture2D *tex, UINT sub,
                         int index, const char *what)
{
    struct pk_bridge *b = ((struct priv *)f->priv)->bridge;
    D3D11_TEXTURE2D_DESC d = {0};
    ID3D11Texture2D_GetDesc(tex, &d);
    D3D11_TEXTURE2D_DESC sd = d;
    sd.ArraySize = 1;
    sd.MipLevels = 1;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.BindFlags = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags = 0;

    ID3D11Texture2D *st = NULL;
    if (FAILED(ID3D11Device_CreateTexture2D(b->dev, &sd, NULL, &st)))
        return;
    ID3D11DeviceContext_CopySubresourceRegion(
        b->ctx, (ID3D11Resource *)st, 0, 0, 0, 0, (ID3D11Resource *)tex, sub,
        NULL);
    ID3D11DeviceContext_Flush(b->ctx);
    pk_wait_misc(f);

    D3D11_MAPPED_SUBRESOURCE map;
    if (SUCCEEDED(ID3D11DeviceContext_Map(b->ctx, (ID3D11Resource *)st, 0,
                                          D3D11_MAP_READ, 0, &map))) {
        // D3D11 hw frames are R8 (NV12) or R16 (P010) luma planes; report 10-bit
        // samples scaled to the same 0-255 range so both depths are comparable.
        const bool wide = d.Format == DXGI_FORMAT_P010 ||
                          d.Format == DXGI_FORMAT_R16_UNORM ||
                          d.Format == DXGI_FORMAT_R16_TYPELESS;
        unsigned long long sum = 0;
        unsigned mn = 255, mx = 0;
        for (UINT y = 0; y < d.Height; y++) {
            const uint8_t *row =
                (const uint8_t *)map.pData + (size_t)y * map.RowPitch;
            for (UINT x = 0; x < d.Width; x++) {
                unsigned v = wide ? (((const uint16_t *)row)[x] >> 8) : row[x];
                sum += v;
                if (v < mn)
                    mn = v;
                if (v > mx)
                    mx = v;
            }
        }
        MP_INFO(f, "playkit: %s[%d] sub=%u y=%.2f min=%u max=%u\n", what,
                index, sub, (double)sum / ((double)d.Width * d.Height), mn, mx);
        ID3D11DeviceContext_Unmap(b->ctx, (ID3D11Resource *)st, 0);
    }
    ID3D11Texture2D_Release(st);
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

static struct mp_image *pk_frame(struct mp_filter *f, struct pk_slot *s,
                                 int index, double pts, double nominal_fps)
{
    struct priv *p = f->priv;
    struct pk_bridge *b = p->bridge;
    if (index < 0 || index >= s->out_count)
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
    if (getenv("PLAYKIT_TRACE"))
        MP_INFO(f, "playkit: trace get_buffer k=%d get=%.2f\n", index, pt1 - pt0);

    ID3D11Texture2D *dst = (ID3D11Texture2D *)av->data[0];
    UINT dst_sub = (UINT)(intptr_t)av->data[1];

    if (p->opts->log) {
        uint64_t src_hash = pk_hash_texture(b, s->out_tex[index], 0);
        MP_INFO(f, "playkit: src[%d] hash=%016llx dst=%p sub=%u\n",
                index, (unsigned long long)src_hash, (void *)dst,
                (unsigned)dst_sub);
        if (index >= 0 && index < 5) {
            static int dumped;
            if (dumped < 5) {
                char path[128];
                snprintf(path, sizeof(path),
                         "E:/code/dlssg-vfi/work/out%d.ppm", index);
                pk_dump_rgba(b, s->out_tex[index], path);
                dumped++;
            }
        }
    }

    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC ivd = {
        .FourCC = 0,
        .ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D,
    };
    ID3D11VideoProcessorInputView *in_view = s->in_view[index];
    HRESULT hr = S_OK;
    if (!in_view) {
        hr = ID3D11VideoDevice_CreateVideoProcessorInputView(
            b->video_dev, (ID3D11Resource *)s->out_tex[index], b->vp_enum,
            &ivd, &in_view);
        if (FAILED(hr)) {
            MP_ERR(f, "playkit: CreateVideoProcessorInputView failed 0x%lx\n",
                   (unsigned long)hr);
            av_frame_free(&av);
            return NULL;
        }
        s->in_view[index] = in_view;
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
    if (getenv("PLAYKIT_TRACE"))
        MP_INFO(f, "playkit: trace blt k=%d blt=%.2f\n", index, pt2 - pt1);
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
    if (getenv("PLAYKIT_DSTSTAT"))
        pk_luma_stat(f, dst, dst_sub, index, "dst");

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

static int pk_free_slot(struct priv *p)
{
    struct pk_bridge *b = p->bridge;
    int idx = -1;

    if (!b)
        return -1;
    mp_mutex_lock(&p->lock);
    for (int i = 0; i < b->slot_count; i++) {
        if (b->slot[i].state == PK_SLOT_FREE) {
            idx = i;
            break;
        }
    }
    mp_mutex_unlock(&p->lock);
    return idx;
}

/*
 * 尽量把输入喂进流水线：每对帧占一个槽位，槽位满就停手（内部缓冲有上限）。
 * 软件路径（引擎不可用、非 D3D11）仍然同步产出重复帧。
 */
static void pk_pump(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct mp_pin *in = f->ppins[0];

    for (;;) {
        if (p->bridge && !p->threads_started)
            pk_threads_start(f);
        int freeslot = p->bridge ? pk_free_slot(p) : -1;
        if (getenv("PLAYKIT_TRACE"))
            MP_INFO(f, "playkit: trace pump free=%d order=%d jobs=%d q=%d/%d started=%d\n",
                    freeslot, p->order_len, p->job_count, p->queue_pos,
                    p->queue_len, p->threads_started);
        if (p->bridge && freeslot < 0)
            return;

        const double t_req0 = pk_now_ms();
        if (!mp_pin_out_request_data(in))
            return;
        struct mp_frame frame = mp_pin_out_read(in);
        p->t_req = pk_now_ms() - t_req0;
        if (frame.type == MP_FRAME_NONE) {
            if (getenv("PLAYKIT_TRACE"))
                MP_INFO(f, "playkit: trace pump no input\n");
            return;
        }

        if (mp_frame_is_signaling(frame)) {
            p->eof_pending = true;
            mp_frame_unref(&frame);
            return;
        }

        if (frame.type != MP_FRAME_VIDEO) {
            mp_frame_unref(&frame);
            mp_filter_internal_mark_failed(f);
            return;
        }

        struct mp_image *cur = frame.data;

        if (p->opts->log && !p->logged) {
            MP_INFO(f, "playkit: %dx%d imgfmt=%s hw_subfmt=%s pts=%.6f multi=%d\n",
                    cur->w, cur->h,
                    mp_imgfmt_to_name(cur->imgfmt),
                    cur->params.hw_subfmt ? mp_imgfmt_to_name(cur->params.hw_subfmt)
                                          : "none",
                    cur->pts, p->opts->multi);
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
            frame.data = cur;
        }

        /* First frame: pass it through and use it as the previous frame. */
        if (!p->prev) {
            p->prev = mp_image_new_ref(cur);
            write_video(f, cur);
            return;
        }

        double step = 0.0;
        if (cur->pts != MP_NOPTS_VALUE && p->prev->pts != MP_NOPTS_VALUE)
            step = (cur->pts - p->prev->pts) / p->opts->multi;

        if (p->opts->log)
            MP_INFO(f, "playkit: in pts=%.6f prev=%.6f multi=%d step=%.6f\n",
                    cur->pts, p->prev->pts, p->opts->multi, step);

        double out_fps = cur->nominal_fps;
        if (!isnan(out_fps) && out_fps > 0)
            out_fps *= p->opts->multi;

        /*
         * D3D11 input: hand the two textures to vdlssg and let it generate the
         * intermediate frames on the GPU.  The filter only copies the shared
         * output textures into mpv's own hw frames.
         */
        if (cur->imgfmt == IMGFMT_D3D11 && p->opts->multi > 1) {
            if (!p->bridge && !p->bridge_failed) {
                if (!pk_bridge_init(f, cur))
                    p->bridge_failed = true;
            }
            if (p->bridge) {
                /* 工作线程必须在第一次提交之前起来（pk_threads_start 会清空 job
                 * 队列，若在提交之后调用就会把在飞的对丢掉）。 */
                if (!p->threads_started)
                    pk_threads_start(f);
                pk_update_colorspace(f, cur);
                const bool submitted = pk_submit(f, p->prev, cur, step, out_fps);
                if (getenv("PLAYKIT_TRACE"))
                    MP_INFO(f, "playkit: trace pump submit=%d\n", (int)submitted);
                if (submitted) {
                    mp_image_unrefp(&p->prev);
                    p->prev = mp_image_new_ref(cur);
                    mp_frame_unref(&frame);
                    continue; /* 继续喂下一对，把槽位填满 */
                }
                /* 没有空闲槽位：等引擎算完再来。 */
                return;
            }
        }

        /*
         * Queue the intermediate frames followed by the current frame.  The
         * current frame's PTS is already the end of the interval, so only the
         * intermediate frames need new timestamps.
         */
        const int multi = p->opts->multi;
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
        return;
    }
}

static void process(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct mp_pin *out = f->ppins[1];
    const double t_entry = pk_now_ms();

    /* 距上一次成对结束的空档；只有 PLAYKIT_TIMING 会在日志里读它。 */
    if (p->t_pair_end > 0)
        p->t_idle = t_entry - p->t_pair_end;

    if (getenv("PLAYKIT_TRACE"))
        MP_INFO(f, "playkit: trace process need=%d order=%d jobs=%d q=%d/%d\n",
                (int)mp_pin_in_needs_data(out), p->order_len, p->job_count,
                p->queue_pos, p->queue_len);
    if (!mp_pin_in_needs_data(out)) {
        /* 输出还不缺数据：顺手把流水线填满，让引擎先算起来。 */
        if (p->bridge && p->queue_pos >= p->queue_len)
            pk_pump(f);
        return;
    }

    /* Drain the queue first. */
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

    /* 队列空：先把已完成的对收回来。 */
    if (p->bridge && p->order_len) {
        pk_reap(f);
        if (p->queue_pos < p->queue_len) {
            struct mp_image *img = p->queue[p->queue_pos];
            p->queue[p->queue_pos] = NULL;
            p->queue_pos++;
            p->t_cycle = p->t_pair_end > 0 ? pk_now_ms() - p->t_pair_end : 0;
            p->t_pair_end = pk_now_ms();
            write_video(f, img);
            return;
        }
    }

    /* The queue is empty; a pending EOF can be forwarded now. */
    if (p->eof_pending && !p->order_len) {
        p->eof_pending = false;
        mp_pin_in_write(out, MP_EOF_FRAME);
        return;
    }

    pk_pump(f);
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
        .description = "PlayKit frame multiplier (D3D11 zero-copy DLSS-G)",
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
