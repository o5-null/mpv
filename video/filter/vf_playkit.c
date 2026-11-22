/*
 * PlayKit frame multiplier (pass-through prototype)
 *
 * For every input frame pair, emit `multi` output frames.  The prototype
 * duplicates the current frame and assigns evenly spaced PTS values.  It is
 * meant to validate D3D11 frame handling, PTS interpolation, queueing and EOF
 * before the real DLSS-G / aji zero-copy path is plugged in.
 */

#include <math.h>

#include "common/common.h"
#include "common/msg.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/frame.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/sws_utils.h"

struct opts {
    int multi;
    bool log;
    int outfmt;
};

#define OPT_BASE_STRUCT struct opts

static const m_option_t vf_opts_fields[] = {
    {"multi", OPT_INT(multi)},
    {"log", OPT_BOOL(log)},
    {"outfmt", OPT_IMAGEFORMAT(outfmt)},
    {0}
};

struct priv {
    struct opts *opts;
    struct mp_sws_context *sws;

    struct mp_image *prev;
    struct mp_image *queue[6];
    int queue_len;
    int queue_pos;

    bool eof_pending;
    bool logged;
    int out_count;
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
    p->out_count = 0;
    mp_image_unrefp(&p->prev);
}

static void destroy(struct mp_filter *f)
{
    reset(f);
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
