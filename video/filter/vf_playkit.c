/*
 * PlayKit frame multiplier (pass-through prototype)
 *
 * For every input frame pair, emit `multi` output frames.  The prototype
 * duplicates the current frame and assigns evenly spaced PTS values.  It is
 * meant to validate D3D11 frame handling, PTS interpolation, queueing and EOF
 * before the real DLSS-G / aji zero-copy path is plugged in.
 */

#include <math.h>
#include <talloc.h>

#include "common/common.h"
#include "common/msg.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/frame.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "video/img_format.h"
#include "video/mp_image.h"

struct opts {
    int multi;
    int log;
};

#define OPT_BASE_STRUCT struct opts

static const m_option_t vf_opts_fields[] = {
    {"multi", OPT_INT(multi)},
    {"log", OPT_BOOL(log)},
    {0}
};

struct priv {
    struct opts *opts;

    struct mp_image *prev;
    struct mp_image *queue[6];
    int queue_len;
    int queue_pos;

    bool logged;
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
    mp_image_unrefp(&p->prev);
}

static void destroy(struct mp_filter *f)
{
    reset(f);
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
        mp_pin_in_write(out, MAKE_FRAME(MP_FRAME_VIDEO, img));
        return;
    }

    if (!mp_pin_out_request_data(in))
        return;

    struct mp_frame frame = mp_pin_out_read(in);
    if (frame.type == MP_FRAME_NONE)
        return;

    if (mp_frame_is_signaling(frame)) {
        mp_pin_in_write(out, frame);
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

    /* First frame: pass it through and use it as the previous frame. */
    if (!p->prev) {
        p->prev = mp_image_new_ref(cur);
        mp_pin_in_write(out, frame);
        return;
    }

    double step = 0.0;
    if (cur->pts != MP_NOPTS_VALUE && p->prev->pts != MP_NOPTS_VALUE)
        step = (cur->pts - p->prev->pts) / multi;

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
    mp_pin_in_write(out, MAKE_FRAME(MP_FRAME_VIDEO, first));
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
