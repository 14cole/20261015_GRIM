/* GHOST 2-D near-pair arithmetic. No third-party source.
   Mirrors, operation for operation, the NumPy expressions of the close-pair
   box recursion (operators._integrate_linear_boxes_sk) and of the thin-gap
   moments (assembly.thin_gap._inner). Transcendental functions stay with the
   caller: Hankel and Bessel values, logarithms and arctangents are evaluated
   by NumPy/SciPy and passed in, so the native code only does the IEEE
   arithmetic (+, -, *, /, sqrt) whose results do not depend on the library.

   NumPy's complex product on FMA hardware rounds as re = fma(ar, br, -(ai*bi)),
   im = fma(ar, bi, ai*br); `fused` selects that form (the caller probes NumPy),
   otherwise the plain two-rounding form. A real operand of a complex product
   is the complex number (x, 0), as NumPy casts it. NumPy's complex division is
   Smith's algorithm, its complex row sums are its pairwise summation, and
   einsum's three-operand complex product is the plain left-to-right product
   accumulated in input order. -ffp-contract=off keeps all of this exact.
   Every call owns its outputs; inputs are read only, so calls may run
   concurrently. */
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define GHOST_EPS 1e-12
#define GHOST_PI 3.14159265358979323846
#define MAX_ORDER 64
#define MAX_DEGREE 3
#define MAX_TERMS 3
#define MAX_INNER 64
#define INLINE static inline __attribute__((always_inline))

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define GHOST_HAVE_FUSED 1
#define FMA_TARGET __attribute__((target("fma")))
#else
#define GHOST_HAVE_FUSED 0
#define FMA_TARGET
#endif

typedef struct { double re, im; } cplx;

/* NumPy's complex product a * b. */
INLINE cplx cmul(cplx a, cplx b, int fused) {
    cplx out;
    if (fused) {
        out.re = __builtin_fma(a.re, b.re, -(a.im * b.im));
        out.im = __builtin_fma(a.re, b.im, a.im * b.re);
    } else {
        out.re = a.re * b.re - a.im * b.im;
        out.im = a.re * b.im + a.im * b.re;
    }
    return out;
}

INLINE cplx creal_(double x) { cplx out = {x, 0.0}; return out; }
INLINE cplx cadd(cplx a, cplx b) { cplx out = {a.re + b.re, a.im + b.im}; return out; }
INLINE cplx csub(cplx a, cplx b) { cplx out = {a.re - b.re, a.im - b.im}; return out; }

/* NumPy's complex a / b (Smith's algorithm, with its zero-divisor branch). */
INLINE cplx cdiv(cplx a, cplx b) {
    double br_abs = fabs(b.re), bi_abs = fabs(b.im);
    cplx out;
    if (br_abs >= bi_abs) {
        if (br_abs == 0 && bi_abs == 0) {
            out.re = a.re / br_abs;
            out.im = a.im / bi_abs;
        } else {
            double rat = b.im / b.re;
            double scl = 1.0 / (b.re + b.im * rat);
            out.re = (a.re + a.im * rat) * scl;
            out.im = (a.im - a.re * rat) * scl;
        }
    } else {
        double rat = b.re / b.im;
        double scl = 1.0 / (b.im + b.re * rat);
        out.re = (a.re * rat + a.im) * scl;
        out.im = (a.im * rat - a.re) * scl;
    }
    return out;
}

/* NumPy's pairwise sum of n interleaved doubles (n/2 complex values). */
static void pairwise(const double *a, int64_t n, double *rr, double *ri) {
    if (n < 8) {
        *rr = -0.0;
        *ri = -0.0;
        for (int64_t i = 0; i < n; i += 2) {
            *rr += a[i];
            *ri += a[i + 1];
        }
    } else if (n <= 128) {
        double r[8];
        int64_t i;
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        *rr = ((r[0] + r[2]) + (r[4] + r[6]));
        *ri = ((r[1] + r[3]) + (r[5] + r[7]));
        for (; i < n; i += 2) {
            *rr += a[i];
            *ri += a[i + 1];
        }
    } else {
        double rr1, ri1, rr2, ri2;
        int64_t n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(a, n2, &rr1, &ri1);
        pairwise(a + n2, n - n2, &rr2, &ri2);
        *rr = rr1 + rr2;
        *ri = ri1 + ri2;
    }
}

/* np.sum(values, axis=-1) of one row of `count` complex values. */
INLINE cplx row_sum(const double *values, int count) {
    double rr, ri;
    pairwise(values, 2 * (int64_t)count, &rr, &ri);
    cplx out = {0.0 + rr, 0.0 + ri};
    return out;
}

INLINE double maximum(double x, double y) { return (x >= y || isnan(x)) ? x : y; }

/* ------------------------------------------------------------------------ */
/* Box recursion: (n, q, q) tensor-Gauss boxes of linear element pairs.       */
/* ------------------------------------------------------------------------ */

typedef struct {
    double xi_obs[MAX_ORDER], xi_src[MAX_ORDER];
    double obs[MAX_ORDER][2], src[MAX_ORDER][2];
    double obs_len, src_len;
} box_t;

INLINE void box_points(int q, const double *iv, const double *qt, const double *op0, const double *oseg,
                       const double *sp0, const double *sseg, double olength, double slength, box_t *box) {
    double obs_scale = maximum(iv[1] - iv[0], 0.0);
    double src_scale = maximum(iv[3] - iv[2], 0.0);
    box->obs_len = olength * obs_scale;
    box->src_len = slength * src_scale;
    for (int i = 0; i < q; ++i) {
        double xo = iv[0] + obs_scale * qt[i], xs = iv[2] + src_scale * qt[i];
        box->xi_obs[i] = xo;
        box->xi_src[i] = xs;
        box->obs[i][0] = op0[0] + xo * oseg[0];
        box->obs[i][1] = op0[1] + xo * oseg[1];
        box->src[i][0] = sp0[0] + xs * sseg[0];
        box->src[i][1] = sp0[1] + xs * sseg[1];
    }
}

/* Returns 0, or 2 when |kr| is too close to the 1e-12 floor to decide it
   without NumPy's complex absolute value (the caller then uses NumPy). */
INLINE int box_arguments_impl(int64_t n, int q, const int64_t *pair, const double *intervals,
    const double *qt, const double *obs_p0, const double *obs_seg, const double *src_p0,
    const double *src_seg, double k_re, double k_im, int fused, double *kr, uint8_t *real_box) {
    cplx k = {k_re, k_im};
    for (int64_t row = 0; row < n; ++row) {
        int64_t p = pair[row];
        box_t box;
        box_points(q, intervals + 4 * row, qt, obs_p0 + 2 * p, obs_seg + 2 * p, src_p0 + 2 * p,
                   src_seg + 2 * p, 0.0, 0.0, &box);
        int real = 1;
        double *out = kr + 2 * row * q * q;
        for (int i = 0; i < q; ++i) {
            for (int j = 0; j < q; ++j) {
                double dx = box.obs[i][0] - box.src[j][0], dy = box.obs[i][1] - box.src[j][1];
                double dist = sqrt(0.0 + (dx * dx + dy * dy));
                cplx value = cmul(k, creal_(maximum(dist, GHOST_EPS)), fused);
                double magnitude = value.im == 0.0 ? fabs(value.re) : hypot(value.re, value.im);
                if (value.im != 0.0 && fabs(magnitude - 1e-12) <= 1e-26) return 2;
                if (magnitude <= 1e-12) {
                    value.re = 1e-12;
                    value.im = 0.0;
                }
                real &= (fabs(value.im) <= 1e-14) && (value.re >= 0.0);
                out[2 * (i * q + j)] = value.re;
                out[2 * (i * q + j) + 1] = value.im;
            }
        }
        real_box[row] = (uint8_t)real;
    }
    return 0;
}

/* acc += ((w * phi_o) * phi_s) as einsum's plain complex product, with the
   real basis values cast to complex. */
INLINE void einsum_add(cplx *acc, cplx w, double x, double y) {
    double re = w.re, im = w.im, tmp;
    tmp = re * x - im * 0.0;
    im = re * 0.0 + im * x;
    re = tmp;
    tmp = re * y - im * 0.0;
    im = re * 0.0 + im * y;
    re = tmp;
    acc->re = re + acc->re;
    acc->im = im + acc->im;
}

INLINE int box_blocks_impl(int64_t n, int q, const int64_t *pair, const double *intervals,
    const double *qt, const double *qw, const double *obs_p0, const double *obs_seg,
    const double *obs_length, const double *src_p0, const double *src_seg, const double *src_length,
    const double *normal, double c_re, double c_im, const double *h0, const double *h1,
    int fused, double *s_out, double *k_out) {
    const cplx quarter_j = {0.0, 0.25}, c = {c_re, c_im};
    for (int64_t row = 0; row < n; ++row) {
        int64_t p = pair[row];
        box_t box;
        box_points(q, intervals + 4 * row, qt, obs_p0 + 2 * p, obs_seg + 2 * p, src_p0 + 2 * p,
                   src_seg + 2 * p, obs_length[p], src_length[p], &box);
        const double *nrm = normal + 2 * p;
        cplx s[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
        cplx kb[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
        for (int i = 0; i < q; ++i) {
            double po[2] = {1.0 - box.xi_obs[i], box.xi_obs[i]};
            for (int j = 0; j < q; ++j) {
                double ps[2] = {1.0 - box.xi_src[j], box.xi_src[j]};
                double w = qw[i] * qw[j];
                int64_t at = 2 * ((row * q + i) * q + j);
                if (h0) {
                    cplx g = {h0[at], h0[at + 1]};
                    cplx weighted = cmul(creal_(w), cmul(quarter_j, g, fused), fused);
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b) einsum_add(&s[2 * a + b], weighted, po[a], ps[b]);
                }
                if (h1) {
                    double dx = box.obs[i][0] - box.src[j][0], dy = box.obs[i][1] - box.src[j][1];
                    double dist = sqrt(0.0 + (dx * dx + dy * dy));
                    /* np.sum over the two components: the zero identity, then the pair. */
                    double proj = (0.0 + (dx * nrm[0] + dy * nrm[1])) / maximum(dist, GHOST_EPS);
                    cplx h = {h1[at], h1[at + 1]};
                    cplx dk = cmul(cmul(c, h, fused), creal_(proj), fused);
                    if (dist <= GHOST_EPS) dk.re = dk.im = 0.0;
                    cplx weighted = cmul(creal_(w), dk, fused);
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b) einsum_add(&kb[2 * a + b], weighted, po[a], ps[b]);
                }
            }
        }
        double scale = box.obs_len * box.src_len;
        int empty = box.obs_len <= 0.0 || box.src_len <= 0.0;
        for (int e = 0; e < 4; ++e) {
            cplx sv = cmul(s[e], creal_(scale), fused), kv = cmul(kb[e], creal_(scale), fused);
            if (empty) sv.re = sv.im = kv.re = kv.im = 0.0;
            s_out[2 * (4 * row + e)] = sv.re;
            s_out[2 * (4 * row + e) + 1] = sv.im;
            k_out[2 * (4 * row + e)] = kv.re;
            k_out[2 * (4 * row + e) + 1] = kv.im;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Thin-gap inner integrals (assembly.thin_gap._inner) at outer points.       */
/* ------------------------------------------------------------------------ */

/* log_series: sum_j series[j] * moment_j with
   moment_j = sum_l comb(j, l) d2^(j-l) sum_i poly[i] b[2l + i]. */
INLINE cplx log_series(int terms, const cplx *series, const double *poly, int npoly,
                       const double *b, double d2, int fused) {
    static const double comb[MAX_TERMS][MAX_TERMS] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};
    cplx total = {0.0, 0.0};
    for (int j = 0; j < terms; ++j) {
        double moment = 0.0;
        for (int l = 0; l <= j; ++l) {
            int e = j - l;
            double power = e == 0 ? 1.0 : (e == 1 ? d2 : d2 * d2);
            double inner = 0.0 + poly[0] * b[2 * l];
            for (int i = 1; i < npoly; ++i) inner = inner + poly[i] * b[2 * l + i];
            moment = moment + (comb[j][l] * power) * inner;
        }
        cplx z = cmul(series[j], creal_(moment), fused);
        total = j == 0 ? cadd(creal_(0.0), z) : cadd(total, z);
    }
    return total;
}

INLINE double sign(double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : (x == 0 ? 0.0 : x)); }

INLINE int thin_analytic_impl(int64_t count, int degree, int terms,
    const double *t, const double *h, const double *delta, const double *d2v,
    const double *u0v, const double *u1v, const double *anglev, const double *l0v, const double *l1v,
    const double *length, const double *t3, const double *l3,
    const double *m0v, const double *m1v, const double *sigmav,
    const double *s_series, const double *v_series, int fused, double *s_out, double *k_out) {
    const int qmax = degree + 1 + 2 * (terms - 1), size = degree + 1;
    static const double comb[MAX_DEGREE + 1][MAX_DEGREE + 1] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};
    const double s_const = 1. / (4. * GHOST_PI), k_const = 1. / (2. * GHOST_PI);
    cplx ss[MAX_TERMS], vs[MAX_TERMS];
    for (int j = 0; j < terms; ++j) {
        ss[j].re = s_series[2 * j];
        ss[j].im = s_series[2 * j + 1];
        vs[j].re = v_series[2 * j];
        vs[j].im = v_series[2 * j + 1];
    }
    for (int64_t p = 0; p < count; ++p) {
        double d2 = d2v[p], u0 = u0v[p], u1 = u1v[p], l0 = l0v[p], l1 = l1v[p];
        double angle = anglev[p], hp = h[p], m0 = m0v[p], m1 = m1v[p];
        double power0[16], power1[16], c[16], b[16];
        power0[0] = power1[0] = 1.0;
        for (int i = 1; i <= qmax + 2; ++i) {
            power0[i] = power0[i - 1] * u0;
            power1[i] = power1[i - 1] * u1;
        }
        c[1] = .5 * (l1 - l0);
        for (int q = 2; q <= qmax + 2; ++q)
            c[q] = (power1[q - 1] - power0[q - 1]) / (double)(q - 1) - (q == 2 ? delta[p] * angle : d2 * c[q - 2]);
        for (int q = 0; q <= qmax; ++q) {
            double e1 = power1[q + 1] == 0. ? 0. : power1[q + 1] * l1;
            double e0 = power0[q + 1] == 0. ? 0. : power0[q + 1] * l0;
            b[q] = ((e1 - e0) - 2. * c[q + 2]) / (double)(q + 1);
        }
        /* (s/L)^m = ((u + t)/L)^m: comb(m, i) t^(m-i) / L^m; NumPy squares
           exactly and passes its cubes (a library power). */
        double tp[MAX_DEGREE + 1] = {1.0, t[p], t[p] * t[p], degree == 3 ? t3[p] : 0.0};
        double lp[MAX_DEGREE + 1] = {1.0, length[p], length[p] * length[p], degree == 3 ? l3[p] : 0.0};
        double mono[(MAX_DEGREE + 1) * (MAX_DEGREE + 2) / 2];
        for (int m = 0, at = 0; m < size; ++m)
            for (int i = 0; i <= m; ++i) mono[at++] = comb[m][i] * tp[m - i] / lp[m];
        double hm0 = hp * m0, nm1 = -m1, sh = sign(hp);
        for (int m = 0, offset = 0; m < size; offset += ++m) {
            const double *poly = mono + offset;
            if (s_out) {
                cplx v = cmul(creal_(s_const), log_series(terms, ss, poly, m + 1, b, d2, fused), fused);
                s_out[2 * (p * size + m)] = v.re;
                s_out[2 * (p * size + m) + 1] = v.im;
            }
            if (k_out) {
                double singular = m0 * poly[0] * (sh * angle) - m1 * poly[0] * c[1];
                for (int i = 1; i <= m; ++i)
                    singular = singular + poly[i] * (hm0 * c[i] - m1 * c[i + 1]);
                double product[MAX_DEGREE + 2];
                for (int i = 0; i < m + 2; ++i) product[i] = 0.;
                for (int i = 0; i <= m; ++i) {
                    product[i] = product[i] + poly[i] * hm0;
                    product[i + 1] = product[i + 1] + poly[i] * nm1;
                }
                cplx half = cmul(creal_(.5), log_series(terms, vs, product, m + 2, b, d2, fused), fused);
                cplx v = cmul(creal_(sigmav[p] * k_const), csub(creal_(singular), half), fused);
                k_out[2 * (p * size + m)] = v.re;
                k_out[2 * (p * size + m) + 1] = v.im;
            }
        }
    }
    return 0;
}

/* Remainder quadrature nodes: the inner panel split at the foot point. */
INLINE void thin_nodes(double t, double length, int nq, const double *qt, const double *qw,
                       double *s, double *w) {
    double split = (t > 0. && t < length) ? t : .5 * length;
    double starts[2] = {0.0, split}, widths[2] = {split, length - split};
    for (int half = 0; half < 2; ++half)
        for (int i = 0; i < nq; ++i) {
            s[half * nq + i] = starts[half] + widths[half] * qt[i];
            w[half * nq + i] = widths[half] * qw[i];
        }
}

INLINE int thin_squares_impl(int64_t count, int nq, const double *t, const double *length,
    const double *d2, const double *qt, const double *qw, double *r2) {
    double s[2 * MAX_INNER], w[2 * MAX_INNER];
    for (int64_t p = 0; p < count; ++p) {
        thin_nodes(t[p], length[p], nq, qt, qw, s, w);
        for (int i = 0; i < 2 * nq; ++i) {
            double u = s[i] - t[p];
            r2[p * 2 * nq + i] = u * u + d2[p];
        }
    }
    return 0;
}

INLINE int thin_remainder_impl(int64_t count, int degree, int terms, int nq,
    const double *t, const double *length, const double *d2v, const double *h, const double *m0v,
    const double *m1v, const double *sigmav, const double *qt, const double *qw,
    const double *rv, const double *log_r2v, const double *green, const double *derivative,
    double kk_re, double kk_im, const double *s_series, const double *v_series, int fused,
    double *s_out, double *k_out) {
    const int size = degree + 1, nodes = 2 * nq;
    const double s_const = 1. / (4. * GHOST_PI), k_const = 1. / (2. * GHOST_PI);
    const cplx kk = {kk_re, kk_im};
    cplx vs[MAX_TERMS];
    for (int j = 0; j < terms; ++j) {
        vs[j].re = v_series[2 * j];
        vs[j].im = v_series[2 * j + 1];
    }
    double s[2 * MAX_INNER], w[2 * MAX_INNER], weights[MAX_DEGREE + 1][2 * MAX_INNER];
    double sv[MAX_DEGREE + 1][4 * MAX_INNER], kv[MAX_DEGREE + 1][4 * MAX_INNER];
    for (int64_t p = 0; p < count; ++p) {
        double tp = t[p], lp = length[p], d2 = d2v[p], hm0 = h[p] * m0v[p], m1 = m1v[p];
        thin_nodes(tp, lp, nq, qt, qw, s, w);
        for (int i = 0; i < nodes; ++i) {
            double ratio = s[i] / lp;
            weights[0][i] = w[i];
            for (int m = 1; m < size; ++m) weights[m][i] = weights[m - 1][i] * ratio;
        }
        for (int i = 0; i < nodes; ++i) {
            int64_t at = p * nodes + i;
            double u = s[i] - tp, r2 = u * u + d2, log_r2 = log_r2v[at];
            if (s_out) {
                cplx x = cmul(kk, creal_(r2), fused);
                cplx powers[MAX_TERMS] = {{1.0, 0.0}, x, cmul(x, x, fused)};
                cplx series = {0.0, 0.0};
                for (int j = 0; j < terms; ++j) {
                    cplx term = cmul(creal_(s_series[j]), powers[j], fused);
                    series = j == 0 ? cadd(creal_(0.0), term) : cadd(series, term);
                }
                cplx g = {green[2 * at], green[2 * at + 1]};
                cplx rem = csub(g, cmul(cmul(creal_(s_const), series, fused), creal_(log_r2), fused));
                for (int m = 0; m < size; ++m) {
                    cplx v = cmul(creal_(weights[m][i]), rem, fused);
                    sv[m][2 * i] = v.re;
                    sv[m][2 * i + 1] = v.im;
                }
            }
            if (k_out) {
                double r2_powers[MAX_TERMS] = {1.0, r2, r2 * r2};
                cplx series = {0.0, 0.0};
                for (int j = 0; j < terms; ++j) {
                    cplx term = cmul(vs[j], creal_(r2_powers[j]), fused);
                    series = j == 0 ? cadd(creal_(0.0), term) : cadd(series, term);
                }
                cplx minus = {-derivative[2 * at], -derivative[2 * at + 1]};
                cplx scaled = cmul(cmul(creal_(.5), series, fused), creal_(log_r2), fused);
                cplx bracket = csub(creal_(1. / r2), scaled);
                cplx v = csub(cdiv(minus, creal_(rv[at])), cmul(creal_(k_const), bracket, fused));
                v = cmul(v, creal_(hm0 - u * m1), fused);
                for (int m = 0; m < size; ++m) {
                    cplx value = cmul(creal_(weights[m][i]), v, fused);
                    kv[m][2 * i] = value.re;
                    kv[m][2 * i + 1] = value.im;
                }
            }
        }
        for (int m = 0; m < size; ++m) {
            if (s_out) {
                cplx total = row_sum(sv[m], nodes);
                s_out[2 * (p * size + m)] = total.re;
                s_out[2 * (p * size + m) + 1] = total.im;
            }
            if (k_out) {
                cplx total = cmul(creal_(sigmav[p]), row_sum(kv[m], nodes), fused);
                k_out[2 * (p * size + m)] = total.re;
                k_out[2 * (p * size + m) + 1] = total.im;
            }
        }
    }
    return 0;
}

/* einsum('nqa,nqb->nab', outer, values): real outer weights cast to complex,
   einsum's plain two-operand product accumulated over q in order. */
EXPORT int ghost_thin_contract(int64_t n, int q, int size, const double *outer, const double *values,
                               double *out) {
    if (n < 0 || q < 1 || size < 1 || size > MAX_DEGREE + 1) return 3;
    for (int64_t row = 0; row < n; ++row) {
        double *block = out + 2 * row * size * size;
        for (int e = 0; e < 2 * size * size; ++e) block[e] = 0.0;
        for (int iq = 0; iq < q; ++iq) {
            const double *o = outer + (row * q + iq) * size;
            const double *v = values + 2 * (row * q + iq) * size;
            for (int a = 0; a < size; ++a)
                for (int b = 0; b < size; ++b) {
                    double re = o[a] * v[2 * b] - 0.0 * v[2 * b + 1];
                    double im = o[a] * v[2 * b + 1] + 0.0 * v[2 * b];
                    block[2 * (a * size + b)] = re + block[2 * (a * size + b)];
                    block[2 * (a * size + b) + 1] = im + block[2 * (a * size + b) + 1];
                }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Exports: plain and fused instantiations of each body.                      */
/* ------------------------------------------------------------------------ */

#define BOX_ARGUMENTS_ARGS int64_t n, int q, const int64_t *pair, const double *intervals,       \
    const double *qt, const double *obs_p0, const double *obs_seg, const double *src_p0,         \
    const double *src_seg, double k_re, double k_im, double *kr, uint8_t *real_box
#define BOX_ARGUMENTS_CALL n, q, pair, intervals, qt, obs_p0, obs_seg, src_p0, src_seg, k_re, k_im
#define BOX_BLOCKS_ARGS int64_t n, int q, const int64_t *pair, const double *intervals,          \
    const double *qt, const double *qw, const double *obs_p0, const double *obs_seg,             \
    const double *obs_length, const double *src_p0, const double *src_seg,                       \
    const double *src_length, const double *normal, double c_re, double c_im,                    \
    const double *h0, const double *h1, double *s_out, double *k_out
#define BOX_BLOCKS_CALL n, q, pair, intervals, qt, qw, obs_p0, obs_seg, obs_length, src_p0,     \
    src_seg, src_length, normal, c_re, c_im, h0, h1
#define ANALYTIC_ARGS int64_t count, int degree, int terms, const double *t, const double *h,     \
    const double *delta, const double *d2, const double *u0, const double *u1,                   \
    const double *angle, const double *l0, const double *l1, const double *length,               \
    const double *t3, const double *l3,                                                          \
    const double *m0, const double *m1, const double *sigma, const double *s_series,             \
    const double *v_series, double *s_out, double *k_out
#define ANALYTIC_CALL count, degree, terms, t, h, delta, d2, u0, u1, angle, l0, l1, length, t3, l3, \
    m0, m1, sigma, s_series, v_series
#define REMAINDER_ARGS int64_t count, int degree, int terms, int nq, const double *t,             \
    const double *length, const double *d2, const double *h, const double *m0,                   \
    const double *m1, const double *sigma, const double *qt, const double *qw,                   \
    const double *r, const double *log_r2, const double *green, const double *derivative,        \
    double kk_re, double kk_im, const double *s_series, const double *v_series,                  \
    double *s_out, double *k_out
#define REMAINDER_CALL count, degree, terms, nq, t, length, d2, h, m0, m1, sigma, qt, qw, r,      \
    log_r2, green, derivative, kk_re, kk_im, s_series, v_series

static int box_arguments_plain(BOX_ARGUMENTS_ARGS) {
    return box_arguments_impl(BOX_ARGUMENTS_CALL, 0, kr, real_box);
}
static int box_blocks_plain(BOX_BLOCKS_ARGS) { return box_blocks_impl(BOX_BLOCKS_CALL, 0, s_out, k_out); }
static int analytic_plain(ANALYTIC_ARGS) { return thin_analytic_impl(ANALYTIC_CALL, 0, s_out, k_out); }
static int remainder_plain(REMAINDER_ARGS) { return thin_remainder_impl(REMAINDER_CALL, 0, s_out, k_out); }
#if GHOST_HAVE_FUSED
static FMA_TARGET int box_arguments_fused(BOX_ARGUMENTS_ARGS) {
    return box_arguments_impl(BOX_ARGUMENTS_CALL, 1, kr, real_box);
}
static FMA_TARGET int box_blocks_fused(BOX_BLOCKS_ARGS) { return box_blocks_impl(BOX_BLOCKS_CALL, 1, s_out, k_out); }
static FMA_TARGET int analytic_fused(ANALYTIC_ARGS) { return thin_analytic_impl(ANALYTIC_CALL, 1, s_out, k_out); }
static FMA_TARGET int remainder_fused(REMAINDER_ARGS) { return thin_remainder_impl(REMAINDER_CALL, 1, s_out, k_out); }
#endif

/* Status: 0 done, 2 the caller must use NumPy for this batch, 3 invalid
   arguments, 4 a fused product was requested without fused support. */
EXPORT int ghost_near_box_arguments(BOX_ARGUMENTS_ARGS, int fused) {
    if (n < 0 || q < 1 || q > MAX_ORDER || !pair || !intervals || !kr || !real_box) return 3;
    if (fused && !GHOST_HAVE_FUSED) return 4;
#if GHOST_HAVE_FUSED
    if (fused) return box_arguments_fused(BOX_ARGUMENTS_CALL, kr, real_box);
#endif
    return box_arguments_plain(BOX_ARGUMENTS_CALL, kr, real_box);
}

EXPORT int ghost_near_box_blocks(BOX_BLOCKS_ARGS, int fused) {
    if (n < 0 || q < 1 || q > MAX_ORDER || (!h0 && !h1) || !s_out || !k_out) return 3;
    if (fused && !GHOST_HAVE_FUSED) return 4;
#if GHOST_HAVE_FUSED
    if (fused) return box_blocks_fused(BOX_BLOCKS_CALL, s_out, k_out);
#endif
    return box_blocks_plain(BOX_BLOCKS_CALL, s_out, k_out);
}

EXPORT int ghost_thin_analytic(ANALYTIC_ARGS, int fused) {
    if (count < 0 || degree < 0 || degree > MAX_DEGREE || terms < 1 || terms > MAX_TERMS ||
        (!s_out && !k_out)) return 3;
    if (fused && !GHOST_HAVE_FUSED) return 4;
#if GHOST_HAVE_FUSED
    if (fused) return analytic_fused(ANALYTIC_CALL, s_out, k_out);
#endif
    return analytic_plain(ANALYTIC_CALL, s_out, k_out);
}

EXPORT int ghost_thin_squares(int64_t count, int nq, const double *t, const double *length,
                              const double *d2, const double *qt, const double *qw, double *r2) {
    if (count < 0 || nq < 1 || nq > MAX_INNER || !r2) return 3;
    return thin_squares_impl(count, nq, t, length, d2, qt, qw, r2);
}

EXPORT int ghost_thin_remainder(REMAINDER_ARGS, int fused) {
    if (count < 0 || degree < 0 || degree > MAX_DEGREE || terms < 1 || terms > MAX_TERMS ||
        nq < 1 || nq > MAX_INNER || (!s_out && !k_out) || (s_out && !green) ||
        (k_out && !derivative)) return 3;
    if (fused && !GHOST_HAVE_FUSED) return 4;
#if GHOST_HAVE_FUSED
    if (fused) return remainder_fused(REMAINDER_CALL, s_out, k_out);
#endif
    return remainder_plain(REMAINDER_CALL, s_out, k_out);
}

/* 1 when this build can form NumPy's fused complex product. */
EXPORT int ghost_near_has_fused(void) { return GHOST_HAVE_FUSED; }
