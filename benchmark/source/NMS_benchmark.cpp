// #include "benchmark.hpp"
// #include "reference_cv.hpp"
// #include "riscv_cv.hpp"
// #include "NMS.hpp"
// #include <cstdio>
// #include <cstdint>

// // ---------------- limits (static storage, no heap) ----------------
// static const int NUM_BATCHES = 2;
// static const int NUM_CLASSES = 3;
// static const int MAX_DIM     = 128;
// static const int64_t MAX_OUT = 10;
// static const int MAX_ROWS    = NUM_BATCHES * NUM_CLASSES * (MAX_OUT < MAX_DIM ? (int)MAX_OUT : MAX_DIM);

// static float   g_boxes[NUM_BATCHES * MAX_DIM * 4];
// static float   g_scores[NUM_BATCHES * NUM_CLASSES * MAX_DIM];
// static int64_t g_out_ref[MAX_ROWS * 3];
// static int64_t g_out_vec[MAX_ROWS * 3];

// // ---------------- deterministic generator (xorshift32) ----------------
// static uint32_t g_state = 12345u;
// static inline uint32_t Rand32()
// {
//     g_state ^= g_state << 13;
//     g_state ^= g_state >> 17;
//     g_state ^= g_state << 5;
//     return g_state;
// }
// static inline float RandFloat(float lo, float hi)
// {
//     return lo + (hi - lo) * ((Rand32() & 0xFFFFFF) / 16777216.0f);
// }

// // Boxes come in clusters so many overlap and suppression really happens.
// // Corners = [y1,x1,y2,x2], Center = [xc,yc,w,h].
// static void FillBoxes(int num_batches, int spatial_dim, CenterPointBox fmt)
// {
//     const int num_clusters = 4;
//     for (int b = 0; b < num_batches; ++b)
//     {
//         for (int i = 0; i < spatial_dim; ++i)
//         {
//             int k    = Rand32() % num_clusters;
//             float cx = 20.0f + 25.0f * k + RandFloat(-3.0f, 3.0f);
//             float cy = 20.0f + 25.0f * k + RandFloat(-3.0f, 3.0f);
//             float w  = RandFloat(15.0f, 25.0f);
//             float h  = RandFloat(15.0f, 25.0f);

//             float* p = &g_boxes[(b * spatial_dim + i) * 4];
//             if (fmt == CenterPointBox::Corners)
//             {
//                 p[0] = cy - h / 2; p[1] = cx - w / 2;
//                 p[2] = cy + h / 2; p[3] = cx + w / 2;
//             }
//             else
//             {
//                 p[0] = cx; p[1] = cy; p[2] = w; p[3] = h;
//             }
//         }
//     }
// }

// // Scores quantized to 1/64 steps so exact ties occur (checks tie-breaking matches).
// static void FillScores(int count)
// {
//     for (int i = 0; i < count; ++i)
//         g_scores[i] = (Rand32() % 64) / 64.0f;
// }

// static void FillOut(int64_t* out, int rows)
// {
//     for (int i = 0; i < rows * 3; ++i) out[i] = -1;
// }

// static bool SameOut(const int64_t* a, const int64_t* b, int rows)
// {
//     for (int i = 0; i < rows * 3; ++i)
//         if (a[i] != b[i]) return false;
//     return true;
// }

// int main()
// {
// #ifdef RISCV_BAREMETAL
//     riscv_enable_vector();
// #endif

//     const int   loop_count = 5;
//     const float iou_thr    = 0.5f;
//     const float score_thr  = 0.3f;

//     // vec::NonMaxSuppression is single-strip: spatial_dim must fit one m4 register group.
//     const int vlmax  = (int)__riscv_vsetvlmax_e32m4();
//     const int dims[] = {8, 16, 32, 64, 128};

//     const CenterPointBox fmts[]      = {CenterPointBox::Corners, CenterPointBox::Center};
//     const char*          fmt_names[] = {"CORNERS", "CENTER"};

//     printf("NMS benchmark: %d batches, %d classes, %d iterations (+1 warm-up), VLMAX(e32m4)=%d\n",
//            NUM_BATCHES, NUM_CLASSES, loop_count, vlmax);

//     bool all_correct = true;

//     for (int f = 0; f < 2 && all_correct; ++f)
//     {
//         for (int d = 0; d < (int)(sizeof(dims) / sizeof(dims[0])) && all_correct; ++d)
//         {
//             const int spatial_dim = dims[d];
//             if (spatial_dim > vlmax)
//             {
//                 printf("%s dim=%d: skipped (> VLMAX)\n", fmt_names[f], spatial_dim);
//                 continue;
//             }

//             FillBoxes(NUM_BATCHES, spatial_dim, fmts[f]);
//             FillScores(NUM_BATCHES * NUM_CLASSES * spatial_dim);

//             const int rows = NUM_BATCHES * NUM_CLASSES *
//                              (MAX_OUT < spatial_dim ? (int)MAX_OUT : spatial_dim);

//             // ---- warm-up (not timed): fills caches and branch predictors ----
//             FillOut(g_out_ref, rows);
//             FillOut(g_out_vec, rows);
//             ref::NonMaxSuppression(g_boxes, g_scores, NUM_BATCHES, NUM_CLASSES, spatial_dim,
//                                    MAX_OUT, iou_thr, &score_thr, fmts[f], g_out_ref);
//             vec::NonMaxSuppression(g_boxes, g_scores, NUM_BATCHES, NUM_CLASSES, spatial_dim,
//                                    MAX_OUT, iou_thr, &score_thr, fmts[f], g_out_vec);

//             uint64_t reference_cycles = 0, vectorized_cycles = 0;
//             uint64_t reference_instrs = 0, vectorized_instrs = 0;
//             int ref_count = 0, vec_count = 0;

//             for (int i = 0; i < loop_count; ++i)
//             {
//                 FillOut(g_out_ref, rows);
//                 FillOut(g_out_vec, rows);

//                 Timer timer_reference, timer_vectorized;

//                 timer_reference.Start();
//                 ref_count = ref::NonMaxSuppression(g_boxes, g_scores,
//                                                    NUM_BATCHES, NUM_CLASSES, spatial_dim,
//                                                    MAX_OUT, iou_thr, &score_thr,
//                                                    fmts[f], g_out_ref);
//                 timer_reference.Stop();

//                 timer_vectorized.Start();
//                 vec_count = vec::NonMaxSuppression(g_boxes, g_scores,
//                                                    NUM_BATCHES, NUM_CLASSES, spatial_dim,
//                                                    MAX_OUT, iou_thr, &score_thr,
//                                                    fmts[f], g_out_vec);
//                 timer_vectorized.Stop();

//                 reference_cycles  += timer_reference.ElapsedCycles();
//                 reference_instrs  += timer_reference.ElapsedInstructions();
//                 vectorized_cycles += timer_vectorized.ElapsedCycles();
//                 vectorized_instrs += timer_vectorized.ElapsedInstructions();

//                 if (ref_count != vec_count || !SameOut(g_out_ref, g_out_vec, rows))
//                 {
//                     printf("MISMATCH (%s, dim=%d): ref_count=%d vec_count=%d\n",
//                            fmt_names[f], spatial_dim, ref_count, vec_count);
//                     for (int r = 0; r < ref_count && r < 8; ++r)
//                         printf("  row %d: ref=[%lld,%lld,%lld] vec=[%lld,%lld,%lld]\n", r,
//                                (long long)g_out_ref[r * 3], (long long)g_out_ref[r * 3 + 1], (long long)g_out_ref[r * 3 + 2],
//                                (long long)g_out_vec[r * 3], (long long)g_out_vec[r * 3 + 1], (long long)g_out_vec[r * 3 + 2]);
//                     all_correct = false;
//                     break;
//                 }
//             }

//             if (!all_correct) break;

//             char name[48];
//             snprintf(name, sizeof(name), "%s dim=%d sel=%d", fmt_names[f], spatial_dim, ref_count);

//             PrintTime(name,
//                       reference_cycles / loop_count, vectorized_cycles / loop_count,
//                       reference_instrs / loop_count, vectorized_instrs / loop_count);
//         }
//     }

//     if (all_correct)
//     {
//         printf("Output is correct.\n");
//         return 0;
//     }
//     else
//     {
//         printf("Output is wrong!\n");
//         return 1;
//     }
// }

#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include "NMS.hpp"
#include <cstdint>

// ---------------- limits (static storage, no heap) ----------------
static const int NUM_BATCHES = 2;
static const int NUM_CLASSES = 3;
static const int MAX_DIM     = 128;
static const int64_t MAX_OUT = 10;
static const int MAX_ROWS    = NUM_BATCHES * NUM_CLASSES * (MAX_OUT < MAX_DIM ? (int)MAX_OUT : MAX_DIM);

static float   g_boxes[NUM_BATCHES * MAX_DIM * 4];
static float   g_scores[NUM_BATCHES * NUM_CLASSES * MAX_DIM];
static int64_t g_out_ref[MAX_ROWS * 3];
static int64_t g_out_vec[MAX_ROWS * 3];

// ---------------- deterministic generator (xorshift32) ----------------
static uint32_t g_state = 12345u;
static inline uint32_t Rand32()
{
    g_state ^= g_state << 13;
    g_state ^= g_state >> 17;
    g_state ^= g_state << 5;
    return g_state;
}
static inline float RandFloat(float lo, float hi)
{
    return lo + (hi - lo) * ((Rand32() & 0xFFFFFF) / 16777216.0f);
}

// Boxes come in clusters so many overlap and suppression really happens.
// Corners = [y1,x1,y2,x2], Center = [xc,yc,w,h].
static void FillBoxes(int num_batches, int spatial_dim, CenterPointBox fmt)
{
    const int num_clusters = 4;
    for (int b = 0; b < num_batches; ++b)
    {
        for (int i = 0; i < spatial_dim; ++i)
        {
            int k    = Rand32() % num_clusters;
            float cx = 20.0f + 25.0f * k + RandFloat(-3.0f, 3.0f);
            float cy = 20.0f + 25.0f * k + RandFloat(-3.0f, 3.0f);
            float w  = RandFloat(15.0f, 25.0f);
            float h  = RandFloat(15.0f, 25.0f);

            float* p = &g_boxes[(b * spatial_dim + i) * 4];
            if (fmt == CenterPointBox::Corners)
            {
                p[0] = cy - h / 2; p[1] = cx - w / 2;
                p[2] = cy + h / 2; p[3] = cx + w / 2;
            }
            else
            {
                p[0] = cx; p[1] = cy; p[2] = w; p[3] = h;
            }
        }
    }
}

// Scores quantized to 1/64 steps so exact ties occur (checks tie-breaking matches).
static void FillScores(int count)
{
    for (int i = 0; i < count; ++i)
        g_scores[i] = (Rand32() % 64) / 64.0f;
}

static void FillOut(int64_t* out, int rows)
{
    for (int i = 0; i < rows * 3; ++i) out[i] = -1;
}

static bool SameOut(const int64_t* a, const int64_t* b, int rows)
{
    for (int i = 0; i < rows * 3; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    const int   loop_count = 5;
    const float iou_thr    = 0.5f;
    const float score_thr  = 0.3f;

    // vec::NonMaxSuppression is strip-mined, so any spatial_dim works.
    const int dims[] = {8, 16, 32, 64, 80, 128};

    // Contract: boxes are always Corners.
    const CenterPointBox fmts[]      = {CenterPointBox::Corners};
    const char*          fmt_names[] = {"CORNERS"};
    const int            num_fmts    = 1;

    printf("NMS benchmark: %d batches, %d classes, %d iterations (+1 warm-up)\n",
           NUM_BATCHES, NUM_CLASSES, loop_count);

    bool all_correct = true;

    for (int f = 0; f < num_fmts && all_correct; ++f)
    {
        for (int d = 0; d < (int)(sizeof(dims) / sizeof(dims[0])) && all_correct; ++d)
        {
            const int spatial_dim = dims[d];

            FillBoxes(NUM_BATCHES, spatial_dim, fmts[f]);
            FillScores(NUM_BATCHES * NUM_CLASSES * spatial_dim);

            const int rows = NUM_BATCHES * NUM_CLASSES *
                             (MAX_OUT < spatial_dim ? (int)MAX_OUT : spatial_dim);

            // ---- warm-up (not timed): fills caches and branch predictors ----
            FillOut(g_out_ref, rows);
            FillOut(g_out_vec, rows);
            ref::NonMaxSuppression(g_boxes, g_scores, NUM_BATCHES, NUM_CLASSES, spatial_dim,
                                   MAX_OUT, iou_thr, &score_thr, fmts[f], g_out_ref);
            vec::NonMaxSuppression(g_boxes, g_scores, NUM_BATCHES, NUM_CLASSES, spatial_dim,
                                   MAX_OUT, iou_thr, &score_thr, fmts[f], g_out_vec);

            uint64_t reference_cycles = 0, vectorized_cycles = 0;
            uint64_t reference_instrs = 0, vectorized_instrs = 0;
            int ref_count = 0, vec_count = 0;

            for (int i = 0; i < loop_count; ++i)
            {
                FillOut(g_out_ref, rows);
                FillOut(g_out_vec, rows);

                Timer timer_reference, timer_vectorized;

                timer_reference.Start();
                ref_count = ref::NonMaxSuppression(g_boxes, g_scores,
                                                   NUM_BATCHES, NUM_CLASSES, spatial_dim,
                                                   MAX_OUT, iou_thr, &score_thr,
                                                   fmts[f], g_out_ref);
                timer_reference.Stop();

                timer_vectorized.Start();
                vec_count = vec::NonMaxSuppression(g_boxes, g_scores,
                                                   NUM_BATCHES, NUM_CLASSES, spatial_dim,
                                                   MAX_OUT, iou_thr, &score_thr,
                                                   fmts[f], g_out_vec);
                timer_vectorized.Stop();

                reference_cycles  += timer_reference.ElapsedCycles();
                reference_instrs  += timer_reference.ElapsedInstructions();
                vectorized_cycles += timer_vectorized.ElapsedCycles();
                vectorized_instrs += timer_vectorized.ElapsedInstructions();

                if (ref_count != vec_count || !SameOut(g_out_ref, g_out_vec, rows))
                {
                    printf("MISMATCH (%s, dim=%d): ref_count=%d vec_count=%d\n",
                           fmt_names[f], spatial_dim, ref_count, vec_count);
                    for (int r = 0; r < ref_count && r < 8; ++r)
                        printf("  row %d: ref=[%d,%d,%d] vec=[%d,%d,%d]\n", r,
                               (long long)g_out_ref[r * 3], (long long)g_out_ref[r * 3 + 1], (long long)g_out_ref[r * 3 + 2],
                               (long long)g_out_vec[r * 3], (long long)g_out_vec[r * 3 + 1], (long long)g_out_vec[r * 3 + 2]);
                    all_correct = false;
                    break;
                }
            }

            if (!all_correct) break;

           printf("dim=%d sel=%d\n", spatial_dim, ref_count);
           PrintTime(fmt_names[f],
          reference_cycles / loop_count, vectorized_cycles / loop_count,
          reference_instrs / loop_count, vectorized_instrs / loop_count);
        }
    }

    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    else
    {
        printf("Output is wrong!\n");
        return 1;
    }
}