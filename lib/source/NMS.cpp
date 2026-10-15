
// #include "NMS.hpp"
// #include <riscv_vector.h>
// #include <cstddef>
// #include <cstdint>

// // One odd-even phase over MEMORY, one strip at a time.
// // Pairs are (phase, phase+1), (phase+2, phase+3), ...
// // Strided loads deal the pairs into a (first of pair) and b (second of pair),
// // so a pair may straddle two strips with no problem.
// // Returns true if any pair was swapped.
// static inline bool OddEvenPhase(float* sc, uint32_t* ix, size_t n, size_t phase)
// {
//     if (n < phase + 2) return false;

//     const size_t    pairs  = (n - phase) / 2;
//     const ptrdiff_t stride = 2 * sizeof(float);   // every other element
//     bool swapped = false;

//     size_t k = 0;
//     while (k < pairs) {
//         size_t vl   = __riscv_vsetvl_e32m4(pairs - k);
//         size_t base = phase + 2 * k;

//         vfloat32m4_t a  = __riscv_vlse32_v_f32m4(sc + base,     stride, vl);
//         vfloat32m4_t b  = __riscv_vlse32_v_f32m4(sc + base + 1, stride, vl);
//         vuint32m4_t  ai = __riscv_vlse32_v_u32m4(ix + base,     stride, vl);
//         vuint32m4_t  bi = __riscv_vlse32_v_u32m4(ix + base + 1, stride, vl);

//         // descending order; strict < keeps ties stable
//         vbool8_t swap = __riscv_vmflt_vv_f32m4_b8(a, b, vl);

//         // swap scores and indices with the SAME mask
//         __riscv_vsse32_v_f32m4(sc + base,     stride, __riscv_vmerge_vvm_f32m4(a, b, swap, vl), vl);
//         __riscv_vsse32_v_f32m4(sc + base + 1, stride, __riscv_vmerge_vvm_f32m4(b, a, swap, vl), vl);
//         __riscv_vsse32_v_u32m4(ix + base,     stride, __riscv_vmerge_vvm_u32m4(ai, bi, swap, vl), vl);
//         __riscv_vsse32_v_u32m4(ix + base + 1, stride, __riscv_vmerge_vvm_u32m4(bi, ai, swap, vl), vl);

//         if (__riscv_vcpop_m_b8(swap, vl) > 0) swapped = true;
//         k += vl;
//     }
//     return swapped;
// }

// // Convert 4 gathered coordinate vectors (r0..r3) to corner format (y1, x1, y2, x2).
// //   Corners: r = [y1, x1, y2, x2] in any diagonal order -> normalize with min/max
// //   Center:  r = [x_center, y_center, width, height]
// static inline void ToCornersVec(vfloat32m4_t r0, vfloat32m4_t r1,
//                                 vfloat32m4_t r2, vfloat32m4_t r3,
//                                 CenterPointBox fmt, size_t n,
//                                 vfloat32m4_t& y1v, vfloat32m4_t& x1v,
//                                 vfloat32m4_t& y2v, vfloat32m4_t& x2v)
// {
//     if (fmt == CenterPointBox::Corners) {
//         // any diagonal order is legal in ONNX, so normalize with min/max
//         y1v = __riscv_vfmin_vv_f32m4(r0, r2, n);
//         y2v = __riscv_vfmax_vv_f32m4(r0, r2, n);
//         x1v = __riscv_vfmin_vv_f32m4(r1, r3, n);
//         x2v = __riscv_vfmax_vv_f32m4(r1, r3, n);
//     } else {
//         // Center: r0 = xc, r1 = yc, r2 = w, r3 = h
//         vfloat32m4_t hw = __riscv_vfmul_vf_f32m4(r2, 0.5f, n);
//         vfloat32m4_t hh = __riscv_vfmul_vf_f32m4(r3, 0.5f, n);
//         x1v = __riscv_vfsub_vv_f32m4(r0, hw, n);
//         x2v = __riscv_vfadd_vv_f32m4(r0, hw, n);
//         y1v = __riscv_vfsub_vv_f32m4(r1, hh, n);
//         y2v = __riscv_vfadd_vv_f32m4(r1, hh, n);
//     }
// }

// int vec::NonMaxSuppression(const float* boxes,
//                            const float* scores,
//                            int num_batches,
//                            int num_classes,
//                            int spatial_dim,
//                            int64_t max_output_boxes_per_class,
//                            float iou_threshold,
//                            const float* score_threshold,
//                            CenterPointBox center_point_box,
//                            int64_t* selected_indices)
// {
//     int total_selected = 0;

//     for (int b = 0; b < num_batches; b++) {
//         const float* batch_boxes = boxes + (b * spatial_dim * 4);
//         for (int c = 0; c < num_classes; c++) {
//             const float* class_scores = scores + (b * num_classes * spatial_dim) + (c * spatial_dim);

//             // candidates live in memory now, so any number of boxes fits (same idea as scalar)
//             float    cand_score[spatial_dim];
//             uint32_t cand_index[spatial_dim];
//             uint8_t  suppressed[spatial_dim];

//     // step 1: compare scalar to vector and mask the killed ones, then compact -- STRIP-MINED
//             size_t candidate_count = 0;
//             size_t offset = 0;
//             while (offset < (size_t)spatial_dim) {
//                 size_t vl = __riscv_vsetvl_e32m4(spatial_dim - offset);

//                 vfloat32m4_t scores_vec = __riscv_vle32_v_f32m4(class_scores + offset, vl);
//                 // box index = lane position + strip offset
//                 vuint32m4_t  idex       = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), (uint32_t)offset, vl);

//                 vbool8_t keep = (score_threshold == nullptr) ?
//                     __riscv_vmset_m_b8(vl) : __riscv_vmfgt_vf_f32m4_b8(scores_vec, *score_threshold, vl);

//                 size_t kept = __riscv_vcpop_m_b8(keep, vl);

//                 vfloat32m4_t scores_1 = __riscv_vcompress_vm_f32m4(scores_vec, keep, vl);  // surviving scores
//                 vuint32m4_t  index_1  = __riscv_vcompress_vm_u32m4(idex, keep, vl);        // their box indices

//                 // append this strip's survivors after the previous strips' survivors
//                 __riscv_vse32_v_f32m4(cand_score + candidate_count, scores_1, kept);
//                 __riscv_vse32_v_u32m4(cand_index + candidate_count, index_1, kept);

//                 candidate_count += kept;
//                 offset += vl;
//             }

//     // step 2: all even comparisons at once THEN all odd comparisons + iterate
//             size_t n = candidate_count;

//             bool is_sorted = (n < 2);
//             while (!is_sorted) {
//                 bool swapped = false;
//                 swapped |= OddEvenPhase(cand_score, cand_index, n, 0);  // EVEN: (0,1), (2,3), ...
//                 swapped |= OddEvenPhase(cand_score, cand_index, n, 1);  // ODD:  (1,2), (3,4), ...
//                 is_sorted = !swapped;   // done when a full even AND odd phase swapped nothing
//             }

//     // step 3: calc IoU and compare all at once THEN iterate to next selected element
//             if (n > 0) {
//                 for (size_t k = 0; k < n; k++) suppressed[k] = 0;   // 0 = alive

//                 int64_t class_selected_count = 0;
//                 size_t i = 0;   // best candidate is always alive at the start

//                 while (i < n && class_selected_count < max_output_boxes_per_class) {
//                     // keep box i
//                     selected_indices[total_selected * 3 + 0] = b;                  // image index
//                     selected_indices[total_selected * 3 + 1] = c;                  // class index
//                     selected_indices[total_selected * 3 + 2] = cand_index[i];      // box index
//                     total_selected++;
//                     class_selected_count++;

//                     // box A (the one just kept) as scalars: 1-lane gather + ToCornersVec
//                     const float* pA = batch_boxes + (size_t)cand_index[i] * 4;
//                     vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(pA[0], 1);
//                     vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(pA[1], 1);
//                     vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(pA[2], 1);
//                     vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(pA[3], 1);
//                     vfloat32m4_t ay1, ax1, ay2, ax2;
//                     ToCornersVec(a0, a1, a2, a3, center_point_box, 1, ay1, ax1, ay2, ax2);
//                     float y1A   = __riscv_vfmv_f_s_f32m4_f32(ay1);
//                     float x1A   = __riscv_vfmv_f_s_f32m4_f32(ax1);
//                     float y2A   = __riscv_vfmv_f_s_f32m4_f32(ay2);
//                     float x2A   = __riscv_vfmv_f_s_f32m4_f32(ax2);
//                     float areaA = (y2A - y1A) * (x2A - x1A);

//                     // box A against ALL later candidates, one strip at a time
//                     size_t j = i + 1;
//                     while (j < n) {
//                         size_t vl = __riscv_vsetvl_e32m4(n - j);

//                         // gather this strip's boxes (byte offset = box_idx * 16 + k * 4)
//                         vuint32m4_t  off = __riscv_vsll_vx_u32m4(__riscv_vle32_v_u32m4(cand_index + j, vl), 4, vl);
//                         vfloat32m4_t r0  = __riscv_vluxei32_v_f32m4(batch_boxes, off, vl);
//                         vfloat32m4_t r1  = __riscv_vluxei32_v_f32m4(batch_boxes, __riscv_vadd_vx_u32m4(off, 4,  vl), vl);
//                         vfloat32m4_t r2  = __riscv_vluxei32_v_f32m4(batch_boxes, __riscv_vadd_vx_u32m4(off, 8,  vl), vl);
//                         vfloat32m4_t r3  = __riscv_vluxei32_v_f32m4(batch_boxes, __riscv_vadd_vx_u32m4(off, 12, vl), vl);

//                         vfloat32m4_t y1v, x1v, y2v, x2v;
//                         ToCornersVec(r0, r1, r2, r3, center_point_box, vl, y1v, x1v, y2v, x2v);

//                         vfloat32m4_t area_v = __riscv_vfmul_vv_f32m4(__riscv_vfsub_vv_f32m4(y2v, y1v, vl), __riscv_vfsub_vv_f32m4(x2v, x1v, vl), vl);

//                         // intersection of box A against this strip
//                         vfloat32m4_t iy1 = __riscv_vfmax_vf_f32m4(y1v, y1A, vl);
//                         vfloat32m4_t ix1 = __riscv_vfmax_vf_f32m4(x1v, x1A, vl);
//                         vfloat32m4_t iy2 = __riscv_vfmin_vf_f32m4(y2v, y2A, vl);
//                         vfloat32m4_t ix2 = __riscv_vfmin_vf_f32m4(x2v, x2A, vl);

//                         vfloat32m4_t inter_h = __riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(iy2, iy1, vl), 0.0f, vl);
//                         vfloat32m4_t inter_w = __riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(ix2, ix1, vl), 0.0f, vl);
//                         vfloat32m4_t inter   = __riscv_vfmul_vv_f32m4(inter_w, inter_h, vl);

//                         vfloat32m4_t uni = __riscv_vfsub_vv_f32m4(
//                             __riscv_vfadd_vf_f32m4(area_v, areaA, vl), inter, vl);

//                         // iou = inter / uni, but 0 when uni <= 0 (same as scalar Calc_IoU)
//                         vbool8_t     uni_pos = __riscv_vmfgt_vf_f32m4_b8(uni, 0.0f, vl);
//                         vfloat32m4_t iou     = __riscv_vfdiv_vv_f32m4(inter, uni, vl);
//                         vbool8_t     kill    = __riscv_vmand_mm_b8(
//                             uni_pos, __riscv_vmfgt_vf_f32m4_b8(iou, iou_threshold, vl), vl);

//                         // suppressed[j + lane] = 1 only where kill is set (e8m1 uses the same vbool8_t mask)
//                         __riscv_vse8_v_u8m1_m(kill, suppressed + j, __riscv_vmv_v_x_u8m1(1, vl), vl);

//                         j += vl;
//                     }

//                     // next survivor
//                     i++;
//                     while (i < n && suppressed[i]) i++;
//                 }
//             }
//         }
//     }

//     return total_selected;
// }

#include "NMS.hpp"
#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

// Convert 4 gathered coordinate vectors (r0..r3) to corner format (y1, x1, y2, x2).
//   Corners: r = [y1, x1, y2, x2] in any diagonal order -> normalize with min/max
//   Center:  r = [x_center, y_center, width, height]
static inline void ToCornersVec(vfloat32m4_t r0, vfloat32m4_t r1,
                                vfloat32m4_t r2, vfloat32m4_t r3,
                                CenterPointBox fmt, size_t n,
                                vfloat32m4_t& y1v, vfloat32m4_t& x1v,
                                vfloat32m4_t& y2v, vfloat32m4_t& x2v)
{
    if (fmt == CenterPointBox::Corners) {
        // any diagonal order is legal in ONNX, so normalize with min/max
        y1v = __riscv_vfmin_vv_f32m4(r0, r2, n);
        y2v = __riscv_vfmax_vv_f32m4(r0, r2, n);
        x1v = __riscv_vfmin_vv_f32m4(r1, r3, n);
        x2v = __riscv_vfmax_vv_f32m4(r1, r3, n);
    } else {
        // Center: r0 = xc, r1 = yc, r2 = w, r3 = h
        vfloat32m4_t hw = __riscv_vfmul_vf_f32m4(r2, 0.5f, n);
        vfloat32m4_t hh = __riscv_vfmul_vf_f32m4(r3, 0.5f, n);
        x1v = __riscv_vfsub_vv_f32m4(r0, hw, n);
        x2v = __riscv_vfadd_vv_f32m4(r0, hw, n);
        y1v = __riscv_vfsub_vv_f32m4(r1, hh, n);
        y2v = __riscv_vfadd_vv_f32m4(r1, hh, n);
    }
}

int vec::NonMaxSuppression(const float* boxes,
                           const float* scores,
                           int num_batches,
                           int num_classes,
                           int spatial_dim,
                           int64_t max_output_boxes_per_class,
                           float iou_threshold,
                           const float* score_threshold,
                           CenterPointBox center_point_box,
                           int64_t* selected_indices)
{
    int total_selected = 0;

    for (int b = 0; b < num_batches; b++) {
        const float* batch_boxes = boxes + (b * spatial_dim * 4);
        for (int c = 0; c < num_classes; c++) {
            const float* class_scores = scores + (b * num_classes * spatial_dim) + (c * spatial_dim);

            // candidates live in memory now, so any number of boxes fits (same idea as scalar)
            float    cand_score[spatial_dim];
            uint32_t cand_index[spatial_dim];
            uint8_t  suppressed[spatial_dim];

    // step 1: compare scalar to vector and mask the killed ones, then compact -- STRIP-MINED
            size_t candidate_count = 0;
            size_t offset = 0;
            while (offset < (size_t)spatial_dim) {
                size_t vl = __riscv_vsetvl_e32m4(spatial_dim - offset);

                vfloat32m4_t scores_vec = __riscv_vle32_v_f32m4(class_scores + offset, vl);
                // box index = lane position + strip offset
                vuint32m4_t  idex       = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), (uint32_t)offset, vl);

                vbool8_t keep = (score_threshold == nullptr) ?
                    __riscv_vmset_m_b8(vl) : __riscv_vmfgt_vf_f32m4_b8(scores_vec, *score_threshold, vl);

                size_t kept = __riscv_vcpop_m_b8(keep, vl);

                vfloat32m4_t scores_1 = __riscv_vcompress_vm_f32m4(scores_vec, keep, vl);  // surviving scores
                vuint32m4_t  index_1  = __riscv_vcompress_vm_u32m4(idex, keep, vl);        // their box indices

                // append this strip's survivors after the previous strips' survivors
                __riscv_vse32_v_f32m4(cand_score + candidate_count, scores_1, kept);
                __riscv_vse32_v_u32m4(cand_index + candidate_count, index_1, kept);

                candidate_count += kept;
                offset += vl;
            }

    // step 2: odd-even sort in SPLIT layout (E = even positions, O = odd positions)
            size_t n = candidate_count;

            float    es[spatial_dim], os[spatial_dim];
            uint32_t ei[spatial_dim], oi[spatial_dim];
            const size_t    nE     = (n + 1) / 2;          // even positions 0, 2, 4, ...
            const size_t    nO     = n / 2;                // odd positions  1, 3, 5, ...
            const ptrdiff_t stride = 2 * sizeof(float);    // every other element

            // split ONCE (strided loads happen only here)
            for (size_t k = 0; k < nE; ) {
                size_t vl = __riscv_vsetvl_e32m4(nE - k);
                __riscv_vse32_v_f32m4(es + k, __riscv_vlse32_v_f32m4(cand_score + 2 * k, stride, vl), vl);
                __riscv_vse32_v_u32m4(ei + k, __riscv_vlse32_v_u32m4(cand_index + 2 * k, stride, vl), vl);
                k += vl;
            }
            for (size_t k = 0; k < nO; ) {
                size_t vl = __riscv_vsetvl_e32m4(nO - k);
                __riscv_vse32_v_f32m4(os + k, __riscv_vlse32_v_f32m4(cand_score + 2 * k + 1, stride, vl), vl);
                __riscv_vse32_v_u32m4(oi + k, __riscv_vlse32_v_u32m4(cand_index + 2 * k + 1, stride, vl), vl);
                k += vl;
            }

            bool is_sorted = (n < 2);
            while (!is_sorted) {
                bool swapped = false;

                // EVEN phase: pairs (E[k], O[k]), nO pairs
                for (size_t k = 0; k < nO; ) {
                    size_t vl = __riscv_vsetvl_e32m4(nO - k);
                    vfloat32m4_t a  = __riscv_vle32_v_f32m4(es + k, vl);
                    vfloat32m4_t bb = __riscv_vle32_v_f32m4(os + k, vl);
                    vuint32m4_t  ai = __riscv_vle32_v_u32m4(ei + k, vl);
                    vuint32m4_t  bi = __riscv_vle32_v_u32m4(oi + k, vl);

                    // descending order; strict < keeps ties stable
                    vbool8_t swap = __riscv_vmflt_vv_f32m4_b8(a, bb, vl);

                    // swap scores and indices with the SAME mask
                    __riscv_vse32_v_f32m4(es + k, __riscv_vmerge_vvm_f32m4(a,  bb, swap, vl), vl);
                    __riscv_vse32_v_f32m4(os + k, __riscv_vmerge_vvm_f32m4(bb, a,  swap, vl), vl);
                    __riscv_vse32_v_u32m4(ei + k, __riscv_vmerge_vvm_u32m4(ai, bi, swap, vl), vl);
                    __riscv_vse32_v_u32m4(oi + k, __riscv_vmerge_vvm_u32m4(bi, ai, swap, vl), vl);

                    if (__riscv_vcpop_m_b8(swap, vl) > 0) swapped = true;
                    k += vl;
                }

                // ODD phase: pairs (O[k], E[k+1]) -- the "shift by one" is just es + k + 1
                const size_t np = (n - 1) / 2;
                for (size_t k = 0; k < np; ) {
                    size_t vl = __riscv_vsetvl_e32m4(np - k);
                    vfloat32m4_t a  = __riscv_vle32_v_f32m4(os + k, vl);
                    vfloat32m4_t bb = __riscv_vle32_v_f32m4(es + k + 1, vl);
                    vuint32m4_t  ai = __riscv_vle32_v_u32m4(oi + k, vl);
                    vuint32m4_t  bi = __riscv_vle32_v_u32m4(ei + k + 1, vl);

                    vbool8_t swap = __riscv_vmflt_vv_f32m4_b8(a, bb, vl);

                    __riscv_vse32_v_f32m4(os + k,     __riscv_vmerge_vvm_f32m4(a,  bb, swap, vl), vl);
                    __riscv_vse32_v_f32m4(es + k + 1, __riscv_vmerge_vvm_f32m4(bb, a,  swap, vl), vl);
                    __riscv_vse32_v_u32m4(oi + k,     __riscv_vmerge_vvm_u32m4(ai, bi, swap, vl), vl);
                    __riscv_vse32_v_u32m4(ei + k + 1, __riscv_vmerge_vvm_u32m4(bi, ai, swap, vl), vl);

                    if (__riscv_vcpop_m_b8(swap, vl) > 0) swapped = true;
                    k += vl;
                }

                is_sorted = !swapped;   // done when a full even AND odd phase swapped nothing
            }

            // interleave back ONCE, so step 3 sees the sorted order in cand_score / cand_index
            for (size_t k = 0; k < nE; ) {
                size_t vl = __riscv_vsetvl_e32m4(nE - k);
                __riscv_vsse32_v_f32m4(cand_score + 2 * k, stride, __riscv_vle32_v_f32m4(es + k, vl), vl);
                __riscv_vsse32_v_u32m4(cand_index + 2 * k, stride, __riscv_vle32_v_u32m4(ei + k, vl), vl);
                k += vl;
            }
            for (size_t k = 0; k < nO; ) {
                size_t vl = __riscv_vsetvl_e32m4(nO - k);
                __riscv_vsse32_v_f32m4(cand_score + 2 * k + 1, stride, __riscv_vle32_v_f32m4(os + k, vl), vl);
                __riscv_vsse32_v_u32m4(cand_index + 2 * k + 1, stride, __riscv_vle32_v_u32m4(oi + k, vl), vl);
                k += vl;
            }

    // step 3: calc IoU and compare all at once THEN iterate to next selected element
            if (n > 0) {
                for (size_t k = 0; k < n; k++) suppressed[k] = 0;   // 0 = alive

                int64_t class_selected_count = 0;
                size_t i = 0;   // best candidate is always alive at the start

                while (i < n && class_selected_count < max_output_boxes_per_class) {
                    // keep box i
                    selected_indices[total_selected * 3 + 0] = b;                  // image index
                    selected_indices[total_selected * 3 + 1] = c;                  // class index
                    selected_indices[total_selected * 3 + 2] = cand_index[i];      // box index
                    total_selected++;
                    class_selected_count++;

                    // box A (the one just kept) as scalars: 1-lane gather + ToCornersVec
                    const float* pA = batch_boxes + (size_t)cand_index[i] * 4;
                    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(pA[0], 1);
                    vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(pA[1], 1);
                    vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(pA[2], 1);
                    vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(pA[3], 1);
                    vfloat32m4_t ay1, ax1, ay2, ax2;
                    ToCornersVec(a0, a1, a2, a3, center_point_box, 1, ay1, ax1, ay2, ax2);
                    float y1A   = __riscv_vfmv_f_s_f32m4_f32(ay1);
                    float x1A   = __riscv_vfmv_f_s_f32m4_f32(ax1);
                    float y2A   = __riscv_vfmv_f_s_f32m4_f32(ay2);
                    float x2A   = __riscv_vfmv_f_s_f32m4_f32(ax2);
                    float areaA = (y2A - y1A) * (x2A - x1A);

                    // box A against ALL later candidates, one strip at a time
                    size_t j = i + 1;
                    while (j < n) {
                        size_t vl = __riscv_vsetvl_e32m4(n - j);

                        // gather this strip's boxes (byte offset = box_idx * 16 + k * 4)
                        vuint32m4_t  off = __riscv_vsll_vx_u32m4(__riscv_vle32_v_u32m4(cand_index + j, vl), 4, vl);
                        vfloat32m4_t r0  = __riscv_vluxei32_v_f32m4(batch_boxes, off, vl);
                        vfloat32m4_t r1  = __riscv_vluxei32_v_f32m4(batch_boxes, __riscv_vadd_vx_u32m4(off, 4,  vl), vl);
                        vfloat32m4_t r2  = __riscv_vluxei32_v_f32m4(batch_boxes, __riscv_vadd_vx_u32m4(off, 8,  vl), vl);
                        vfloat32m4_t r3  = __riscv_vluxei32_v_f32m4(batch_boxes, __riscv_vadd_vx_u32m4(off, 12, vl), vl);

                        vfloat32m4_t y1v, x1v, y2v, x2v;
                        ToCornersVec(r0, r1, r2, r3, center_point_box, vl, y1v, x1v, y2v, x2v);

                        vfloat32m4_t area_v = __riscv_vfmul_vv_f32m4(__riscv_vfsub_vv_f32m4(y2v, y1v, vl), __riscv_vfsub_vv_f32m4(x2v, x1v, vl), vl);

                        // intersection of box A against this strip
                        vfloat32m4_t iy1 = __riscv_vfmax_vf_f32m4(y1v, y1A, vl);
                        vfloat32m4_t ix1 = __riscv_vfmax_vf_f32m4(x1v, x1A, vl);
                        vfloat32m4_t iy2 = __riscv_vfmin_vf_f32m4(y2v, y2A, vl);
                        vfloat32m4_t ix2 = __riscv_vfmin_vf_f32m4(x2v, x2A, vl);

                        vfloat32m4_t inter_h = __riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(iy2, iy1, vl), 0.0f, vl);
                        vfloat32m4_t inter_w = __riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(ix2, ix1, vl), 0.0f, vl);
                        vfloat32m4_t inter   = __riscv_vfmul_vv_f32m4(inter_w, inter_h, vl);

                        vfloat32m4_t uni = __riscv_vfsub_vv_f32m4(
                            __riscv_vfadd_vf_f32m4(area_v, areaA, vl), inter, vl);

                        // iou = inter / uni, but 0 when uni <= 0 (same as scalar Calc_IoU)
                        vbool8_t     uni_pos = __riscv_vmfgt_vf_f32m4_b8(uni, 0.0f, vl);
                        vfloat32m4_t iou     = __riscv_vfdiv_vv_f32m4(inter, uni, vl);
                        vbool8_t     kill    = __riscv_vmand_mm_b8(
                            uni_pos, __riscv_vmfgt_vf_f32m4_b8(iou, iou_threshold, vl), vl);

                        // suppressed[j + lane] = 1 only where kill is set (e8m1 uses the same vbool8_t mask)
                        __riscv_vse8_v_u8m1_m(kill, suppressed + j, __riscv_vmv_v_x_u8m1(1, vl), vl);

                        j += vl;
                    }

                    // next survivor
                    i++;
                    while (i < n && suppressed[i]) i++;
                }
            }
        }
    }

    return total_selected;
}