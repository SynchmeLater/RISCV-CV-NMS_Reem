#include "NMS.hpp"

// Helper: Swap two candidate items
static inline void Swapy(BoxCandidate* a, BoxCandidate* b) {
    BoxCandidate temp = *a;
    *a = *b;
    *b = temp;
}

// Helper: Convert a box from either ONNX format to corner format [y1, x1, y2, x2]
// I use static to control visiability so these helpers are only for local usages
static inline void To_Corners(const float* b, CenterPointBox fmt,
                             float& y1, float& x1, float& y2, float& x2) {
    if (fmt == CenterPointBox::Corners) {
        y1 = b[0]; x1 = b[1]; y2 = b[2]; x2 = b[3];
    } else { // Center format: [x_center, y_center, width, height]
        float xc = b[0], yc = b[1], w = b[2], h = b[3];
        x1 = xc - w / 2.0f;
        x2 = xc + w / 2.0f;
        y1 = yc - h / 2.0f;
        y2 = yc + h / 2.0f;
    }
}

// Helper: Calculate IoU for two boxes (format given by fmt)
static float Calc_IoU(const float* boxA, const float* boxB, CenterPointBox fmt) {

    // Assumed convention: image coordinates, where the origin is the top-left,
    // x increases to the right and y increases downward.
    // Every box is [y1, x1, y2, x2] with (y1, x1) the top-left corner and
    // (y2, x2) the bottom-right corner, so y1 < y2 and x1 < x2.
    // Coordinates may be normalized or in pixels; only the ordering matters.

    float y1A, x1A, y2A, x2A, y1B, x1B, y2B, x2B;
    To_Corners(boxA, fmt, y1A, x1A, y2A, x2A);
    To_Corners(boxB, fmt, y1B, x1B, y2B, x2B);

    float inter_y1 = (y1A >= y1B) ? y1A : y1B;
    float inter_x1 = (x1A >= x1B) ? x1A : x1B;

    float inter_y2 = (y2A <= y2B) ? y2A : y2B;
    float inter_x2 = (x2A <= x2B) ? x2A : x2B;

    float inter_w = (inter_x2 - inter_x1 > 0.0f) ? (inter_x2 - inter_x1) : 0.0f;
    float inter_h = (inter_y2 - inter_y1 > 0.0f) ? (inter_y2 - inter_y1) : 0.0f;
    float inter_area = inter_w * inter_h;

    float area_A = (y2A - y1A) * (x2A - x1A);
    float area_B = (y2B - y1B) * (x2B - x1B);

    float union_area = area_A + area_B - inter_area;

    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

int ref::NonMaxSuppression(const float* boxes,
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

    for (int b = 0; b < num_batches; b++) { // outer loop is for the different batches/images
        const float* batch_boxes = boxes + (b * spatial_dim * 4);//batch_boxes: pointer at the first co-ordiates of the first box of the b image
        //Img0 Cls0 Box0	Img0 Cls0 Box1	Img0 Cls0 Box2	Img0 Cls1 Box0
        for (int c = 0; c < num_classes; c++) { // confusing this way but this is true as we follow ONNX layout 
            const float* class_scores = scores + (b * num_classes * spatial_dim) + (c * spatial_dim);

            // Step 1:remove low scores into a compact array 
            BoxCandidate candidates[spatial_dim];
            int candidate_count = 0;

            for (int i = 0; i < spatial_dim; i++) {
                if (score_threshold == nullptr || class_scores[i] > *score_threshold) {
                    candidates[candidate_count].score = class_scores[i];
                    candidates[candidate_count].index = i;
                    candidate_count++;
                }
            }
            // step 2: even-odd sorting in descending order following ONNX
            if (candidate_count > 0) {
                bool is_sorted = false;
                while (!is_sorted) {
                    is_sorted = true;
                    // Even iteration 
                    for (int i = 0; i < candidate_count - 1; i += 2) 
                        if (candidates[i].score < candidates[i + 1].score) {
                            Swapy(&candidates[i], &candidates[i + 1]);
                            is_sorted = false;
                        }
                    // Odd iteration
                    for (int i = 1; i < candidate_count - 1; i += 2) 
                        if (candidates[i].score < candidates[i + 1].score) {
                            Swapy(&candidates[i], &candidates[i + 1]);
                            is_sorted = false;
                        }
                }
            }
            // Step 3: Greedy Selection & IoU Suppression Loop
                uint8_t suppressed[candidate_count+1]; // I added one just in case candidate_count is 0 
                for (int i = 0; i < candidate_count; ++i) {
                    suppressed[i] = 0; // zero means still alive 
                }
                int64_t class_selected_count = 0; // this counts how many boxes are kept for this class so we compare to max_output_boxes_per_class

                //It goes from the best score to the worst, and stops early once you’ve kept enough boxes. That’s like a replacement for break.
                for (int i = 0; i < candidate_count && class_selected_count < max_output_boxes_per_class; i++) {
                    if (!suppressed[i]) {
                        int box_idx = candidates[i].index;

                        selected_indices[total_selected * 3 + 0] = b; // image index
                        selected_indices[total_selected * 3 + 1] = c; // class index
                        selected_indices[total_selected * 3 + 2] = box_idx; // box index

                        total_selected++;
                        class_selected_count++;

                        const float* boxA = batch_boxes + (box_idx * 4);

                        for (int j = i + 1; j < candidate_count; ++j) {
                            if (!suppressed[j]) {
                                const float* boxB = batch_boxes + (candidates[j].index * 4);
                                if (Calc_IoU(boxA, boxB, center_point_box) > iou_threshold)
                                    suppressed[j] = 1; // kill
                               
                        }
                    }
                }
            }
        }
    }

    return total_selected;
}