#pragma once

#include "types.hpp"

#include <cstdint>

namespace ref
{
    // Scalar educational NonMaxSuppression. Same contract and algorithm as
    // vec::NonMaxSuppression, including odd-even sort (corners only, y1 <= y2
    // and x1 <= x2, no box-format conversion). See lib/include/NMS.hpp.

    int NonMaxSuppression(const float* boxes, // co-ordinates
                          const float* scores, // confidence
                          int num_batches,     // number of images
                          int num_classes,    // number of objects per image
                          int spatial_dim,    // number of boxes in the entire image
                          int64_t max_output_boxes_per_class, 
                          float iou_threshold,   
                          const float* score_threshold, // When nullptr => the score threshold was omitted by the user or model exporter. 
                          CenterPointBox center_point_box,  //??? not sure yet 
                          int64_t* selected_indices); // the output array and it has the indexes of image, class, box

                          // it returns the total number of boxes that survived the superession in all images and classes
}
