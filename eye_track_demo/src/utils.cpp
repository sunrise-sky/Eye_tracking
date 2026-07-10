#include "../include/utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape) {
    osd_device.Initialize(in_img_shape[0], in_img_shape[1]);
}

void VISUALIZER::Draw() {
    std::vector<sst::device::osd::OsdQuadRangle> quad_rangle_vec;
    sst::device::osd::OsdQuadRangle q;
    q.color = 0;
    q.box = {100, 100, 200, 200};
    q.border = 3;
    q.alpha = fdevice::TYPE_ALPHA75;
    q.type = fdevice::TYPE_HOLLOW;
    quad_rangle_vec.emplace_back(q);
    osd_device.Draw(quad_rangle_vec);
}

void VISUALIZER::Draw(const std::vector<std::array<float, 4>>& boxes) {
    std::vector<sst::device::osd::OsdQuadRangle> quad_rangle_vec;
    for (size_t i = 0; i < boxes.size(); i++) {
        sst::device::osd::OsdQuadRangle q;
        int xmin = static_cast<int>(boxes[i][0]);
        int ymin = static_cast<int>(boxes[i][1]);
        int xmax = static_cast<int>(boxes[i][2]);
        int ymax = static_cast<int>(boxes[i][3]);
        q.box = {xmin, ymin, xmax, ymax};
        q.alpha = fdevice::TYPE_ALPHA100;
        q.border = 3;
        if (ymin >= 480) {
            if ((xmax - xmin) > 200) {
                q.color = 16; q.type = fdevice::TYPE_SOLID; q.border = 0;
            } else {
                q.color = 12; q.type = fdevice::TYPE_SOLID; q.border = 0;
            }
        } else if (i == 0) {
            q.color = 2; q.type = fdevice::TYPE_HOLLOW;
        } else if (i == 1 || i == 2) {
            q.color = 2; q.type = fdevice::TYPE_HOLLOW; q.border = 2;
        } else {
            q.color = 0; q.type = fdevice::TYPE_SOLID; q.border = 0;
        }
        quad_rangle_vec.emplace_back(q);
    }
    osd_device.Draw(quad_rangle_vec);
}


void VISUALIZER::Release() {
    osd_device.Release();
}
