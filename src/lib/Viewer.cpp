#include "Viewer.hpp"

Viewer::Viewer():
    recStream([]{
        return rerun::RecordingStream("Visual Odometry");
    }()) {
    
    this->recStream.log_file_from_path("visual_odometry.rbl");
    this->recStream.spawn().exit_on_failure();
    
}

void Viewer::update(std::vector<cv::Point2d> trajectory, cv::Mat rgb, cv::Mat depth, cv::Mat mask){
    std::vector<rerun::Position2D> positions;
    for(const cv::Point2f& pt : trajectory){
        positions.push_back(
            rerun::Position2D(pt.x, pt.y)
        );
    }

    this->recStream.log(
        "/world/trajectory/points",
        rerun::Points2D(positions)
        .with_colors(rerun::Color(0, 0, 255, 150))
        .with_radii(0.2f)
    );

    this->recStream.log(
        "/camera/0/rgb",
        rerun::Image::from_rgb24(
            rerun::borrow(rgb.data, rgb.total() * rgb.elemSize()),
            {(uint32_t)rgb.cols, (uint32_t)rgb.rows}
        )
    );

    cv::convertScaleAbs(depth, depth, 0.02);
    cv::applyColorMap(depth, depth, cv::COLORMAP_JET);
    cv::cvtColor(depth, depth, cv::COLOR_BGR2RGB);

    this->recStream.log(
        "/camera/0/depth",
        rerun::Image::from_rgb24(
            rerun::borrow(depth.data, depth.total() * depth.elemSize()),
            {(uint32_t)depth.cols, (uint32_t)depth.rows}
        )
    );

    if (mask.channels() == 1) {
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    }

    this->recStream.log(
        "/camera/0/mask",
        rerun::Image::from_rgb24(
            rerun::borrow(mask.data, mask.total() * mask.elemSize()),
            {(uint32_t)mask.cols, (uint32_t)mask.rows}
        )
    );
}
