#include "Viewer.hpp"

Viewer::Viewer():
    recStream([]{
        return rerun::RecordingStream("Visual Odometry");
    }()) {
    
    this->recStream.log_file_from_path("visual_odometry.rbl");
    this->recStream.spawn().exit_on_failure();
    
}

void Viewer::update(std::vector<cv::Point2d> trajectory, std::vector<cv::Point2d> keyframes, cv::Mat rgb, cv::Mat depth, cv::Mat mask){
    std::cout << "IMPORTANT. Trajectory size is: " << trajectory.size() << std::endl;
    std::cout << "IMPORTANT. Keyframes size is: " << keyframes.size() << std::endl;

    std::vector<rerun::Position2D> traj;
    for(const cv::Point2f& pt : trajectory){
        traj.push_back(
            rerun::Position2D(pt.x, pt.y)
        );
    }
    rerun::LineStrip2D line(traj);

    this->recStream.log(
        "/world/trajectory/points",
        rerun::LineStrips2D(line)
        .with_colors(rerun::Color(0, 0, 255, 150))
        .with_radii(0.5f)
    );

    std::vector<rerun::Position2D> kfs;
    for(const cv::Point2f& pt : keyframes){
        kfs.push_back(
            rerun::Position2D(pt.x, pt.y)
        );
    } 

    this->recStream.log(
        "/world/trajectory/keyframes",
        rerun::Points2D(kfs)
        .with_colors(rerun::Color(255, 0, 0, 150))
        .with_radii(0.5f)
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
