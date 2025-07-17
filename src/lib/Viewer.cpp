#include "Viewer.hpp"

Viewer::Viewer():
    recStream([]{
        return rerun::RecordingStream("Visual Odometry");
    }()) {
    
    this->recStream.log_file_from_path("visual_odometry.rbl");
    this->recStream.spawn().exit_on_failure();
    
}

void Viewer::update(std::vector<cv::Point3d> trajectory, std::vector<cv::Point3d> keyframes, std::vector<cv::Point3d> mapPoints, cv::Mat rgb, cv::Mat depth, cv::Mat mask){
    std::cout << "IMPORTANT. Trajectory size is: " << trajectory.size() << std::endl;
    std::cout << "IMPORTANT. Keyframes size is: " << keyframes.size() << std::endl;

    std::vector<rerun::Position3D> traj;
    for(const cv::Point3f& pt : trajectory){
        traj.push_back(
            rerun::Position3D(pt.x, pt.y, pt.z)
        );
    }
    rerun::LineStrip3D line(traj);

    this->recStream.log(
        "/world/trajectory/points",
        rerun::LineStrips3D(line)
        .with_colors(rerun::Color(0, 0, 255, 150))
        .with_radii(0.5f)
    );

    std::vector<rerun::Position3D> kfs;
    for(const cv::Point3f& pt : keyframes){
        kfs.push_back(
            rerun::Position3D(pt.x, pt.y, pt.z)
        );
    } 

    this->recStream.log(
        "/world/trajectory/keyframes",
        rerun::Points3D(kfs)
        .with_colors(rerun::Color(255, 0, 0, 150))
        .with_radii(0.5f)
    );

    std::vector<rerun::Position3D> map;
    for(const cv::Point3f& pt : mapPoints){
        map.push_back(
            rerun::Position3D(pt.x, pt.y, pt.z)
        );
    } 

    this->recStream.log(
        "/world/trajectory/map",
        rerun::Points3D(map)
        .with_colors(rerun::Color(0, 255, 0, 150))
        .with_radii(0.05f)
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
