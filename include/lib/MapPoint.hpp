#pragma once

#include <opencv2/core.hpp>
#include <map>

class KeyFrame;

class MapPoint {
public:
    MapPoint(Eigen::Vector3d position) : 
    ptPosition(position) {}

    void AddObservation(std::shared_ptr<KeyFrame> keyframe, int keypointIdx) {
        this->mapObservations[keyframe] = keypointIdx;
    }

    void RemoveObservation(std::shared_ptr<KeyFrame> keyframe) {
        this->mapObservations.erase(keyframe);
    }

    void SetPosition(Eigen::Vector3d pt) {
        this->ptPosition = pt;
    }

    Eigen::Vector3d GetPosition() const {
        return this->ptPosition;
    }

    bool IsBad() const {
        return this->mapObservations.size() < 2;
    }

    std::map<std::weak_ptr<KeyFrame>, int, std::owner_less<std::weak_ptr<KeyFrame>>> GetObservations() {
        return this->mapObservations;
    }
private:
    Eigen::Vector3d ptPosition;
    std::map<std::weak_ptr<KeyFrame>, int, std::owner_less<std::weak_ptr<KeyFrame>>> mapObservations;

};