#include <plot.hpp>

Plot::Plot(){
    this->screenWidth = 800;
    this->screenHeight = 600;

    this->margin = 50.0f;
    this->padding = 50.0f;

    this->gridSpacingX = 10.0;
    this->gridSpacingY = 10.0;

    this->fps = 60;

    this->minX = std::numeric_limits<float>::infinity();
    this->maxX = - std::numeric_limits<float>::infinity();
    this->minY = std::numeric_limits<float>::infinity();
    this->maxY = -std::numeric_limits<float>::infinity();

    this->plottingAreaWidth = screenWidth - 2 * margin - 2 * padding;
    this->plottingAreaHeight = screenWidth - 2 * margin - 2 * padding;
}

// TODO: add constructor to allow the user to specify some values

float Plot::World2Pixel(float world_value, bool axis){
    if (axis){
        return screenHeight - (margin + padding + (world_value - minY) * scale + (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0.0));
    }
    return margin + padding + (world_value - minX) * scale + (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0.0);

}

float Plot::Pixel2World(float pixel_value, bool axis){
    // axis = 0 -> x axis, axis = 1 -> y axis
    if (axis){
        return (screenHeight - pixel_value - margin - padding - (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0.0))/scale + minY;
    }
    return (pixel_value - margin - padding - (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0.0))/scale + minX;
}

void Plot::add_point(cv::Point2f point){
    trajectory.push_back(point);

    this->minX = (point.x < minX) ? point.x : minX;
    this->maxX = (point.x > maxX) ? point.x : maxX;
    this->minY = (point.y < minY) ? point.y : minY;
    this->maxY = (point.y > maxY) ? point.y : maxY;

    this->scaleX = (screenWidth - 2 * margin - 2 * padding) / (maxX - minX);
    this->scaleY = (screenHeight - 2 * margin - 2 * padding) / (maxY - minY);
    this->scale = (scaleX < scaleY) ? scaleX : scaleY;
}

void Plot::draw_plot(){
    BeginDrawing();
    ClearBackground(BLACK);

    if(trajectory.size() == 0){
        const char* message = "Still waiting for points ...";
        int fontSize = 20;
        int textWidth = MeasureText(message, fontSize);
        int x = (screenWidth - textWidth) / 2;
        int y = screenHeight / 2 - fontSize / 2;

        DrawText(message, x, y, fontSize, YELLOW);
        EndDrawing();
        return;
    }

    // draw x grid
    float startX = std::ceil(Pixel2World(margin,0) / gridSpacingX) * gridSpacingX;
    float endX = Pixel2World(screenWidth - margin,0);
    for (float i = startX; i <= endX; i += gridSpacingX) {
        float linePlot = World2Pixel(i,0);
        DrawLine(linePlot, margin, linePlot, screenHeight - margin, DARKGRAY);

        char xLabel[16];
        sprintf(xLabel, "%.1fm", i);
        DrawText(xLabel, linePlot - 15, screenHeight - margin + 5, 12, DARKGRAY);
    }
    
    // draw z grid
    float startY = std::ceil(Pixel2World(screenHeight - margin,1) / gridSpacingY) * gridSpacingY;
    float endY = Pixel2World(margin,1);
    for (float i = startY; i <= endY; i += gridSpacingY) {
        float linePlot = World2Pixel(i,1);
        DrawLine(margin, linePlot, screenWidth - margin, linePlot, DARKGRAY);

        char yLabel[16];
        float yValue = Pixel2World(i,1);
        sprintf(yLabel, "%.1fm", i);
        DrawText(yLabel, margin - 45, linePlot - 5, 12, DARKGRAY);
    }

    // draw axis
    DrawLine(margin, screenHeight - margin, screenWidth - margin, screenHeight - margin, WHITE);
    DrawLine(margin, margin, margin, screenHeight - margin, WHITE); 
    DrawText("x (meters)", screenWidth - 100, screenHeight - margin + 10, 14, WHITE);
    DrawText("z (meters)", margin - 40, margin - 20, 14, WHITE);

    // draw points
    for (size_t i = 0; i < trajectory.size() - 1; ++i) {
        cv::Point2f point = trajectory[i];
        DrawCircle((int)World2Pixel(point.x, 0), (int)World2Pixel(point.y, 1), 3, YELLOW);
    }
    cv::Point2f last_point = trajectory.back();
    DrawCircle((int) World2Pixel(last_point.x,0), (int)World2Pixel(last_point.y,1), 4, BLUE);
    EndDrawing();
}

bool Plot::check_condition(){
    return !WindowShouldClose();
}

Plot::~Plot(){
    if (IsWindowReady()) {
        CloseWindow();
    }
    
    std::cout << "Raylib window closed" << std::endl;
}