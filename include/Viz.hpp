#ifndef VIZ_HPP
#define VIZ_HPP


#include "Utils.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"
#include <pangolin/display/display.h>
#include <pangolin/plot/plotter.h>

class Viewer {
    public:
        typedef std::shared_ptr<Viewer> Ptr; 
        Viewer();
        Map::Ptr viewerMap;
        void addCurrentFrame(Frame::Ptr frame);
        // Pin the map used for visualization 
        void setMap(Map::Ptr _map);
        // Every now and then lock the viewer thread and update KFs and MPs 
        void updateMap();
        // main thread loop that plots all KFs and MPs
        void plotterLoop();
        // draw single frame
        void drawFrame(Frame::Ptr frame, const float *color);
        // draw single mappoint
        void drawMPs();
        // follow camera when drawing
        void followCurrentFrame(pangolin::OpenGlRenderState& visCamera);
        // Close everything 
        void close();
        void viewerRun();
        cv::Mat plotFromImage();
        // static Viewer::Ptr createViewer();
        std::thread viewerThread;

    private:
        std::mutex viewerMutex;
        Map::Ptr map = nullptr;
        Frame::Ptr currentFrame = nullptr;
        bool viewerRunning = true;
        std::unordered_map<unsigned long, MapPoint::Ptr> mps;
        std::unordered_map<unsigned long, Frame::Ptr> frames;
};

#endif // TODOITEM_H