//
//  HOGPersonDetector.m
//  EBOLO
//
//  Created by Joel Brogan on 2/3/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//
//http://www.magicandlove.com/blog/2011/08/26/people-detection-in-opencv-again/
#import "HOGPersonDetector.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
@implementation HOGPersonDetector
-(void) detect
{
    VideoCapture cap("/Users/joel/Documents/triskett_bridge/tsktt_bridge_morning.h264.mp4");
    if (!cap.isOpened())
        return;
    
    Mat img;
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    namedWindow("video capture", CV_WINDOW_AUTOSIZE);
    while (true)
    {
        cap >> img;
        if (!img.data)
            continue;
        for (int i = 0; i < 10; i++) {
            cap.grab();
        }
        vector<cv::Rect> found, found_filtered;
        hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
        
        size_t i, j;
        for (i=0; i<found.size(); i++)
        {
            cv::Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }
        for (i=0; i<found_filtered.size(); i++)
        {
            cv::Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.06);
            r.height = cvRound(r.height*0.9);
            rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
        }
        imshow("video capture", img);
        if (waitKey(20) >= 0)
            break;
    }
    return;
}
@end
