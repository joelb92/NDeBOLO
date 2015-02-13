////
////  LBPFaceDetector.m
////  EBOLO
////
////  Created by Joel Brogan on 2/3/15.
////  Copyright (c) 2015 Joel Brogan. All rights reserved.
////
//
#import "LBPFaceDetector.h"
#import "opencv2/opencv.hpp"
#import "opencv2/objdetect/objdetect.hpp"
#import "opencv2/video/video.hpp"
#import "opencv2/highgui/highgui.hpp"
#import "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
//using namespace std;
using namespace cv;
/** Function Headers */
//void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
cv::Mat mask,addEmUp;
int maxSize = 0;
int total = 0;
@implementation LBPFaceDetector
@synthesize shouldStop;
/**
 * @function main
 */
-(int) detect
{
    VideoCapture capture;
    Mat frame;
    addEmUp = cv::Mat::zeros(0,0,CV_32FC1);
    mask = cv::imread("/Users/joel/Documents/triskett_bridge/Trisket_mask.png");
    cv::threshold(mask, mask, 126, 255, CV_THRESH_BINARY);
    cv::imshow("mask", mask);
    //-- 1. Load the cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    
    //-- 2. Read the video stream
    capture.open( "/Users/joel/Documents/triskett_bridge/tsktt_bridge_morning.h264.mp4" );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
    
    while ( capture.read(frame) && !shouldStop )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        if (addEmUp.cols == 0) {
            addEmUp = cv::Mat::zeros(frame.rows, frame.cols, CV_32FC1);
        }
        for(int i = 0; i < 10; i++) capture.grab();
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        
        //-- bail out if escape was pressed
        int c = waitKey(10);
        if( (char)c == 27 ) { break; }
    }
    addEmUp = addEmUp/maxSize; //normalize!;
    addEmUp = addEmUp*255;
    addEmUp.convertTo(addEmUp, CV_8UC1);
    NSLog(@"MaxVal: %i",maxSize);
    NSLog(@"TotalDetections: %i",total);
    cv::dilate(addEmUp, addEmUp, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10,10)));
    cv::imwrite("/Users/joel/Library/Developer/Xcode/DerivedData/EBOLO-dxzotbkiffzcnjcimmgaznvoyteo/Build/Products/Debug/totalDetections.jpg", addEmUp);
    return 0;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
    std::vector<cv::Rect> faces;
    Mat frame_gray,frame2;
    frame.copyTo(frame2, mask);
    float resize = 4.5;
    cv::resize(frame2, frame2, cv::Size(frame.cols*resize,frame.rows*resize));
    cvtColor( frame2, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 3, 0, cv::Size(15, 15),cv::Size(85,85) );
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = frame_gray( faces[i] );
        std::vector<cv::Rect> eyes;
        
        //-- In each face, detect eyes

            //-- Draw the face
            cv::Point center( faces[i].x/resize + faces[i].width/2/resize, faces[i].y/resize + faces[i].height/2/resize );
            ellipse( frame, center, cv::Size( faces[i].width/2/resize, faces[i].height/2/resize ), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
        float newVal =addEmUp.at<float>(center)+1;
        addEmUp.at<float>(center) = newVal;
        if (newVal > maxSize) maxSize = newVal;
        total++;
       
    }

    //-- Show what you got
    imshow( window_name, frame );
    cv::imshow("Density Map", addEmUp);
}
@end
