//
//  AppDelegate.m
//  EBOLO
//
//  Created by Joel Brogan on 2/3/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import "AppDelegate.h"
#import "LBPFaceDetector.h"
#import "HOGPersonDetector.h"
#import "opencv2/opencv.hpp"
#import "SilhouettePartitioner.h"
#import "CSVParser.h"
#import "ParticleClusterer.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/video/background_segm.hpp"
#include "opencv2/legacy/legacy.hpp"
@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;
@end

@implementation AppDelegate

using namespace std;
using namespace cv;
//String face_cascade_name = "haarcascade_frontalface_alt.xml";
//String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//string window_name = "Capture - Face detection";
//RNG rng(12345);
//
//Mat frame; //current frame
//Mat fgMaskMOG; //fg mask generated by MOG method
//Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
//Ptr< BackgroundSubtractor> pMOG; //MOG Background subtractor  BackgroundSubtractor pMOG2; //MOG2 Background subtractor
int keyboard;
- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    f = [[LBPFaceDetector alloc] init];
    HOGPersonDetector *p = [[HOGPersonDetector alloc] init ];
    mct = [[MCT alloc] init];
//    [f detect];
//    [self faceCascadeClassifier];
	[self BS];
//    [self basicDetection];
}

-(void)videoSave
{
	VideoCapture* capture2 = new VideoCapture("/Users/joel/Desktop/Crowd_Short.mp4" );
	cv::Size size2 = cv::Size(352,240);
	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	// Unlike in C, here we use an object of the class VideoWriter//
	VideoWriter writer2("video_.avi",codec,15.0,size2,true);
	writer2.open("video_.avi",codec,15.0,size2,true);
	if(writer2.isOpened())
	{
		int a = 100;
		Mat frame2;
		while ( a > 0 ) {
			capture2->read(frame2);
			imshow("live",frame2);
			waitKey(10);
			writer2.write(frame2);
			a--;
		}
	}
	else
	{
		cout<<"ERROR while opening"<<endl;
	}
	//No Need to release the Writer as the distructor will called automatically
	capture2->release();
	return;
}

-(void)basicDetection
{
    //load in video
    cv::VideoCapture cap;
    cap.open("/Users/joel/Documents/triskett_bridge/tsktt_bridge_morning.h264.mp4");
    cv::Mat frame,grayframe,prevFrame;
    int frameNum = 0;
    while (true) {
        cap >> frame;
        for(int i = 0; i < 10; i++) cap.grab();
        cv::imshow("frame", frame);
        cv::Mat censusImage = [mct ModifiedColorCensusTransform:frame];
        cv::imshow("census", censusImage);
        cv::waitKey(10);
        prevFrame = grayframe;
        frameNum++;
    }
}

-(IBAction)stop:(id)sender
{
    f.shouldStop = true;
}
//-(void)faceCascadeClassifier
//{
//
//    NSString *fileName = @"/Users/joel/Documents/triskett_bridge/tsktt_bridge_afternoon.h264.mp4";
//    cv::VideoCapture capture(fileName.UTF8String);
//    Mat frame;
//    frameNum = 0;
//    totalDetections = 0;
//    rectDictionary = [NSMutableDictionary dictionary];
//    orderedKeys = [NSMutableArray array];
//    pMOG = new cv::BackgroundSubtractorMOG(3,4,.7);  //MOG approach
//    //-- 1. Load the cascades
//    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return; };
//    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return; };
//    //-- 2. Read the video stream
//    if( capture.isOpened() )
//    {
//        while( true )
//        {
//            for(int i = 0; i < 10; i++)
//            {
//                capture.grab(); //get rid of dupe frames
//                frameNum++;
//            }
//            capture >> frame;
//            //update the background model
//            pMOG->operator()(frame, fgMaskMOG);
//            
//            cv::morphologyEx(fgMaskMOG, fgMaskMOG, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9,9)));
//            cv::imshow("Foreground", fgMaskMOG);
//            std::vector<std::vector<cv::Point> >conts;
//            cv::findContours(fgMaskMOG.clone(), conts, cv::noArray(), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
//            std::vector<cv::Rect> ROIs;
//            float resizeFactor = 2.5;
//            cv::resize(frame, frame, cv::Size(frame.cols*resizeFactor,frame.rows*resizeFactor));
//            for (int i = 0; i < conts.size(); i++) {
//                std::vector<cv::Point> cont = conts[i];
//                if (cv::contourArea(cont) > 10) {
//                    cv::Rect r = cv::boundingRect(cont);
//                    r.x = r.x*resizeFactor; r.y = r.y*resizeFactor; r.width = r.width*resizeFactor; r.height = r.height*resizeFactor;
//                    ROIs.push_back(r);
//                    std::vector<cv::Rect> faces = [self detectAndDisplay:frame(r).clone()];
//                    if (faces.size() > 0) {
//                        for( size_t i = 0; i < faces.size(); i++ )
//                        {
//                            cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
//                            ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
//                            cv::imwrite([[fileName stringByDeletingPathExtension] stringByAppendingFormat:@"detection%i.jpg",totalDetections].UTF8String, frame(faces[i]));
//                            totalDetections++;
//                        }
//                    }
//                }
//            }
//            for (int i = 0; i < ROIs.size(); i++) {
//                cv::Rect r  = ROIs[i];
//                cv::line(frame, cv::Point(r.x,r.y), cv::Point(r.x+r.width,r.y), cv::Scalar(0,0,255));
//                cv::line(frame, cv::Point(r.x+r.width,r.y), cv::Point(r.x+r.width,r.y+r.height), cv::Scalar(0,0,255));
//                cv::line(frame, cv::Point(r.x+r.width,r.y+r.height), cv::Point(r.x,r.y+r.height), cv::Scalar(0,0,255));
//                cv::line(frame, cv::Point(r.x,r.y+r.height), cv::Point(r.x,r.y), cv::Scalar(0,0,255));
//                
//            }
//            cv::imshow(window_name, frame);
//            
//            //-- 3. Apply the classifier to the frame
//            if( !frame.empty() )
//            {  }
//            else
//            { printf(" --(!) No captured frame -- Break!"); break; }
//            
//            int c = waitKey(10);
//            if( (char)c == 'c' ) { break; }
//            frameNum++;
//        }
//    }
//    NSMutableString *mString = [@"frame,x,y,width,height\n" mutableCopy];
//    for (int i = 0; i < orderedKeys.count; i++) {
//        NSArray *rects = [rectDictionary objectForKey:[orderedKeys objectAtIndex:i]];
//        int frameNumber = [[orderedKeys objectAtIndex:i] intValue];
//        for (int j = 0; j < rects.count; j++) {
//            NSRect r = [[rects objectAtIndex:j] rectValue];
//            NSString *s = [NSString stringWithFormat:@"%i,%i,%i,%i,%i\n",frameNumber,(int)r.origin.x,(int)r.origin.y,(int)r.size.width,(int)r.size.height];
//            [mString appendString:s];
//            NSLog(s);
//        }
//    }
//    [ mString writeToFile:[[fileName stringByDeletingPathExtension] stringByAppendingString:@"FACERECTS.csv"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
//    return;
//}
///** @function detectAndDisplay */
//-(std::vector<cv::Rect>) detectAndDisplay:(cv::Mat)frame
//{
//    std::vector<cv::Rect> faces;
//    Mat frame_gray;
//    
//    cvtColor( frame, frame_gray, CV_BGR2GRAY );
//    equalizeHist( frame_gray, frame_gray );
//    
//    //-- Detect faces
//    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
//    
//    for( size_t i = 0; i < faces.size(); i++ )
//    {
//        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
//        ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
//        
//        Mat faceROI = frame_gray( faces[i] );
//        
//    }
//    if (faces.size() > 0) {
//        NSMutableArray *faceArr = [NSMutableArray arrayWithCapacity:faces.size()];
//        for (int i = 0; i <faces.size(); i++) {
//            cv::Rect r = faces[i];
//            [faceArr addObject:[NSValue valueWithRect:NSMakeRect(r.x, r.y, r.width, r.height)]];
//        }
//        [rectDictionary setObject:faceArr forKey:@(frameNum)];
//        [orderedKeys addObject:@(frameNum)];
//    }
//    //-- Show what you got
//    return faces;
//}

-(void)BS
{
	//load in video
	//	pMOG2->set("nmixtures", 3);
	//	pMOG2->set("", <#int value#>)
	
	cv::VideoCapture cap;
	cap.open("/Users/joel/Desktop/Crowd_Short.mp4");
	cv::Mat frame,grayframe,prevFrame,mog;
	int frameNum = 0;
	bgfg_vibe bgfg;
	cap >> frame;
	bgfg.init_model(frame);
	
	while (true) {
		for (int i = 0; i < 10; i++) cap.grab();
		prevFrame = frame;
		cap >> frame;
		cv::cvtColor(frame, frame, CV_BGR2GRAY);
		//		cv::GaussianBlur(frame, frame, cv::Size(3,3), 4);
		if (!prevFrame.empty()) {
			cv::Mat sub = cv::Mat::zeros(frame.rows, frame.cols, CV_32FC1);
			cv::Mat subc = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			int maxVal = 0;
			
			for(int x = 0; x < frame.cols; x++)
			{
				for (int y = 0; y < frame.rows; y++) {
					int cur = frame.at<uchar>(y,x);
					int last = prevFrame.at<uchar>(y,x);
					int newVal = abs(cur-last);
					if (newVal > maxVal) {
						maxVal = newVal;
					}
					sub.at<float>(y,x) = newVal;
				}
			}
			for(int x = 0; x < frame.cols; x++)
			{
				for (int y = 0; y < frame.rows; y++)
				{
					int newVal =sub.at<float>(y,x)/maxVal*255;
					if (newVal > 50) {
						subc.at<uchar>(y,x) = newVal;
					}
					
				}
			}
			cv::Mat subMask,subMaskNoHoles,subMaskNoHolesConvex;
			subMaskNoHoles = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			subMaskNoHolesConvex = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			cv::threshold(subc, subMask, 60, 255, CV_THRESH_BINARY);
			cv::dilate(subMask, subMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15)));
			cv::erode(subMask, subMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15)));
			std::vector<std::vector<cv::Point> > conts,newConts;
			
			std::vector<Vec4i> hierarchy;

			cv::findContours(subMask.clone(), conts,hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
			cv::drawContours(subMaskNoHoles, conts, -1, cv::Scalar(255,255,255),-1,8,hierarchy,1);
			cv::findContours(subMaskNoHoles.clone(), conts,hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
			for (int i = 0; i < conts.size(); i++)
			{
				if (conts[i].size() > 0 && cv::contourArea(conts[i]) > 20) {
					std::vector<cv::Point>cont;
					cv::convexHull(conts[i], cont);
					newConts.push_back(cont);
				}
			}
			cv::drawContours(subMaskNoHolesConvex, newConts, -1, cv::Scalar(255,255,255),-1,8);
			cv::Mat masked,foreground;
			cv::bitwise_and(frame, subMaskNoHolesConvex, masked);
			cv::imshow("sub", subc);
			cv::imshow("subMask", subMask);
			cv::imshow("subFilled", subMaskNoHoles);
			cv::imshow("subMaskFilledConvex",subMaskNoHolesConvex);
			cv::imshow("Masked", masked);
			
			
			foreground = *bgfg.fg(frame);
			cv::imshow("vibe", foreground);
		}
		cv::imshow("frame", frame);
		cv::waitKey();
		prevFrame = grayframe;
		frameNum++;
	}
}





-(void) runClusterTest
{
    SilhouettePartitioner *partitioner = [[SilhouettePartitioner alloc] init];
    ParticleClusterer *clusterer = [[ParticleClusterer alloc] init];
    cv::Mat matrix = [CSVParser ReadCSV:@"/Users/joel/Documents/Notre Dame/Research/ReID/RandomClusters.csv"];
    clusterOutput output = [clusterer mixtures4forData:matrix KMin:1 KMax:25 regFactor:0 StoppingThresh:1*powf(10,-4) CovarianceOption:1];
    
}

@end