//
//  MCTBoost.h
//  EBOLO
//
//  Created by Joel Brogan on 2/25/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <opencv2/opencv.hpp>
#import "MCT.h"
@interface MCTBoost : NSObject
{
	std::vector<float>truthLabels;
	std::vector<cv::Mat>images;
	cv::Mat trainingMat;
	cv::Mat labelsMat;
}
-(void)buildTrainingSetYaleFaces:(NSString *)folderPath;
-(void)run;
-(std::vector<cv::Rect>)slidingWindowDetection:(cv::Mat)img;
@end
