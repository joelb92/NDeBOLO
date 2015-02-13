//
//  ParticleClusterer.h
//  ReIDd
//
//  Created by Joel Brogan on 11/4/14.
//  Copyright (c) 2014 Joel Brogan. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#include "opencv2/opencv.hpp"
struct clusterOutput
{
	cv::Mat bestpp,bestmu;
	std::vector<cv::Mat> bestcov;
	int countf, bestk;
	std::vector<double>dl;
	
};
@interface ParticleClusterer : NSObject
- (clusterOutput) mixtures4forData:(cv::Mat)y KMin:(int)kmin KMax:(int)kmax regFactor:(float)regularize StoppingThresh:(float)th CovarianceOption:(int)covoption;
@end
