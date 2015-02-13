//
//  SilhouettePartitioner.h
//  ReIDd
//
//  Created by Joel Brogan on 10/30/14.
//  Copyright (c) 2014 Joel Brogan. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "opencv2/opencv.hpp"
@interface SilhouettePartitioner : NSObject
-(void)partitionImage:(cv::Mat) img;
@end
