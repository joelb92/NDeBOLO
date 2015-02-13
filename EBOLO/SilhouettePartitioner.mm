//
//  SilhouettePartitioner.m
//  ReIDd
//
//  Created by Joel Brogan on 10/30/14.
//  Copyright (c) 2014 Joel Brogan. All rights reserved.
//

#import "SilhouettePartitioner.h"
@implementation SilhouettePartitioner
-(void)partitionImage:(cv::Mat) img
{
    cv::imshow("img", img);
}
@end
