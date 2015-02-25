//
//  CSVParser.h
//  ReIDd
//
//  Created by Joel Brogan on 12/30/14.
//  Copyright (c) 2014 Joel Brogan. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "opencv2/opencv.hpp"
@interface CSVParser : NSObject
+(cv::Mat)ReadCSV:(NSString *)fileName;
@end
