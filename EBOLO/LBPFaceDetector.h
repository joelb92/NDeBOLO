//
//  LBPFaceDetector.h
//  EBOLO
//
//  Created by Joel Brogan on 2/3/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface LBPFaceDetector : NSObject
{
    bool shouldStop;
}
@property bool shouldStop;
-(int)detect;
@end
