//
//  AppDelegate.h
//  EBOLO
//
//  Created by Joel Brogan on 2/3/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import "LBPFaceDetector.h"
#import "MCT.h"
#include "bgfg_vibe.h"
#import "MCTBoost.h"
@interface AppDelegate : NSObject <NSApplicationDelegate>
{
    int frameNum;
    NSMutableDictionary *rectDictionary;
    NSMutableArray *orderedKeys;
    int totalDetections;
    LBPFaceDetector *f;
    MCT *mct;
}

@end

