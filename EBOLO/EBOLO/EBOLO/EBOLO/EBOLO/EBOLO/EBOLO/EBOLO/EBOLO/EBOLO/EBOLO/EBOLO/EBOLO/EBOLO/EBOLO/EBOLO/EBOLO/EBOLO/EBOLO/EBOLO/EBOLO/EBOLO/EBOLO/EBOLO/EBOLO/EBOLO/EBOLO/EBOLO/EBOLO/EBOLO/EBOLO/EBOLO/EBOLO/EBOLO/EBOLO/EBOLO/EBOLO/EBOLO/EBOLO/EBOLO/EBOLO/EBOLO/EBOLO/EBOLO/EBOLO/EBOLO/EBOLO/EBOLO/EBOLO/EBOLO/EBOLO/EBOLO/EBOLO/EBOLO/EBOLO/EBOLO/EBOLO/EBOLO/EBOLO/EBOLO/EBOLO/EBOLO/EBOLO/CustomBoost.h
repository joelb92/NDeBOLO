//
//  CustomBoost.h
//  EBOLO
//
//  Created by Joel Brogan on 2/15/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <opencv2/opencv.hpp>
struct StrongClassifier
{
	std::vector<float>A;
	std::vector<int>featureIndexes;
	
};
@interface CustomBoost : NSObject
{
	std::vector<cv::Mat> trainingSet;
	std::vector<bool> trainingSetTruth;
	int numPositive,numNegative;
	int numStageClassifiers;
	std::vector<int>stageFeatureSizes;
	std::vector<std::vector<float> >weights;
	int imageHeight;
	int imageWidth;
	int bitsPerFeature;
	std::vector<std::vector<int> > gp;
	std::vector<std::vector<int> > gn;
	std::vector<StrongClassifier > StrongClassifiers;
}
@end
