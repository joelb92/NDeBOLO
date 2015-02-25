//
//  CustomBoost.m
//  EBOLO
//
//  Created by Joel Brogan on 2/15/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import "CustomBoost.h"

@implementation CustomBoost

-(void)trainClassifierForFeatureAmount:(int)K
{
	//Initialize Weights
	std::vector<float> firstWeights(trainingSet.size());
	std::vector<int> featureLocations;
	for(int i = 0; i < trainingSet.size(); i++)
	{
		float weight;
		if (trainingSetTruth[i]) weight = 1/(2*numPositive);
		else weight = 1/(2*numNegative);
		firstWeights[i] = weight;
	}
	
	weights.push_back(firstWeights);
	for (int f = 0; f < K; f++) {
		//Normalize weights
		float sum = cv::sum(weights[f])[0];
		for (int i = 0; i < weights[f].size(); i++) {
			weights[f][i] = weights[f][i]/sum;
		}
		//Generate a weak classifier ck for a single feature k
		int featureAmount = imageWidth*imageHeight;
		std::vector<float> A(K);
		std::vector<float> E(K);
		float minError = FLT_MAX;
		float minErrorA = FLT_MAX;
		int minErrorFeatureIndex = 0;
		//Iterate all possible feature classifiers and chose single pixel classifier with lowest error
		for (int k = 0; k < featureAmount; k++)
		{
			float error = 0;
			for(int i = 0; i < trainingSet.size(); i++)
			{
				error += weights[k][i]*abs([self weakClassifierForFeature:k imageNumber:i] - trainingSetTruth[i]);
			}
			E[k] = error;
			float a = .5*log((1-error)/error);
			A[k] = a;
			if (error <= minError) {
				minError = error;
				minErrorFeatureIndex = k;
				minErrorA = a;
			}
		}
		featureLocations.push_back(minErrorFeatureIndex);
		std::vector<float> nextWeights(trainingSet.size());
		StrongClassifier C;
		
		for(int i = 0; i < trainingSet.size(); i++)
		{
			float weight = 1;
			if ([self weakClassifierForFeature:minErrorFeatureIndex imageNumber:i] == trainingSetTruth[i]) weight = exp2f(-minErrorA);
			nextWeights[i] = weight*weights[f][i];
		}
	}
	
}

-(float)gkp:(int)r K:(int)k
{
	float sum = 0;
	int x = k%imageWidth;
	int y = k/imageWidth;
	for (int i = 0; i < trainingSet.size(); i++)
	{
		if (trainingSetTruth[i]) {
			cv::Mat img =trainingSet[i];
			if (x < img.cols && y < img.rows) {
				sum += img.at<short int>(y,x) == r;
			}
			else{
				NSLog(@"WARNING: Coord %i, %i is not in bounds for classifier training",x,y);
			}
		}
		
	}
	return sum;
}

-(float)gkn:(int)r K:(int)k
{
	//Extra comment
	float sum = 0;
	int x = k%imageWidth;
	int y = k/imageWidth;
	for (int i = 0; i < trainingSet.size(); i++)
	{
		if (!trainingSetTruth[i]) {
			cv::Mat img =trainingSet[i];
			if (x < img.cols && y < img.rows) {
				sum += img.at<short int>(y,x) == r;
			}
			else{
				NSLog(@"WARNING: Coord %i, %i is not in bounds for classifier training",x,y);
			}
		}
		
	}
	return sum;
}

-(void)buildHistograms
{
	int featureSize = pow(2, bitsPerFeature);
	int featureAmount = imageWidth*imageHeight;
	gp = std::vector<std::vector<int> > (featureAmount);
	gn = std::vector<std::vector<int> > (featureAmount);
	for (int k = 0; k < featureAmount; k++) {
		std::vector<int> gkp(featureSize);
		for (int r = 0; r < featureSize; r++) {
			gkp[r] = [self gkp:r K:k];
		}
		gp[k] = gkp;
	}
	for (int k = 0; k < featureAmount; k++) {
		std::vector<int> gkn(featureSize);
		for (int r = 0; r < featureSize; r++) {
			gkn[r] = [self gkn:r K:k];
		}
		gn[k] = gkn;
	}
	
}

-(bool)weakClassifierForFeature:(int)k imageNumber:(int)i
{
	return gp[k][trainingSet[i].at<short int>(k/imageWidth,k%imageWidth)] > gn[k][trainingSet[i].at<short int>(k/imageWidth,k%imageWidth)];
}

@end
