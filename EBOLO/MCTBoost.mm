//
//  MCTBoost.m
//  EBOLO
//
//  Created by Joel Brogan on 2/25/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#import "MCTBoost.h"

@implementation MCTBoost
-(void)run
{
	// Training data
	[self buildTrainingSetYaleFaces:@"/Users/joel/Documents/TrainingFaces"];
	MCT *mct = [[MCT alloc] init];
	NSLog(@"Transforming image set...");
	int width = 15;
	int height = 15;
	trainingMat = cv::Mat((int)images.size(),width*height, CV_32FC1);
	labelsMat = cv::Mat((int)images.size(),1,CV_32FC1);
	for(int i = 0; i <images.size(); i++)
	{
		cv::Mat im = images[i];
		cv::resize(im, im, cv::Size(height,width));
		im = [mct ModifiedColorCensusTransform:im];
		
		
		int j = 0;
		for (int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++)
			{
				trainingMat.at<float>(i,j) = im.at<short int>(y,x);
				j++;
			}
		}
		labelsMat.at<float>(i,0) = truthLabels[i];
	}
	
	// Train a boost classifier
	cv::Boost boost;
	cv::BoostParams boostSettings;
	boostSettings.boost_type = CvBoost::REAL;
	boostSettings.weak_count = 100;
	boostSettings.weight_trim_rate = .95;
	boostSettings.cv_folds = 0;
	boostSettings.max_depth = 1;
	NSLog(@"training...");
	boost.train(trainingMat,
				CV_ROW_SAMPLE,
				labelsMat);
	NSLog(@"done!");
	boost.save("/Users/joel/Documents/TrainingFaces/trained_boost.xml");
	// Test the classifiers
	cv::Mat testSample1 = (cv::Mat_<float>(1,2) << 251, 5);
	cv::Mat testSample2 = (cv::Mat_<float>(1,2) << 502, 11);
	
//	float svmResponse1 = SVM.predict(testSample1);
//	float svmResponse2 = SVM.predict(testSample2);
	
//	float boostResponse1 = boost.predict(testSample1);
//	float boostResponse2 = boost.predict(testSample2);
	
//	std::cout << "SVM:   " << svmResponse1 << " " << svmResponse2 << std::endl;
//	std::cout << "BOOST: " << boostResponse1 << " " << boostResponse2 << std::endl;
}

-(void)buildTrainingSetYaleFaces:(NSString *)folderPath
{
	NSString *faceFolderPath = [folderPath stringByAppendingPathComponent:@"rawFaces"];
	NSString *nonFaceFolderPath = [folderPath stringByAppendingPathComponent:@"rawNonFaces"];

	NSFileManager *fm = [NSFileManager defaultManager];
	NSArray *dirContents = [fm contentsOfDirectoryAtPath:faceFolderPath error:nil];
	NSPredicate *fltr = [NSPredicate predicateWithFormat:@"self ENDSWITH '.ppm'"];
	NSArray *onlyppms = [dirContents filteredArrayUsingPredicate:fltr];
	NSLog(@"Loading face set");
	for(NSString *fileName in onlyppms)
	{
		NSString *fullPath = [faceFolderPath stringByAppendingPathComponent:fileName];
		std::cout << ".";
		cv::Mat img = cv::imread(fullPath.UTF8String);
		images.push_back(img);
		truthLabels.push_back(1.0);
	}
	std::cout << std::endl;
	NSLog(@"loading non-face set");
	dirContents = [fm contentsOfDirectoryAtPath:nonFaceFolderPath error:nil];
	fltr = [NSPredicate predicateWithFormat:@"self ENDSWITH '.jpg'"];
	NSArray *onlyjpgs = [dirContents filteredArrayUsingPredicate:fltr];
	for(NSString *fileName in onlyjpgs)
	{
		NSString *fullPath = [nonFaceFolderPath stringByAppendingPathComponent:fileName];
		std::cout << ".";
		cv::Mat img = cv::imread(fullPath.UTF8String);
		images.push_back(img);
		truthLabels.push_back(-1.0);
		truthLabels.push_back(1.0);
	}
	std::cout << std::endl;
}
-(void)loadClassifier:(NSString *)classifierPath
{
	
}
-(std::vector<cv::Rect>)slidingWindowDetection:(cv::Mat)img
{
	cv::Boost boost;
	boost.load("/Users/joel/Documents/TrainingFaces/trained_boost.xml");
	MCT *mct = [[MCT alloc] init];
	cv::Mat mctImg = [mct ModifiedColorCensusTransform:img];
//	mctImg.convertTo(mctImg, CV_32FC(1));
//	cv::imshow("mct", mctImg);
	int windowWidth = 15;
	int windowHeight = 15;
	std::vector<cv::Rect> rects;
	for (int x = 0; x < mctImg.cols-windowWidth; x++) {
		for (int y = 0; y < mctImg.rows-windowHeight; y++)
		{
			cv::Rect r(x,y,windowWidth,windowHeight);
			cv::Mat window = mctImg(r);
			cv::Mat windowFlat(1,windowWidth*windowHeight,CV_32FC1);
			int j = 0;
			for (int x1 = 0; x1 < window.cols; x1++) {
				for (int y1 = 0; y1 < window.rows; y1++)
				{
					windowFlat.at<float>(0,j) = window.at<short int>(y1,x1);
					j++;
				}
			}
			float p = boost.predict(windowFlat);
			if(p >0)
			{
//				NSLog(@"%f",p);
				rects.push_back(r);
//				NSLog(@"found");
			}
		}
	}
	return rects;
}
@end
