//
//  CSVParser.m
//  ReIDd
//
//  Created by Joel Brogan on 12/30/14.
//  Copyright (c) 2014 Joel Brogan. All rights reserved.
//

#import "CSVParser.h"

@implementation CSVParser
+(cv::Mat)ReadCSV:(NSString *)fileName
{
	NSString *fileString = [NSString stringWithContentsOfFile:fileName encoding:NSUTF8StringEncoding error:nil];
	NSArray *rows = [fileString componentsSeparatedByString:@"\r"];
	long rowsCount = rows.count;
	long colsCount = 0;
	std::vector<double> vals;
	for (int i = 0; i < rows.count; i++) {
		NSString *row = [rows objectAtIndex:i];
		NSArray *cols = [row componentsSeparatedByString:@","];
		if (cols.count > colsCount) colsCount = cols.count;
		for (int j = 0; j < cols.count; j++) {
			vals.push_back([[cols objectAtIndex:j] doubleValue]);
		}
	}
	cv::Mat matrix = cv::Mat::zeros(rowsCount, colsCount, CV_64FC1);
	int ind = 0;
	for (int rows = 0; rows < rowsCount; rows++) {
		for (int cols = 0 ; cols < colsCount; cols++) {
			matrix.at<double>(rows,cols) = vals[ind];
			ind++;
		}
	}
	return matrix;
}
@end
