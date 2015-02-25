//
//  bgfg_vibe.h
//  EBOLO
//
//  Created by Joel Brogan on 2/25/15.
//  Copyright (c) 2015 Joel Brogan. All rights reserved.
//

#ifndef __EBOLO__bgfg_vibe__
#define __EBOLO__bgfg_vibe__

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct Model {
	cv::Mat*** samples;
	cv::Mat** fgch;
	cv::Mat* fg;
};

class bgfg_vibe
{
#define rndSize 256
	unsigned char ri;
#define rdx ri++
public:
	bgfg_vibe();
	int N,R,noMin,phi;
	void init_model(cv::Mat& firstSample);
	void setphi(int phi);
	cv::Mat* fg(cv::Mat& frame);
private:
	bool initDone;
	cv::RNG rnd;
	Model* model;
	void init();
	void fg1ch(cv::Mat& frame,cv::Mat** samples,cv::Mat* fg);
	int rndp[rndSize],rndn[rndSize],rnd8[rndSize];
};
#endif /* defined(__EBOLO__bgfg_vibe__) */
