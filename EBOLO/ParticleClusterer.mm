//
//  ParticleClusterer.m
//  ReIDd
//
//  Created by Joel Brogan on 11/4/14.
//  Copyright (c) 2014 Joel Brogan. All rights reserved.
//

#import "ParticleClusterer.h"
using namespace std;
using namespace cv;
@implementation ParticleClusterer
- (clusterOutput) mixtures4forData:(cv::Mat)y KMin:(int)kmin KMax:(int)kmax regFactor:(float)regularize StoppingThresh:(float)th CovarianceOption:(int)covoption
{
	int verb = 1; //verbose mode;change to zero for silent mode
	int bins = 40; //number of bins for the univariate data historgrams for visualization
	int npoints = y.cols;
	int dimens = y.rows;
	float npars;
	switch (covoption) {
		case 0:
			npars = (dimens + dimens*(dimens+1)/2);
			//this is for free covariance matrices
			break;
		case 1:
			npars = 2*dimens;
			//this is for diagonal covariance matrices
			break;
		case 2:
			npars = dimens;
			break;
		case 3:
			npars = dimens;
			break;
		default:
			npars = (dimens + dimens*(dimens+1)/2);
			break;
	}
	float nparsover2 = npars/2;
	
	// we choose which axes to use in the plot,
	// in case of higher dimensional data (>2)
	// Change this to have other axes being shown
	int axis1 = 1;
	int axis2 = 2;
	
	// kmax is the initial number of mixture components
	int k = kmax;
	
	// indic will contain the assignments of each data point to
	// the mixture components, as result of the E-step
	//	indic = zeros(k,npoints);
	cv::Mat indic = cv::Mat::zeros(k, npoints, CV_64F);
	// Initialization: we will initialize the means of the k components
	// with k randomly chosen data points. Randperm(n) is a MATLAB function
	// that generates random permutations of the integers from 1 to n.
	std::vector<int> randindex = randperm(npoints);
	randindex = std::vector<int>(&randindex[0],&randindex[k]);
	cv::Mat estmu(dimens,(int)randindex.size(),CV_64F);
	for (int i = 0; i < randindex.size(); i++) {
		y.col(randindex[i]).copyTo(estmu.col(i));
	}

	// the initial estimates of the mixing probabilities are set to 1/k
	cv::Mat estpp = (1.0/k)*cv::Mat::ones(1, k, CV_64F); //THIS SHOULD BE OPTIMIZED
	
	// here we compute the global covariance of the data
	cv::Mat globcov = cov(y.t());
    std::vector<cv::Mat> estcov(k);
	cv::Mat d2 = getDiag(globcov/10);
	double m = 0.0;
	cv::minMaxLoc(d2, NULL,&m);
	cv::Mat d1 = getDiag(cv::Mat::ones(1, dimens, CV_64F)*m);
    for(int i = 0; i < k; i++)
	{
        estcov[i] = d1;
	}

    //we plot the data and the initial estimates
    
    
    //having the initial means, covariances, and probabilities, we can
    //initialize the indicator functions following the standard EM equation
    //Notice that these are unnormalized values
    cv::Mat semi_indic(k,npoints,CV_64F);
    for(int i = 0; i < k; i++)
    {
        cv::Mat m = multinorm(y, estmu.col(i), estcov[i]);
		
        m.copyTo(semi_indic.row(i));
		cv::Mat(semi_indic.row(i)*estpp.at<double>(0,i)).copyTo(indic.row(i));
    }
    //we can use the indic variables (unnormalized) to compute the
    //loglikelihood and store it for later plotting its evaluation
    //we also compute and store the number of components
    std::vector<double>loglike,dl,kappas;
    int countf = 0;
    cv::Mat reduced;
    cv::reduce(DBL_MIN+indic, reduced, 0, CV_REDUCE_SUM);
    cv::log(reduced, reduced);
    double s = cv::sum(reduced)[0];
    loglike.push_back(s);
    cv::Mat log2;
    cv::log(estpp,log2);
    double dlength = -loglike[countf] + nparsover2*cv::sum(log2)[0] + (nparsover2 + .5)*k*log(npoints);
    dl.push_back(dlength);
    kappas.push_back(k);
    
    //the transition vectors will store the iteration
    //number at which components are killed.
    //transitions1 stores the interations at which components are
    //killed by the M-step, while transitions2 stores the iterations at which
    //we force components to zero.
    std::vector<int> transitions1, transitions2;
    
    //minimum description length seen so far, and corresponding
    //parameter estimates
    double mindl = dl[countf];
    cv::Mat bestpp = estpp;
    cv::Mat bestmu = estmu;
    std::vector<cv::Mat> bestcov = estcov;
    int bestk = k;
    int innercount = 0;
    int k_cont = 1; //auxiliary variable for the outer loop
    while (k_cont)  //the outer loop will take us down from kmax to kmin components
    {
//		NSLog(@"1");
        int cont = 1;   //auxiliary variable of the inner loop
        while(cont)
        {
//			NSLog(@"2");
            if (verb == 1) {
                //verbose mode not impelemented
            }
            //we begin at component 1
            int comp = 0;
            // ...and can only go to the last component, k.
            //Since k may change during the process, we can not use for a loop
            while (comp < k) {
//				NSLog(@"3");
				innercount++;
				NSLog(@"inner: %i",innercount);
                //we start with the M step
                //first, we compute a normalized indicator function
                indic = cv::Mat(k,npoints,CV_64F);

                for(int i = 0; i < k; i++)
                {
                    cv::Mat tmp = semi_indic.row(i)*estpp.at<double>(0,i);
                    tmp.copyTo(indic.row(i));
                }
                cv::Mat r;
                cv::reduce(indic, r, 0, CV_REDUCE_SUM);
                cv::Mat normindic;
				
                cv::divide(indic,(DBL_MIN+kron3(cv::Mat::ones(k, 1, CV_64F), r)),normindic);
				

                //now we perform the standard M-step for mean and covariance
                double normalize = 1/cv::sum(normindic.row(comp))[0];
                cv::Mat aux = kron3(normindic.row(comp), cv::Mat::ones(dimens, 1, CV_64F));
                cv::multiply(aux, y, aux);
                cv::reduce(aux, r, 1, CV_REDUCE_SUM);
                cv::Mat mul = normalize*r;
                mul.copyTo(estmu.col(comp));
                if (covoption == 0 || covoption == 2) { //USED
					cv::Mat m1 = normalize*(aux*y.t());
					cv::Mat m2 = estmu.col(comp)*estmu.col(comp).t();
					cv::Mat m3 = regularize*cv::Mat::eye(dimens, dimens, CV_64FC1);
					cv::Mat m4 =m1-m2+m3;
                    estcov[comp] = m4;
                }
                else //unused
                {
                    cv::multiply(aux, y, r);
                    cv::reduce(r, r, 1, CV_REDUCE_SUM);
                    cv::Mat p;
                    cv::pow(estmu.col(comp), 2, p);
					cv::Mat covar = normalize*getDiag(r) - getDiag(p);
//					if (isnan(covar.at<double>(0,0)))
//					{
//						NSLog(@"bad");
//					}
                    estcov[comp] = covar;
                }
                if (covoption == 2) { //unused
                    cv::Mat comcov = cv::Mat::zeros(dimens, dimens, CV_64F);
                    for(int comp2 = 0; comp2 < k; comp2++)
                    {
                        comcov = comcov + estpp.at<double>(0,comp2)*estcov[comp2];
                    }
                    for(int comp2 = 0; comp2 < k; comp2++)
                    {
                        estcov[comp2] = comcov;
                    }
                }
                if (covoption == 3) { //unused
                    cv::Mat comcov = cv::Mat::zeros(dimens,dimens,CV_64F);
                    for(int comp2 = 0; comp2 < k; comp2++)
                    {
                        comcov = comcov + estpp.at<double>(0,comp2)*getDiag(getDiag(estcov[comp2]));
                    }
                    for(int comp2 = 0; comp2 < k; comp2++)
                    {
                        estcov[comp2] = comcov;
                    }
                }

                //this is the special part of the M step that is able to
                //kill components
				
				double max = maxd(cv::sum(normindic.row(comp))[0]-nparsover2,0)/npoints;
				estpp.at<double>(0,comp) = maxd(cv::sum(normindic.row(comp))[0]-nparsover2,0)/npoints;
                estpp = estpp/(cv::sum(estpp)[0]);
                //this is an auxiliary variable that will be used the
                //signal is killing of the current component beign updated
                bool killed = false;
                
                //we now have to do some book-keeping if the current component was killed
                //that is, we have to rearrange the vectors and matries that store the
                //parameter estimates
                if (estpp.at<double>(0,comp) == 0) {
                    killed = true;
                    //we also register that at the current iteration a component was killed
                    transitions1.push_back(countf);
                    if(comp == 0)
                    {
                        estmu = estmu.colRange(1, k);
						estcov.erase(estcov.begin());
                        estpp = estpp.colRange(1, k);
                        semi_indic = semi_indic.rowRange(1, k);
                    }
                    else
                    {
                        if (comp==k-1) {
                            estmu = estmu.colRange(0, k-1);
                            estcov = std::vector<cv::Mat>(&estcov[0],&estcov[k-1]);
                            estpp = estpp.colRange(0, k-1);
                            semi_indic = semi_indic.rowRange(1, k-1);
                        }
                        else{
							if (comp > 0 && comp+1 < k) {
								hconcat(estmu.colRange(0, comp), estmu.colRange(comp+1, k), estmu);
								hconcat(estpp.colRange(0, comp), estpp.colRange(comp+1, k), estpp);
								vconcat(semi_indic.rowRange(0, comp), semi_indic.rowRange(comp+1, k), semi_indic);
								
							}
							else if (comp == 0)
							{
								estmu = estmu.colRange(comp+1, k);
								estpp = estpp.colRange(comp+1, k);
								semi_indic = semi_indic.rowRange(comp+1, k);

							}
							else if (comp+1 == k)
							{
								estmu = estmu.colRange(0, comp);
								estpp = estpp.colRange(0, comp);
								semi_indic = semi_indic.rowRange(0, comp);
							}
							std::vector<cv::Mat> newcov(k-1);
							for(int kk = 0; kk < comp; kk++)
							{
								newcov[kk] = estcov[kk];
							}
							for (int kk = comp+1; kk < k; kk++) {
								newcov[kk-1] = estcov[kk];
							}
							estcov = newcov;
                        }
                    }
					// since we've just killed a component, k must decrease
					k--;
                }
				if(covoption == 2 || covoption ==3)
				{
					if (innercount == 25) {
						int t =2;
					}
					for (int kk = 0; kk < k; kk++) {
						multinorm(y, estmu.col(kk), estcov[kk]).copyTo(semi_indic.row(kk));
					}
				}
				
				if (!killed) {
					//if the component was not killed, we update the corresponding
					//indicator variables...

					cv::Mat mn =multinorm(y, estmu.col(comp), estcov[comp]);
					mn.copyTo(semi_indic.row(comp));//**
					//...and go on to the next component
					comp++;
				}
				//if killed is true, it means the in the position "comp", we now
				//have a component that was not yet visited in this sweep, and
				//so all we have to do is go back to the M step without
				//increasing "comp"
            }
			// increment the iterations counter
			countf++;

			indic = cv::Mat::zeros(indic.rows, indic.cols, CV_64F);
			semi_indic = cv::Mat::zeros(semi_indic.rows, semi_indic.cols, CV_64F);//**
			for (int i = 0; i < k; i++) {
				multinorm(y, estmu.col(i), estcov[i]).copyTo(semi_indic.row(i));
				cv::Mat((semi_indic.row(i)*estpp.at<double>(0,i))).copyTo(indic.row(i));//**
			}
			if (k != 1) {
				//if the number of surviving components is not just one, we compute
				//the logliklihood from the unnormalized assignment variables
				
				cv::Mat r2;
				reduce(indic, r2, 0, CV_REDUCE_SUM);
				r2 = r2+DBL_MIN;
				cv::log(r2, r2);
				
				loglike.push_back(cv::sum(r2)[0]);
			}
			else
			{
				//if it is just one component, it is even simpler
				cv::Mat l;
				cv::log(DBL_MIN+indic, l);
				loglike.push_back(cv::sum(l)[0]);
			}
			//compute and store the description length and the current number of components
			cv::Mat l;
			cv::log(estpp, l);
			dlength = -loglike[countf] + (nparsover2*cv::sum(l)[0]) + (nparsover2 +.5)*k*log(npoints);
			dl.push_back(dlength);
			kappas.push_back(k);
			
			//compute the change in loglikelihood to check if we should stop
			double deltlike = loglike[countf] - loglike[countf-1];
			if (abs(deltlike/loglike[countf-1]) < th) {
				//if the relative change in loglikelihood is below the threshold, we stop CEM2
				cont=0;
			}
        } //this is the end of the inner loop: "while(cont)"

		//now check if the latest description length is the best
		//if it is, we store its value and the corresponding estimates
		if (dl[countf] < mindl) {
			bestpp = estpp;
			bestmu = estmu;
			bestcov = estcov;
			bestk = k;
			mindl = dl[countf];
		}
		
		//at this point, we may try smaller mixtures by killing the
		//component with the samllest mixing probability and then restarting CEM2,
		//as long as k is not yet at kmin
		if (k > kmin) {
			double minp;
			int indminpC[2];
			cv::minMaxIdx(estpp, &minp, NULL,indminpC);
			int indminp = indminpC[1];
			// what follows is the book-keeping associated with removing one component
			if (indminp==0) {
				estmu = estmu.colRange(1, k-1);
				estcov = std::vector<cv::Mat>(&estcov[1],&estcov[k-1]);
				estpp = estpp.colRange(1, k-1);
			}
			else{
				if (indminp == k-1) {
					estmu = estmu.colRange(0, k-1);
					estcov = std::vector<cv::Mat>(&estcov[0],&estcov[k-1]);
					estpp = estpp.colRange(0, k-1);
				}
				else
				{
					cv::hconcat(estmu.colRange(0, indminp), estmu.colRange(indminp+1, k), estmu);
					std::vector<cv::Mat> newcov(k-1);
					for(int kk = 0; kk < indminp; kk++)
					{
						newcov[kk] = estcov[kk];
					}
					for (int kk  = indminp+1; kk < k; kk++) {
						newcov[kk-1] = estcov [kk];
					}
					estcov = newcov;
					cv::hconcat(estpp.colRange(0, indminp), estpp.colRange(indminp+1, k), estpp);
				}
			}
			k--;
			//we renormalize the mixing probabilities after killing the component
			estpp = estpp/cv::sum(estpp)[0];
			//and register the fact that we have forced one component to zero
			transitions2.push_back(countf);
			
			//increment the iterations counter
			countf++;
			
			//...and compute the loglikelihood function and the description length
			if (innercount == 182) {
				NSLog(@"bad");
			}
			indic = cv::Mat::zeros(k, indic.cols, CV_64F);
			semi_indic = cv::Mat::zeros(k, semi_indic.cols, CV_64F);//**
			printMatrix(indic);
			printMatrix(semi_indic);
			for (int i = 0; i < k; i++) {
				multinorm(y, estmu.col(i), estcov[i]).copyTo(semi_indic.row(i));//**
				cv::Mat(semi_indic.row(i)*estpp.at<double>(0,i)).copyTo(indic.row(i));
			}
			cv::Mat r;
			if (k != 1) {
				cv::reduce(indic, r, 0, CV_REDUCE_SUM);
				cv::log(DBL_MIN+r, r);
				
				loglike.push_back(cv::sum(r)[0]);
			}
			else{
				cv::log(DBL_MIN+indic,r);
				loglike.push_back(cv::sum(r)[0]);
			}
			cv::log(estpp, r);
			dl.push_back(-loglike[countf] + (nparsover2*cv::sum(r)[0]) + (nparsover2+.5)*k*log(npoints));
			kappas.push_back(k);
			
		}
		//this else corresponds to "if k > kmin"
		//of course, if k is not larger than kmin, we must stop
		else{
			k_cont = 0;
		}
    }
	clusterOutput output;
	output.bestk = bestk;
	output.bestpp = bestpp;
	output.bestmu = bestmu;
	output.bestcov = bestcov;
	output.dl = dl;
	output.countf = countf;
	return output;
}

double maxd(double d1, double d2)
{
	if (d1 >= d2) {
		return d1;
	}
	return d2;
}
cv::Mat kron3(cv::Mat A, cv::Mat B)
{
	cv::Mat Atmp,output;
	cv::repeat(B, A.rows, A.cols, output);
	cv::resize(A, Atmp, cv::Size(output.cols,output.rows),0,0,cv::INTER_NEAREST);
	cv::multiply(Atmp, output, output);
	return output;
}
cv::Mat kron2(cv::Mat A, cv::Mat B)
{
	int newCols = A.cols*B.cols;
	int newRows = A.rows*B.rows;
	cv::Mat output = cv::Mat::zeros(newRows, newCols, CV_64FC1);
	for (int col = 0; col < output.cols; col++) {
		for(int row = 0; row < output.rows; row++)
		{
			int aIndCol = col/A.cols;
			int aIndRow = row/A.rows;
			int bIndCol = col%B.cols;
			int bIndRow = row%B.rows;
			output.at<double>(row,col) = A.at<double>(aIndRow,aIndCol)*B.at<double>(bIndRow,bIndCol);
		}
	}
	return output;
}
cv::Mat kron(cv::Mat A, cv::Mat B)
{
	if ( ! A.isContinuous() )
	{
		A = A.clone();
	}
	if ( ! B.isContinuous() )
	{
		B = B.clone();
	}
    int I = A.rows;
    int J = A.cols;
    int K = B.rows;
    int L = B.cols;
    
    //Only implemented both matrices as dense
    int dim1[4] ={1,I,1,J};
    int dim2[4] ={K,1,L,1};
    int dim3[2] = {I*K,J*L};
    A = A.reshape(0, 4, dim1);
    B = B.reshape(0,4,dim2);
    cv::Mat x  = (A*B);
    x = x.reshape(0,2,dim3);
    return x;
    
}
cv::Mat multinorm(cv::Mat x, cv::Mat m, cv::Mat covar)
{
    cv::Mat y;
    int dim = x.rows;
    int npoints = x.cols;
    cv::Mat cvar2 = covar +DBL_MIN*cv::Mat::eye(dim, dim, CV_64FC1);
    double dd = cv::determinant(cvar2);
//	printMatrix(covar);
    cv::Mat inv;
    cv::invert(cvar2, inv);
    double ff = pow(2*M_PI,-dim/2)*pow(dd, -.5);
    cv::Mat quadform = cv::Mat::zeros(1, npoints, CV_64F);
    cv::Mat centered = (x-m*cv::Mat::ones(1, npoints, CV_64F));
    if (dim != 1) {
        cv::Mat exponent,mul;
        cv::multiply(centered, inv*centered, mul);
        cv::reduce(mul, mul, 0, CV_REDUCE_SUM);
        cv::exp(-.5*mul, mul);
        y = ff * mul;
    }
    else{
        cv::Mat mul;
        cv::pow(centered, 2, mul);
        exp(-.5*inv*mul,mul);
        y = ff * mul;
    }
    return y;
}

std::vector<int> randperm(int size)
{
	std::vector<int>returnVect(size);
	for (int i = 0; i < size; i++) {
		returnVect[i] = i;
	}
//	std::srand((unsigned int)std::time(0));
	std::random_shuffle(returnVect.begin(), returnVect.end());
	return returnVect;
}

cv::Mat cov(cv::Mat matrix) {
	// Input matrix size
	
	const int rows = matrix.rows;
	const int cols = matrix.cols;
	// Place input into CvMat**
	CvMat** input = new CvMat*[rows];
	for(int i=0; i<rows; i++) {
		input[i] = cvCreateMat(1, cols, CV_64FC1);
		for(int j=0; j<cols; j++) {
			cvmSet(input[i], 0, j, matrix.at<double>(i,j));
		}
	}
	
	// Covariance matrix is N x N,
	// where N is input matrix column size
	const int n = cols;
	
	// Output variables passed by reference
	CvMat* output = cvCreateMat(n, n, CV_64FC1);
	CvMat* meanvec = cvCreateMat(1, rows, CV_64FC1);
	
	// Calculate covariance matrix
	cvCalcCovarMatrix((const void **) input, \
					  rows, output, meanvec, CV_COVAR_NORMAL);
	
	//Show result
	cv::Mat returnMat(output);
	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			// normalize by n - 1 so that results are the same
			// as MATLAB's cov() and Mathematica's Covariance[]
			returnMat.at<double>(i,j) =cvGetReal2D(output,i,j) / (rows - 1);
		}
	}
	return returnMat;
}
cv::Mat getDiag(cv::Mat matrix)
{
	if (matrix.cols == matrix.rows && matrix.type() == CV_64FC1) {
		cv::Mat output = cv::Mat::zeros(matrix.rows, 1, CV_64FC1);
		for (int i = 0; i < matrix.rows; i++) {
			output.at<double>(i, 0) = matrix.at<double>(i, i);
		}
		return output;
	}
	else if(matrix.cols == 1 && matrix.rows > 1)
	{
		cv::Mat output = cv::Mat::zeros(matrix.rows, matrix.rows, CV_64FC1);
		for (int i = 0; i < matrix.rows; i++) {
			output.at<double>(i,i) = matrix.at<double>(i,0);
		}
		return output;
	}
	else if(matrix.cols > 1 && matrix.rows == 1)
	{
		cv::Mat output = cv::Mat::zeros(matrix.cols, matrix.cols, CV_64FC1);
		for (int i = 0; i < matrix.cols; i++) {
			output.at<double>(i,i) = matrix.at<double>(0,i);
		}
		return output;
	}
	else{
		NSLog(@"ERROR: diag on incorrect mat type");
		return cv::Mat();
	}
}
void printMatrix(cv::Mat matrix)
{
	cout << "M = "<< endl << " "  << matrix << endl << endl;
}
@end
