#include "api.h"
Process pp;

void Rearrang(Mat &mag){
	int cx = mag.cols/2;
	int cy = mag.rows/2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


Mat Expand_Kernel(Mat &kernel,Size s){
	Mat K(s,kernel.type(),Scalar(0));

	Mat roi(K,Rect((s.width/2)-(kernel.cols/2),(s.height/2)-(kernel.rows/2),kernel.cols,kernel.rows));
	kernel.copyTo(roi);

	Rearrang(K);


	return K;
}


Mat Expand_Mat(Mat &A){


	Mat tempA(2*A.rows,2*A.cols, A.type(), Scalar::all(0)),temp;

	temp = A.clone();
	temp.copyTo(tempA(Rect(0, 0, A.cols, A.rows)));
	flip(A,temp,0);
	temp.copyTo(tempA(Rect(0, A.rows, A.cols, A.rows)));
	flip(A,temp,1);
	temp.copyTo(tempA(Rect(A.cols, 0, A.cols, A.rows)));
	flip(A,temp,-1);
	temp.copyTo(tempA(Rect(A.cols, A.rows, A.cols, A.rows)));


	return tempA.clone();
}


Mat L0_Gradient(Mat &X,double lambda,double beta){
	Mat g = X.clone();

	
	for(int y = 0;y<X.rows;y++)
		for(int x = 0;x<X.cols;x++){
			if(abs(g.at<double>(y,x))<(lambda*beta))
				g.at<double>(y,x) = 0;
			else
				g.at<double>(y,x) = (abs(g.at<double>(y,x))-(lambda*beta))*(abs(g.at<double>(y,x))/g.at<double>(y,x));//
		}

		return g;
}

void computeDenominator(Mat &y,Mat &k,Mat &Nomin1,Mat &Denom1,Mat &Denom2){

	//% computes denominator and part of the numerator for Equation (3) of the
	//% paper
	//%
	//% Inputs: 
	//%  y: blurry and noisy input
	//%  k: convolution kernel  
	//% 
	//% Outputs:
	//%      Nomin1  -- F(K)'*F(y)
	//%      Denom1  -- |F(K)|.^2
	//%      Denom2  -- |F(D^1)|.^2 + |F(D^2)|.^2
	Mat fy = y.clone(),fgv,fgh;
	Mat fk = Expand_Kernel(k,fy.size());

	Mat Gv(1,3,CV_64F,Scalar(0)),Gh;
	Gv.at<double>(0,1) = 1;
	Gv.at<double>(0,2) = -1;
	Gh = Gv.t();

	fgv = Expand_Kernel(Gv,fy.size());
	fgh = Expand_Kernel(Gh,fy.size());

	dft(fy,fy,DFT_COMPLEX_OUTPUT);
	dft(fk,fk,DFT_COMPLEX_OUTPUT);
	dft(fgv,fgv,DFT_COMPLEX_OUTPUT);
	dft(fgh,fgh,DFT_COMPLEX_OUTPUT);

	Nomin1.create(fy.size(),fy.type());
	Denom1.create(fy.size(),fy.type());
	Denom2.create(fy.size(),fy.type());

	for(int y = 0;y<fk.rows;y++)
		for(int x = 0;x<fk.cols;x++){

			Nomin1.at<complex<double>>(y,x) = conj(fk.at<complex<double>>(y,x))*fy.at<complex<double>>(y,x);

			Denom1.at<complex<double>>(y,x) = abs(fk.at<complex<double>>(y,x));

			Denom1.at<complex<double>>(y,x) = pow(Denom1.at<complex<double>>(y,x),2.);

			Denom2.at<complex<double>>(y,x) = conj(fgv.at<complex<double>>(y,x))*fgv.at<complex<double>>(y,x)+
				conj(fgh.at<complex<double>>(y,x))*fgh.at<complex<double>>(y,x);

		}
}

Mat computeResult(Mat &Nomin1,Mat &Nomin2,Mat &Denom1,Mat &Denom2,double gamma){
	Mat output(Nomin1.size(),Nomin2.type());
	complex<double> weight = 0;
	weight._Val[0] = gamma;

	for(int y = 0;y<output.rows;y++)
		for(int x = 0;x<output.cols;x++){
			output.at<complex<double>>(y,x) = Nomin1.at<complex<double>>(y,x)+weight*Nomin2.at<complex<double>>(y,x);
			output.at<complex<double>>(y,x) /= Denom1.at<complex<double>>(y,x)+weight*Denom2.at<complex<double>>(y,x);
		}
		return output;
}
Mat Algorithm1(Mat &B,Mat &k,double lambda){
	double beta = 1/lambda,beta_min = 0.001,gamma;
	Mat I,Ix,Iy,Wx,Wy,Wxx,Wyy,Nomin1,Nomin2,Denom1,Denom2,fout;
	Mat Gv(1,3,CV_64F,Scalar(0)),Gh;//Gradient operation
	Gv.at<double>(0,1) = -1;
	Gv.at<double>(0,2) = 1;
	Gh = Gv.t();
	Mat Fgv,Fgh;
	flip(Gv,Fgv,1);
	flip(Gh,Fgh,0);

	I = Expand_Mat(B);
	I /=255;

	filter2D(I,Ix,-1,Gv,Point(-1,-1),0,BORDER_REFLECT );
	filter2D(I,Iy,-1,Gh,Point(-1,-1),0,BORDER_REFLECT );

	computeDenominator(I,k,Nomin1,Denom1,Denom2);
	//p.ShowImg64F("Ix",Ix);

	while(beta>beta_min){
		//imshow("Ix",Ix);

		gamma = 1/(2*beta);
		Wx = L0_Gradient(Ix,lambda,beta);
		Wy = L0_Gradient(Iy,lambda,beta);
		//imshow("Wx",Wx);
		filter2D(Wx,Wxx,-1,Fgv,Point(-1,-1),0,BORDER_REFLECT );
		filter2D(Wy,Wyy,-1,Fgh,Point(-1,-1),0,BORDER_REFLECT );
		
		Nomin2 = Wxx+Wyy;
		//imshow("nomin2",Nomin2);
		
		dft(Nomin2,Nomin2,DFT_COMPLEX_OUTPUT);
		
		fout = computeResult(Nomin1,Nomin2,Denom1,Denom2,gamma);

		dft(fout,I,DFT_INVERSE + DFT_REAL_OUTPUT+DFT_SCALE);

		//imshow("I",I);
		I = I.colRange(0,I.cols/2).rowRange(0,I.rows/2).clone();
		I = Expand_Mat(I);
		//cout<<"1"<<endl;
		//waitKey(0);
		filter2D(I,Ix,-1,Gv,Point(-1,-1),0,BORDER_REFLECT);
		filter2D(I,Iy,-1,Gh,Point(-1,-1),0,BORDER_REFLECT);
		beta /= 2;
		//waitKey(5);
	}
	I = I.colRange(0,I.cols/2).rowRange(0,I.rows/2).clone();
	I *= 255;
	//Mat I0;
	//pow(Wxx+Wyy,1,I0);
	//cout<<"norm0"<<pp.Sum_64F(I0)<<endl;
	//imshow("norm0",I0);
	return I;
}
