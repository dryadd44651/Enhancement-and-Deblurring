#include "opencv2\opencv.hpp"


using namespace cv;
using namespace std;



class Process{
private:

	

public:
	Process(){}
	void ShowImg32F(char *WindowName,Mat Img32F) //Show影像(32F)
	{
		Mat Img8U(Img32F.rows,Img32F.cols,CV_8U);

		namedWindow(WindowName,0);
		Img32F.convertTo(Img8U,CV_8U);
		imshow(WindowName,Img8U);
		resizeWindow(WindowName, Img32F.cols, Img32F.rows);
		cvWaitKey(1);
	}

	void ShowImg64F(char *WindowName,Mat Img64F) //Show影像(64F)
	{
		Mat Img8U(Img64F.rows,Img64F.cols,CV_8U);

		namedWindow(WindowName,0);
		Img64F.convertTo(Img8U,CV_8U);
		imshow(WindowName,Img8U);
		resizeWindow(WindowName, Img64F.cols, Img64F.rows);
		cvWaitKey(1);
	}

	Mat Img8U2Img32F(Mat Img8U) //Mat影像 8U轉32F
	{
		Mat Img32F(Img8U.rows,Img8U.cols,CV_32F);

		Img8U.convertTo(Img32F,CV_32F);

		return Img32F;
	}

	Mat Img8U2Img64F(Mat Img8U) //Mat影像 8U轉32F
	{
		Mat Img64F(Img8U.rows,Img8U.cols,CV_32F);

		Img8U.convertTo(Img64F,CV_64F);

		return Img64F;
	}

	Mat Img32F2Img8U(Mat Img32) //Mat影像 32F轉8U
	{
		Mat Img8U(Img32.rows,Img32.cols,CV_8U);

		Img32.convertTo(Img8U,CV_8U);

		return Img8U;
	}
	Mat Img64F2Img8U(Mat Img64) //Mat影像 32F轉8U
	{
		Mat Img8U(Img64.rows,Img64.cols,CV_8U);

		Img64.convertTo(Img8U,CV_8U);

		return Img8U;
	}

	Mat Img64F2Img32F(Mat Img64) //Mat影像 32F轉8U
	{
		Mat Img32F(Img64.rows,Img64.cols,CV_32F);

		Img64.convertTo(Img32F,CV_32F);

		return Img32F;
	}

	Mat Img32F2Img64F(Mat Img32) //Mat影像 32F轉8U
	{
		Mat Img64F(Img32.rows,Img32.cols,CV_8U);

		Img32.convertTo(Img64F,CV_64F);

		return Img64F;
	}
	void ImgNormalization(Mat &mat)
	{
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(mat,&minVal,&maxVal,&minLoc,&maxLoc);

		float temp=0;

		for(int i=0;i<mat.rows;i++)
		{
			for(int j=0;j<mat.cols;j++)
			{
				temp=mat.at<float>(i,j);
				mat.at<float>(i,j)=((temp-minVal)*255)/(maxVal-minVal);
			}
		}
	}

	Mat GetResize(Mat img,float scaler,float scaler1,int flag){
		Mat input;
		input = img.clone();
		Mat output(img.rows*(scaler),img.cols*(scaler1),CV_32F);

		switch (flag){
		case 1:
			resize(input,output,output.size(),CV_INTER_CUBIC);
			return output;
			break;
		case 2:
			resize(input,output,output.size(),CV_INTER_LINEAR);
			return output;
			break;
		case 3:
			resize(input,output,output.size(),CV_INTER_NN);
			return output;
			break;
		default:
			resize(input,output,output.size());
			return output;
			break;
		}
	}

	int constrain(int v){
		if(v<0)
			v=0;
		if(v>255)
			v=255;
		return v;
	}

	Mat Dct(Mat img,int M,int N){
		double PI = 3.1415926535;
		// 暫存DCT中的運算值和
		double sum = 0;
		// 計算DCT公式中的兩個參數
		double ai = 0, aj = 0;
		Mat img32f;
		img32f.create(img.rows,img.cols,CV_32F);

		// 開始DCT計算
		for(int y=0;y<img.rows;y+=M)
			for(int x=0;x<img.cols;x+=M){
				for (int i=0; i<M; i++) {
					for (int j=0; j<N; j++) {
						// 每一次將和歸零
						sum = 0;
						// DCT頭兩個參數
						ai = (i==0?1/sqrt(double(M)):sqrt(2.0/M));
						aj = (j==0?1/sqrt(double(N)):sqrt(2.0/N));
						// DCT中sumation的部份，照公式打的
						for (int m=0; m<M; m++) {
							for (int n=0; n<N; n++) {
								sum += img.at<uchar>(y+m,x+n)*
									cos(PI*(2*m+1)*i/(M*2)) * 
									cos(PI*(2*n+1)*j/(N*2));
							}
						}
						// assign DCT的值
						//dct[i][j] = ai*aj*sum;
						img32f.at<float>(y+i,x+j) = ai*aj*sum;
						cout<<img32f.at<float>(y+i,x+j)<<endl;
					}
				}
			}

			return img32f;
	}

	Mat IDct(Mat img32f,int M,int N){

		Mat img;
		img.create(img32f.rows,img32f.cols,CV_8U);
		double PI = 3.1415926535;
		// 暫存DCT中的運算值和
		double sum = 0;
		// 計算DCT公式中的兩個參數
		double ai = 0, aj = 0;

		for(int y=0;y<img.rows;y+=M)
			for(int x=0;x<img.cols;x+=M){
				for (int m=0; m<M; m++) {
					for (int n=0; n<N; n++) {
						// 每一次將和歸零
						sum = 0;
						// DCT頭兩個參數

						// DCT中sumation的部份，照公式打的
						for (int i=0; i<M; i++) {
							for (int j=0; j<N; j++) {
								ai = (i==0?1/sqrt(double(M)):sqrt(2.0/M));
								aj = (j==0?1/sqrt(double(N)):sqrt(2.0/N));

								//sum += ai*aj*dct[y+i][x+j]*
								sum += ai*aj*img32f.at<float>(y+i,x+j)*
									cos(PI*(2*m+1)*i/(M*2)) * 
									cos(PI*(2*n+1)*j/(N*2));
							}
						}
						// assign DCT的值
						//dct[i][j] = ai*aj*sum;
						img.at<uchar>(y+m,x+n) = sum;
					}
				}
			}
			return img;
	}

	Mat Inter_blur(Mat img,float scaler,int flag){
		Mat input;
		input = img.clone();
		Mat tmp(img.cols*(1/scaler),img.rows*(1/scaler),CV_32F);

		switch (flag){
		case 1:
			resize(input,tmp,tmp.size(),CV_INTER_CUBIC);
			resize(tmp,input,input.size(),CV_INTER_CUBIC);
			return input;
			break;
		case 2:
			resize(input,tmp,tmp.size(),CV_INTER_LINEAR);
			resize(tmp,input,input.size(),CV_INTER_LINEAR);
			return input;
			break;
		case 3:
			resize(input,tmp,tmp.size(),CV_INTER_NN);
			resize(tmp,input,input.size(),CV_INTER_NN);
			return input;
			break;
		default:
			resize(input,tmp,tmp.size());
			resize(tmp,input,input.size());
			return input;
			break;
		}

	}

	Mat Error_Bicubic_Synthesis(Mat input,int scaler){
		Mat input32f,blur,error,bi_error,output;
		input.convertTo(input32f,CV_32F);//get 32f input mat
		//blur.create(input.cols,input.rows,CV_32F);
		bi_error.create(input.cols*scaler,input.rows*scaler,CV_32F);
		output.create(input.cols*scaler,input.rows*scaler,CV_32F);

		blur = Inter_blur(input32f,scaler*1,1);//get blurred mat
		error = input32f - blur;//get error mat
		resize(error,bi_error,bi_error.size(),CV_INTER_CUBIC);//get bicubic error mat
		resize(input32f,output,output.size(),CV_INTER_CUBIC);//get bicubic input mat
		output = output + 0.2*bi_error;//plus error value
		return output;
	}

	Mat second_Error_Bicubic_Synthesis(Mat input,int scaler){
		Mat input32f,blur,error,bi_error,output;
		input.convertTo(input32f,CV_32F);//get 32f input mat
		//blur.create(input.cols,input.rows,CV_32F);
		bi_error.create(input.cols*scaler,input.rows*scaler,CV_32F);
		output.create(input.cols*scaler,input.rows*scaler,CV_32F);

		blur = Inter_blur(input32f,scaler*1,1);//get blurred mat
		error = input32f - blur;//get error mat
		//resize(error,bi_error,bi_error.size(),CV_INTER_CUBIC);//get bicubic error mat
		bi_error = Error_Bicubic_Synthesis(error,scaler);
		resize(input32f,output,output.size(),CV_INTER_CUBIC);//get bicubic input mat
		output = output + 0.2*bi_error;//plus error value
		return output;
	}

	double GetPSNR(const Mat& I1, const Mat& I2)
	{	
		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
		
		s1 = s1.mul(s1);           // |I1 - I2|^2
		s1 = Img32F2Img8U(s1);
		Scalar s = sum(s1);        // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if( sse <= 1e-10) // for small values return zero
			return 0;
		else
		{
			double mse  = sse / (double)(I1.channels() * I1.total());
			double psnr = 10.0 * log10((255 * 255) / mse);
			return psnr;
		}
	}
	double GetPSNR_64F(const Mat& I1, const Mat& I2)
	{	
		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1.convertTo(s1, CV_64F);  // cannot make a square on 8 bits
		
		s1 = s1.mul(s1);           // |I1 - I2|^2
		s1 = Img64F2Img8U(s1);
		Scalar s = sum(s1);        // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if( sse <= 1e-10) // for small values return zero
			return 0;
		else
		{
			double mse  = sse / (double)(I1.channels() * I1.total());
			double psnr = 10.0 * log10((255 * 255) / mse);
			return psnr;
		}
	}
	double getMSSIM( const Mat& i1, const Mat& i2)
	{
		const double C1 = 6.5025, C2 = 58.5225;
		/***************************** INITS **********************************/
		int d = CV_32F;

		Mat I1, I2;
		i1.convertTo(I1, d);            // cannot calculate on one byte large values
		i2.convertTo(I2, d);

		Mat I2_2   = I2.mul(I2);        // I2^2
		Mat I1_2   = I1.mul(I1);        // I1^2
		Mat I1_I2  = I1.mul(I2);        // I1 * I2

		/*************************** END INITS **********************************/

		Mat mu1, mu2;                   // PRELIMINARY COMPUTING
		GaussianBlur(I1, mu1, Size(11, 11), 1.5);
		GaussianBlur(I2, mu2, Size(11, 11), 1.5);

		Mat mu1_2   =   mu1.mul(mu1);
		Mat mu2_2   =   mu2.mul(mu2);
		Mat mu1_mu2 =   mu1.mul(mu2);

		Mat sigma1_2, sigma2_2, sigma12;

		GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
		sigma1_2 -= mu1_2;

		GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
		sigma2_2 -= mu2_2;

		GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
		sigma12 -= mu1_mu2;

		///////////////////////////////// FORMULA ////////////////////////////////
		Mat t1, t2, t3;

		t1 = 2 * mu1_mu2 + C1;
		t2 = 2 * sigma12 + C2;
		t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		t1 = mu1_2 + mu2_2 + C1;
		t2 = sigma1_2 + sigma2_2 + C2;
		t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

		Mat ssim_map;
		divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

		Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
		return mssim.val[0];
	}

	Point Bound(Point p,Size s){
		p.y = (p.y>0)?p.y:0;
		p.x = (p.x>0)?p.x:0;

		p.y = (p.y<s.height) ?p.y:s.height-1;
		p.x = (p.x<s.width)  ?p.x:s.width-1;
		return p;
	}

	Mat GetPatch(Mat &input,Point p,Size s){
		Mat output(s,input.type());
		Point tmp;
		if(input.type() == CV_64F){
		for(int y1=0,y=p.y-(s.height/2);y<p.y+((s.height+1)/2);y1++,y++)
			for(int x1=0,x=p.x-(s.width/2);x<p.x+((s.width+1)/2);x1++,x++){
				tmp.x = x,tmp.y = y;
				tmp = Bound(tmp,input.size());
				output.at<double>(y1,x1) = input.at<double>(tmp.y,tmp.x);
			}
			return output;
		}

		for(int y1=0,y=p.y-(s.height/2);y<p.y+((s.height+1)/2);y1++,y++)
			for(int x1=0,x=p.x-(s.width/2);x<p.x+((s.width+1)/2);x1++,x++){
				tmp.x = x,tmp.y = y;
				tmp = Bound(tmp,input.size());
				output.at<float>(y1,x1) = input.at<float>(tmp.y,tmp.x);
			}
			return output;
	}

	vector<Mat> GatPatch_set(Mat &input,int size,int x_step,int y_step,int flag = 0){
		vector<Mat> patch_set;
		Point p = Size(0,0);
		if(flag){
			p = Size(size/2,size/2);
		}
		Mat patch(Size(size,size),input.type());
		for(int y=p.y;y<input.rows;y+=y_step)
			for(int x=p.x;x<input.cols;x+=x_step){
				patch = 0;
				patch = GetPatch(input,Point(x,y),Size(size,size));
				patch_set.push_back(patch.clone());//1958
				
			}
			return patch_set;
	}

	vector<float> bubbleSort(vector<float> arr)
	{
		int i = arr.size(),j;
		float temp;

		while(i > 0)
		{
			for(j = 0; j < i - 1; j++)
			{
				if(arr[j] > arr[j+1])
				{   temp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = temp;
				}
			}
			i--;
		}
		return arr;
	}

	Mat Expression_Mat(Mat input){
		Mat output(input.size(),CV_8UC3);
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
				//cout<<input.at<float>(y,x)<<endl;
				if(input.at<float>(y,x)<0){//negative:blue
					output.at<Vec3b>(y,x)[0] = constrain(-input.at<float>(y,x));
					output.at<Vec3b>(y,x)[1] = 0;
					output.at<Vec3b>(y,x)[2] = 0;
				}
				else if(input.at<float>(y,x)>255){
					output.at<Vec3b>(y,x)[2] = constrain(input.at<float>(y,x)-255);
					output.at<Vec3b>(y,x)[0] = 0;
					output.at<Vec3b>(y,x)[1] = 0;
				}
				else{
					output.at<Vec3b>(y,x)[1] = input.at<float>(y,x);
					output.at<Vec3b>(y,x)[0] = 0;
					output.at<Vec3b>(y,x)[2] = 0;
				}
			}

			return output;
	}

	//============================zero mean================================
	Mat zero_mean (Mat input,double &mean){

		//Mat patch(patch.size(),CV_32F);
		Mat patch = input.clone();

		int sum_pixel=0;
		mean = 0;
		for(int x=0;x<patch.cols;x++){
			for(int y=0;y<patch.rows;y++){
				sum_pixel += patch.at<float>(y,x);
			}
		}
		mean=sum_pixel/(patch.cols*patch.rows);
		patch-=mean;
		return patch;
		//要把每個patch的平均記下來
	}

	Mat zero_mean (Mat input){

		//Mat patch(patch.size(),CV_32F);
		Mat patch = input.clone();

		int sum_pixel=0;
		double mean = 0;
		for(int x=0;x<patch.cols;x++){
			for(int y=0;y<patch.rows;y++){
				sum_pixel += patch.at<float>(y,x);
			}
		}
		mean=sum_pixel/(patch.cols*patch.rows);
		patch-=mean;
		return patch;
		//要把每個patch的平均記下來
	}
	//=========================Euclidean distance=========================
	float Euclidean_distance (Mat img, Mat img2){
		Mat diff = img-img2;
		float sum_pixel = 0;
		for(int x=0;x<diff.cols;x++){
			for(int y=0;y<diff.rows;y++){
				sum_pixel += pow(diff.at<float>(y,x),(float)2.0);
			}
		}
		//ssum_pixel/=(diff.cols*diff.rows);
		return sqrt(sum_pixel);
	}

	//沒有overlap的貼回去
	void Paste_Patch(Mat &input,Mat patch,Point p){
		for(int y=0;y<patch.rows;y++)
			for(int x=0;x<patch.cols;x++){
				if(p.y+y<input.rows&&p.y+y>0&&p.x+x<input.cols&&p.x+x>0)
				input.at<float>(p.y+y,p.x+x) = patch.at<float>(y,x);
			}
	}

	Mat Paste_Patchs(Size s,vector<Mat> patchs){
		Mat output(s,patchs[0].type());
		int counter = 0;
		for(int y=0;y<=s.height-patchs[0].rows;y+=patchs[0].rows)
			for(int x=0;x<=s.width-patchs[0].cols;x+=patchs[0].cols,counter++){
				Paste_Patch(output,patchs[counter],Point(x,y));
				ShowImg32F((char*)"out",output);
			}
			return output;
	}

	Mat Change_Mean(Mat input,double mean){
		double tmp;
		Mat output = zero_mean(input,tmp);
		output=output+mean;
		return output;
	}

	Mat Mean_mat(vector<Mat> &input){//mean mat of vector
		Mat mean = input[0].clone();
		for(int i=1;i<input.size();i++){
			mean+=input[i];
		}
		return mean/input.size();
	}

	double Mean(Mat &input){
		Mat one_h(input.cols,1,input.type(),Scalar(1.));
		Mat one_w(1,input.rows,input.type(),Scalar(1.));
		Mat tmp =(one_w*input)/one_w.cols;

		tmp *=one_h;
		tmp /=one_h.rows;
		return tmp.at<float>(0,0);
	}
	double Mean_64F(Mat &input){
		Mat one_h(input.cols,1,input.type(),Scalar(1.));
		Mat one_w(1,input.rows,input.type(),Scalar(1.));
		Mat tmp =(one_w*input)/one_w.cols;

		tmp *=one_h;
		tmp /=one_h.rows;
		return tmp.at<double>(0,0);
	}
	double Sum(Mat &input){
		Mat one_h(input.cols,1,input.type(),Scalar(1.));
		Mat one_w(1,input.rows,input.type(),Scalar(1.));
		Mat tmp =(one_w*input*one_h);
		return tmp.at<float>(0,0);
	}
	
	double Sum_64F(Mat &input){
		Mat one_h(input.cols,1,input.type(),Scalar(1.));
		Mat one_w(1,input.rows,input.type(),Scalar(1.));
		Mat tmp =(one_w*input*one_h);
		return tmp.at<double>(0,0);
	}

	double log(double input,double base){
		return (input==0)?1:std::log(input)/std::log(base);
	}

	float Min(Mat &input){
		double min = input.at<float>(0,0);
		
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
			if(input.at<float>(y,x)<min)
				min = input.at<float>(y,x);
			}
		return min;
	}
	double Max(Mat &input){
		double max = input.at<float>(0,0);
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
			if(input.at<float>(y,x)>max)
				max = input.at<float>(y,x);
			}
		return max;
	}
	double Max_64F(Mat &input){
		double max = input.at<double>(0,0);
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
			if(input.at<double>(y,x)>max)
				max = input.at<double>(y,x);
			}
		return max;
	}
	Point loc(Mat &input,float val){
		
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
				if(input.at<float>(y,x)==val)
					return Point(x,y);
			}
			
	}
	vector<int> histogram(Mat &input,int range){
		Mat tmp = Img32F2Img8U(input);
		vector<int> output(range);
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
				output[tmp.at<uchar>(y,x)]++;
			}
			return output;
	}

	Mat Histogram(Mat &input,int range){
		Mat tmp = Img32F2Img8U(input);
		Mat output(1,range,CV_32F);
		output = 0;
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
				output.at<float>(0,tmp.at<uchar>(y,x))++;
			}
			return output;
	}

	Mat Show_Hist(Mat &hist){
		int max = 0;
		int range = hist.cols;


		for(int i = 0;i<range;i++){
			if(hist.at<float>(0,i)>max)
				max = hist.at<float>(0,i);
		}
		if(max==0){
			return hist;
		}
		Mat output(max,range,CV_32F);
		output = 0;


		for(int i = 0;i<range;i++)
			for(int y = 0;y<max;y++)
				if(hist.at<float>(0,i)>y)
					output.at<float>((max-1)-y,i) = 255;
				else
					output.at<float>((max-1)-y,i) = 0;
		
		return GetResize(output,float(range)/max,1,0);
	}

	Mat Show_Hist(vector<int> &hist){
		int max = 0;
		int range = hist.size();
		for(int i = 0;i<range;i++)
			if(hist[i]>max)
				max = hist[i];

		Mat output(max,range,CV_32F);
		for(int i = 0;i<range;i++)
			for(int y = 0;y<max;y++)
				if(hist[i]>y)
					output.at<float>((max-1)-y,i) = 255;
				else
					output.at<float>((max-1)-y,i) = 0;

		return GetResize(output,float(range)/max,1,0);
	}

	Mat Hard_limit(Mat &input){
		Mat output(input.size(),input.type());
		for(int y=0;y<input.rows;y++)
			for(int x=0;x<input.cols;x++){
				if(input.at<float>(y,x)>0)
					output.at<float>(y,x) = 1;
				else if(input.at<float>(y,x)<0)
					output.at<float>(y,x) = -1;
				else
					output.at<float>(y,x) = 0;

			}
			return output;

	}



void DFT(Mat &img,Mat &real,Mat &imag,float M,float N){
	float PI = 3.1415926535;
	complex<float> sum = 0;
	real.create(img.rows,img.cols,CV_32F);
	imag.create(img.rows,img.cols,CV_32F);
	real = 0;
	imag = 0;
	complex<float> t(0,2*PI),tmp;
	Mat roi_img,roi_real,roi_imag;

	//Real(z) = exp(x1) * cos(y1)

	//Img(z) = exp(x1) * sin(y1)
	for(float i=0;i<img.rows;i+=N)
		for(float j=0;j<img.cols;j+=M){
			roi_img = img.rowRange(i,i+N).colRange(j,j+M);
			roi_real = real.rowRange(i,i+N).colRange(j,j+M);
			roi_imag = imag.rowRange(i,i+N).colRange(j,j+M);

			for(float v=0;v<N;v++)
				for(float u=0;u<M;u++){

					//sum = 0;
					for (float y=0; y<N; y++)//patch
						for (float x=0; x<M; x++)	{
							tmp = 0;
							tmp = roi_img.at<float>(y,x)*exp(-t*(u*x/M+v*y/N));
							
							roi_real.at<float>(v,u)+=tmp._Val[0];
							roi_imag.at<float>(v,u)+=tmp._Val[1];
						}
						

						
				}
		}




}

Mat IDFT(Mat &real,Mat &imag,float M,float N){
	float PI = 3.1415926535;
	complex<float> sum = 0;
	Mat output(real.rows,real.cols,CV_32F);
	
	output = 0;
	complex<float> t(0,2*PI),tmp;

	Mat roi_output,roi_real,roi_imag;

	for(float i=0;i<output.rows;i+=N)
		for(float j=0;j<output.cols;j+=M){
			roi_output = output.rowRange(i,i+N).colRange(j,j+M);
			roi_real = real.rowRange(i,i+N).colRange(j,j+M);
			roi_imag = imag.rowRange(i,i+N).colRange(j,j+M);

			for(float v=0;v<N;v++)
				for(float u=0;u<M;u++){
					tmp = 0;
					sum = 0;
					for (float y=0; y<N; y++)
						for (float x=0; x<M; x++){
							tmp = 0;		
							tmp._Val[0] = roi_real.at<float>(y,x);
							tmp._Val[1] = roi_imag.at<float>(y,x);
							sum += tmp*exp(t*(u*x/M+v*y/N));
							//tmp = tmp*exp(t*(u*x/M+v*y/N));
							
							roi_output.at<float>(v,u) += tmp.real();

						//cout<<tmp*exp(t*(u*x/M+v*y/N))<<endl;
						//waitKey(0);
						}
						
						roi_output.at<float>(v,u) = sum.real();
						roi_output.at<float>(v,u) /= N*M;
						//cout<<sum<<endl;

				}
		}
		
		return output;
}



	Mat Sigma(Mat &input,Size s = Size(3,3)){
		Mat tmp,out_dev;
		tmp = input.mul(input);
		blur(tmp,tmp, s);
		blur(input, out_dev, s);
		out_dev = out_dev.mul(out_dev);
		out_dev = tmp - out_dev;
		//sqrt(out_dev,out_dev);
		return out_dev;
	}

Mat Lucy_deconvolution(Mat &input,Mat &mask,int times){//Richardson–Lucy deconvolution
Mat deblur = input.clone(),tmp,fmask,s0,s1;
	flip(mask,fmask,-1);
	Laplacian(input,s0,-1,3);
	s0 /= Sum(s0)/255;
	
for(int i = 0 ;i<times;i++){
		
		
		
		GaussianBlur(deblur,tmp,Size(5,5),2.);
		tmp = (input/tmp);
		
		filter2D(tmp,tmp,-1,fmask);
		
		deblur = deblur.mul(tmp);
		
		
		Laplacian(deblur,s1,-1,3);
		s1 /= Sum(s1)/255;
		s1 = s1.mul(s0);
		//cout<<GetPSNR(g,deblur)<<" "<<Mean(s1)<<endl;
		//ShowImg32F("1",deblur);
		//waitKey(0);
		if(Mean(s1)<0)
			return deblur;
	}

return deblur;
}

};

