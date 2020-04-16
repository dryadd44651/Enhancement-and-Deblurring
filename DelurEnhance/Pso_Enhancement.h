#include "pso.h"

Process p;

//========================  Pso Enhancement
Mat Run(Mat &img, int method, int objective);
Mat Min(Mat src, Size s = Size(3, 3));
Mat Max(Mat src, Size s = Size(3, 3));
void set_parameter(pso &pso, double Gmean);
Mat enhance(Mat &input, Mat &dev, Mat &Lmean, double Gmean, vector<float> particle);
double Evaluation(Mat &input);



//========================  Pso Enhancement
double Evaluation(Mat &input) {
	Mat thres, sobelx, sobely, sobel;
	int input_size = input.rows*input.cols;
	vector<int> h;

	double term1 = 0, term2 = 0, term3 = 0, output = 0;
	Sobel(input, sobelx, -1, 1, 0, 3);
	Sobel(input, sobely, -1, 0, 1, 3);
	sobel = sobelx.mul(sobelx) + sobely.mul(sobely);
	pow(sobel, 0.5, sobel);

	term1 = p.Sum(sobel);

	term1 = (term1 == 0) ? 0 : log10(log10(term1));

	threshold(sobel, thres, 127, 1, THRESH_BINARY);//threshold img to 0/1

	term2 = p.Sum(thres);//n_edngels()/M*N	

	h = p.histogram(input, 256);//histogram


	for (int i = 0; i < 255; i++)
		term3 -= h[i] * p.log((double)h[i] / input_size, 2);//e = h*log(h)  sum(-e)    
	term3 /= input_size;


	output = term1 * term2*term3;
	return output;

}

Mat enhance(Mat &input, Mat &dev, Mat &Lmean, double Gmean, vector<float> particle) {//k a b c [0 1 2 3]


	Mat output(input.size(), input.type());




	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++) {


			output.at<float>(y, x) = (particle[0] * Gmean / (dev.at<float>(y, x) + particle[2]))*// k*dev/(D+b) 
				(input.at<float>(y, x) - particle[3] * Lmean.at<float>(y, x))//*f-c*m
				+ pow(input.at<float>(y, x), particle[1]);//+particle[3];//+f^a

		}

	waitKey(1);
	p.ImgNormalization(output);
	p.ShowImg32F((char *)"output", output);
	return output;
	//return p.Img8U2Img32F(p.Img32F2Img8U(output));
}

void set_parameter(pso &pso, double Gmean) {

	pso.range.push_back(1);// 1 3 m 1
	pso.range.push_back(3);
	pso.range.push_back(Gmean * 1000);//Gmean*1000
	pso.range.push_back(1);

	pso.scale.push_back(1);// 1 0.5 0.5 1
	pso.scale.push_back(0.5);
	pso.scale.push_back(0.5*0.001);//0.5*0.001
	pso.scale.push_back(1);

	pso.shift.push_back(0.5);// 0.5 0 1 0
	pso.shift.push_back(0);
	pso.shift.push_back(1);//1
	pso.shift.push_back(0.5);


}

Mat Max(Mat src, Size s)
{
	cv::Mat dst;

	cv::Mat kernel;   // Use the default structuring element (kernel) for erode and dilate

					  // Perform max filtering on image using dilate
	cv::dilate(src, dst, kernel);

	return dst;
}

Mat Min(Mat src, Size s)
{
	cv::Mat dst;

	cv::Mat kernel;   // Use the default structuring element (kernel) for erode and dilate

	cv::Mat minImage;
	// Perform min filtering on image using erode
	cv::erode(src, dst, kernel);



	return dst;
}

Mat Run(Mat &img, int method, int objective) {
	pso pso;
	Mat dev = p.Sigma(img);
	Mat Lmean;
	blur(img, Lmean, Size(3, 3));
	Mat max, min, rst;

	max = Max(img);
	min = Min(img);
	rst = max - min;
	Mat output, tmp;
	//p.ShowImg64F("Max", max);
	//waitKey(0);

	double Gmean = cv::mean(img)[0];


	set_parameter(pso, Gmean);
	pso.init_param();

	for (int t = 0; t < pso.generation; t++) {
		cout << endl << t << " th" << endl;
		for (int i = 0; i < pso.number; i++) {
			tmp = enhance(img, dev, Lmean, Gmean, pso.particles[i]);
			pso.fitness[i] = Evaluation(tmp);//***
			cout << i << " output evaluation Criterion : " << pso.fitness[i] << endl;
			tmp.release();
		}
		pso.update();
		output = enhance(img, dev, Lmean, Gmean, pso.gbest);
		cout << "output evaluation Criterion : " << Evaluation(output) << endl;//***
		for (int i = 0; i < pso.gbest.size(); i++) {
			cout << "gbest " << i << " : " << pso.gbest[i] << endl;
		}
		p.ShowImg32F((char *)"current output", output);

		waitKey(1);

		cout << pso.gbest_fitness << endl;
	}

	return output;

}
//========================  Pso Enhancement