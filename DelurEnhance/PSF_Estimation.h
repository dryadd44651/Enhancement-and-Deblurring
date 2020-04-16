

double Max_64F(Mat &input){
	double max = input.at<double>(0,0);
	for(int y=0;y<input.rows;y++)
		for(int x=0;x<input.cols;x++){
			if(input.at<double>(y,x)>max)
				max = input.at<double>(y,x);
		}
		return max;
}

double Mean_64F(Mat &input){
	Mat one_h(input.cols,1,input.type(),Scalar(1.));
	Mat one_w(1,input.rows,input.type(),Scalar(1.));
	Mat tmp =(one_w*input)/one_w.cols;

	tmp *=one_h;
	tmp /=one_h.rows;
	return tmp.at<double>(0,0);
}

double Sum_64F(Mat &input){
	Mat one_h(input.cols,1,input.type(),Scalar(1.));
	Mat one_w(1,input.rows,input.type(),Scalar(1.));
	Mat tmp =(one_w*input*one_h);
	return tmp.at<double>(0,0);
}
//====================================================================
double Min_Max_Mean_64F(Mat &input){//painting like
	double mean = Mean_64F(input),max_mean = 0,min_mean = 0;
	double f = input.at<double>(input.rows/2,input.cols/2);
	double max_number = 0,min_number = 0;
	bool c = (f>mean)?1:0;


	if(f==mean)
		return mean;

	for(int y=0;y<input.rows;y++)
		for(int x=0;x<input.cols;x++){
			if(c){
				if(input.at<double>(y,x)>mean){
					max_mean+=input.at<double>(y,x);
					max_number++;
				}
			}//if(c)
			else{
				if(input.at<double>(y,x)<mean){
					min_mean+=input.at<double>(y,x);
					min_number++;
				}
			}


		}

		if(c){
			if(max_number<2)
				return f;
			return max_mean/max_number;//max_mean/max_number
		}
		else{
			if(min_number<2)
				return f;
			return min_mean/min_number;//min_mean/min_number
		}

}

Point Bound(Point p,Size s){
	p.y = abs(p.y);
	p.x = abs(p.x);

	p.y = (p.y<s.height) ?p.y:s.height+(s.height-p.y)-2;
	p.x = (p.x<s.width)  ?p.x:s.width+(s.width-p.x)-2;
	
	return p;
}

Mat GetPatch(Mat &input,Point p,Size s){
	Mat output(s,input.type());
	Point tmp;
	for(int y1=0,      y = p.y-(s.height/2);  y<p.y+((s.height+1)/2); y1++,    y++)
		for(int x1=0,  x = p.x-(s.width/2);   x<p.x+((s.width+1)/2);  x1++,      x++){
			tmp.x = x,tmp.y = y;
			tmp = Bound(tmp,input.size());
			//cout<<y<<"  "<<x<<endl;
			output.at<double>(y1,x1) = input.at<double>(tmp.y,tmp.x);
		}
		return output;

}


Mat HL_Filter(Mat &input,Size s= Size(3,3)){

	Mat output(input.size(),input.type(),Scalar(0)),tmp;

	for(int y=0;y<input.rows;y++)
		for(int x=0;x<input.cols;x++){
			tmp = GetPatch(input,Point(x,y),s);
			output.at<double>(y,x) = Min_Max_Mean_64F(tmp);
		}
	return output;
}

//=====================================
Mat Kernel_Estimating(Mat &input,Mat &con,Size S = Size(5,5)){//input*kernel = con
	int shift = S.height/2;
	Mat Kernel(S,input.type(),Scalar(0));
	Mat X((con.rows-Kernel.rows)*(con.cols-Kernel.cols),Kernel.rows*Kernel.cols,input.type(),Scalar(0)),
		K(X.cols,1,input.type()),Y(X.rows,1,K.type()),tmp,row,I;
	I = Mat::eye(X.cols,X.cols,X.type());
	I = 0*I;
	double mean = Mean_64F(con);
	int i =0;

	for(int x = shift;x<con.cols-Kernel.cols-shift;x++)
		for(int y = shift;y<con.rows-Kernel.rows-shift;y++){

			tmp = X.rowRange(i,i+1);
			row = input(Rect(x, y, Kernel.cols,Kernel.rows)).clone();
			row = row.reshape(0,1);
			row.copyTo(tmp);
			Y.at<double>(i,0) = con.at<double>(y+Kernel.rows/2,x+Kernel.cols/2);
			i++;

		}
		X = X.rowRange(0,i);
		Y = Y.rowRange(0,i);
		K = (X.t()*X+I.t()*I).inv()*X.t()*Y;
		K = K.reshape(0,Kernel.cols);
		Kernel = K.clone();
		return Kernel;
}

Mat Edge_Estimating(Mat &input){

	Mat output = HL_Filter(input,Size(3,3));

	for(int c = 0;c<2;c++){

		output = HL_Filter(output,Size(3,3));

	}

	return output;
}

void Kernel_Refine(Mat &kernel){
	double max;
	Mat f(3,3,CV_64F,Scalar(1)),iso;
	f.at<double>(1,1) = 0;
	f /=Sum_64F(f);

	for(int i =0;i<2;i++){
		max = Max_64F(kernel)/20;
		for(int y = 0;y<kernel.rows;y++)
			for(int x = 0;x<kernel.cols;x++){
				if(abs(kernel.at<double>(y,x))<max)
					kernel.at<double>(y,x) = 0;

			}


			filter2D(kernel,iso,-1,f,Point(-1,-1),0,BORDER_CONSTANT);
			for(int y = 0;y<kernel.rows;y++)
				for(int x = 0;x<kernel.cols;x++){
					if(iso.at<double>(y,x)==0)
						kernel.at<double>(y,x) = 0;
				}
				kernel /=Sum_64F(kernel);
	}

}