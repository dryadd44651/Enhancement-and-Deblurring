#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

//using namespace cv;
using namespace std;



class pso{

public:
	vector<vector<float>> particles,pbest,velocity,bound;
	vector<float> scale,shift;
	vector<int> range;
	vector<float> gbest,fitness,l_fitness;
	
	int number,digit,time,generation;
	float gbest_fitness,w,w_max,w_min,c1,c2,r1,r2;


	pso(){
		time = 0;
		number = 25;//particles number
		digit = 1000;
		w_max = 1;
		w_min = 0;
		w = w_max;
		generation = 10;//iteration times
		c1 = 2*rand()&digit;
		c1 /=digit;
		c2 = 2*rand()&digit;
		c2 /=digit;
	}
	void init_param(){
		particles.resize(number);
		fitness.resize(number);
		velocity.resize(number);
		float tmp;
		for(int j = 0;j<number;j++){
			for(int i = 0;i<range.size();i++){
				tmp = rand()%(range[i]*digit);//significant digit
				particles[j].push_back(tmp/(float)digit);
				
			}
			for(int i = 0;i<scale.size();i++)
				particles[j][i] *=scale[i];

			for(int i = 0;i<shift.size();i++)
				particles[j][i] +=shift[i];

			velocity[j].resize(particles[0].size());//create space for velocity
		}

		pbest = particles;

		bound.resize(range.size());
		for(int i = 0;i<bound.size();i++)
			bound[i].resize(2);//upper bound lower bound

		for(int i = 0;i<bound.size();i++){
			bound[i][0] = shift[i];
			bound[i][1] = range[i]*scale[i]+shift[i];
		}

	}

	void update(){
		if(time==0){
			l_fitness = fitness;
			gbest = pbest[0];
			gbest_fitness = fitness[0];
			for(int j = 1;j<number;j++){
				if(fitness[j]>gbest_fitness){
					gbest = pbest[j];
					gbest_fitness = fitness[j];
				}
			}//for
		}//if
		else{
			for(int j = 0;j<number;j++){
				if(fitness[j] > l_fitness[j]){
					pbest[j] = particles[j];

					if(fitness[j]>gbest_fitness){
						gbest = pbest[j];
						gbest_fitness = fitness[j];
					}
				}
			}//for
			l_fitness = fitness;
		}//else
	
		w = w_max-((w_max-w_min)/generation)*time;//inertia weight
		r1 = rand()&digit;
		r1 /=digit;
		r2 = rand()&digit;
		r2 /=digit;
		for(int j = 0;j<number;j++){
			for(int i = 0;i<range.size();i++){
				velocity[j][i] = w*velocity[j][i]+r1*c1*(pbest[j][i]-particles[j][i])+r2*c2*(gbest[i]-particles[j][i]);
				particles[j][i] += velocity[j][i];
				if(particles[j][i]<bound[i][0])//lower bound restriction
					particles[j][i] = bound[i][0];
				if(particles[j][i]>bound[i][1])//upper bound restriction
					particles[j][i] = bound[i][1];

			}
		}
		
		
		time++;
	}//update

};