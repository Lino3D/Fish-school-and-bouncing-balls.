#include <windows.h>  // for MS Windows
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
#include <Math.h>     // Needed for sin, cos
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define PI 3.14159265f
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Math.h>// Needed for sin, cos
#include <vector>
#include <time.h>
#include <ctime>
using std::vector;
#define PI 3.14159265f
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Global variables
char title[] = "Flocks";  // Windowed mode's title
int windowWidth = 640;     // Windowed mode's width
int windowHeight = 480;     // Windowed mode's height
int windowPooldVx = 50;      // Windowed mode's top-left corner x
int windowPooldVy = 50;      // Windowed mode's top-left corner y

float ballRadius = 0.01f;
float XMax, XMin, YMax, YMin; // Ball's center (x, y) bounds
GLfloat xSpeed = 0.02f;      // Ball's speed in x and y directions
GLfloat ySpeed = 0.007f;
int refreshMillis = 10;      // Refresh period in milliseconds
float Angle = atan2(-xSpeed, ySpeed) * 180 / PI;





const int size = 1000;





static double mytime = 0;

static float fpsCurrent=0;
static float fpsTotal=0;
static int counter = 0;
static float mean = 0;

// Projection clipping area
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;

static float* OldPositionX;
static float* OldPositionY;

static float* NewPositionX;
static float* NewPositionY;

static float* OldVelocityX;
static float* NewVelocityX;

static float* OldVelocityY;
static float* NewVelocityY;








__global__ void update(float* oldX, float* oldY, float * oldVx, float* oldVy, float* newX, float* newY, float* newVx, float* newVy, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	oldX[i] = newX[i];
	oldY[i] = newY[i];
	oldVx[i] = newVx[i];
	oldVy[i] = newVy[i];

}





__global__ void cudasetup(float* oldX, float* oldY, float * oldVx, float* oldVy, float* newX, float* newY, float* newVx, float* newVy,
	int size, float* alignment, float* cohesion, float* separation, float Xmax, float Xmin, float Ymax, float Ymin)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int neighborCount = 0;

	alignment[0] = 0;
	alignment[1] = 0;
	cohesion[0] = 0;
	cohesion[1] = 0;
	separation[0] = 0;
	separation[1] = 0;


	float CurrentX, CurrentY, CurrentVx, CurrentVy, oldCurrentX, oldCurrentY, oldCurrentVx, oldCurrentVy;

	CurrentX = newX[j];
	CurrentY = newY[j];
	CurrentVx = newVx[j];
	CurrentVy = newVy[j];
	oldCurrentX = oldX[j];
	oldCurrentY = oldY[j];
	oldCurrentVx = oldVx[j];
	oldCurrentVy = oldVy[j];

	//alignment
	int i = 0;

	for (i = 0; i < size; i++)
	{
		float x, y;
		x = oldCurrentX - oldX[i];
		y = oldCurrentY - oldY[i];

		float	distance = sqrt(x*x + y*y);
		if (distance != 0)
		{
		//	if (distance < 5)
			{

				alignment[0] += oldVx[i];
				alignment[1] += oldVy[i];
				neighborCount++;
			}
		}
	}
	alignment[0] -= oldCurrentVx;
	alignment[1] -= oldCurrentVy;
	alignment[0] /= 8;
	alignment[1] /= 8;

	//Separation
	neighborCount = 0;
	for (i = 0; i < size; i++)
	{
		float x, y;
		x = oldCurrentX - oldX[i];
		y = oldCurrentY - oldY[i];

		float	distance = sqrt(x*x + y*y);
		if (distance != 0)
		{
			if (distance < 0.5f)
			{
				separation[0] += oldX[i] - oldCurrentX;
				separation[1] += oldY[i] - oldCurrentY;
				neighborCount++;
			}
		}
	}
	separation[0] *= -1;
	separation[1] *= -1;
	//paration[0] /= 100;
//eparation[1] /= 100;
	//Cohesion
	neighborCount = 0;

			for (i = 0; i < size; i++)
	{

		float x, y;
		x = CurrentX - oldX[i];
		y = CurrentY - oldY[i];
		float	distance = sqrt(x*x + y*y);
		if (distance != 0)
		{
		//	if (distance <10)
			{
				cohesion[0] += oldX[i];
				cohesion[1] += oldY[i];
				neighborCount++;
			}
		}
		
	}
	cohesion[0] = cohesion[0] - oldCurrentX;
	cohesion[1] = cohesion[1] - oldCurrentY;
	cohesion[0] /= 100;
	cohesion[1] /= 100;

	CurrentVx += alignment[0] + separation[0] + cohesion[0];
	CurrentVy += alignment[1] + separation[1] + cohesion[1];


	CurrentX += CurrentVx;
	CurrentY += CurrentVy;

	if (CurrentX> Xmax) 
	{
		CurrentVx= -CurrentVx;
		CurrentX = Xmax;
	
	}
	else if (CurrentX < Xmin) {
		CurrentVx = -CurrentVx;
		CurrentX = Xmin;
	
	}
	if (CurrentY > Ymax) {
		CurrentVy = -CurrentVy;
		CurrentY = Ymax;
	}
	else if (CurrentY<Ymin) {
		CurrentVy = -CurrentVy;
		CurrentY = Ymin;

	}




	newX[j] = CurrentX;
	newY[j] = CurrentY;
	newVx[j] = CurrentVx;
	newVy[j] = CurrentVy;
	
	oldVx[j] = oldCurrentVx;
	oldVy[j] = oldCurrentVy;
	oldX[j] = oldCurrentX;
	oldY[j] = oldCurrentY;



}







__device__ bool ballColision(float X1, float Y1, float X2, float Y2)
{
	float x = X1 - X2;
	float y = Y1 - Y2;


	float distance = (x * x) + (y * y);

	float sum = 0.01f +0.01f;
	float multi = sum * sum;

	if (distance <= multi)
	{
		return true;
	}
	return false;
}

 bool cpuballColision(float X1, float Y1, float X2, float Y2)
{
	float x = X1 - X2;
	float y = Y1 - Y2;


	float distance = (x * x) + (y * y);

	float sum = 0.01f + 0.01f;
	float multi = sum * sum;

	if (distance <= multi)
	{
		return true;
	}
	return false;
}







__global__ void calculateBalls(float* oldX, float* oldY, float * oldVx, float* oldVy, float Xmax, float Xmin, float Ymax, float Ymin, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float newVx, newVy;
	newVx = oldX[i] + oldVx[i];
	newVy = oldY[i] + oldVy[i];

	//	Ymax = 0.0f;
	oldX[i] = newVx;
	oldY[i] = newVy;
	int counter = 0;
	int k = 0;
	for (k = 0; k < size; k++)
	{
		bool collided = ballColision(oldX[i], oldY[i], oldX[k], oldY[k]);
		if (collided == true)
		{
				if (oldVy[k] >0 && oldVy[i] < oldVy[k])
				{
					//k leci w gore, i leci w dó³ 
					if (oldVy[i] < 0)
					{
						oldVy[i] = -(oldVy[i] * 0.8);

				//	oldVy[k] -= (oldVy[i] * 0.8);
					}
				}
				if (oldVy[k] < 0 && oldVy[i] < 0)
				{
					if (oldVy[k] < oldVy[i])
					{
						oldVy[k] += 0.001*oldVy[k];
						oldVy[i] -= 0.001*oldVy[k];
					}
					else
					{
						oldVy[k] -= 0.001*oldVy[i];
						oldVy[i] == 0.001*oldVy[i];
					}
					
				}
				if (oldVy[k] > 0 && oldVy[i] > 0)
				{
					if (oldVy[k] > oldVy[i])
					{
						oldVy[k] -= 0.001*oldVy[k];
						oldVy[i] += 0.001*oldVy[k];
					}
					else
					{
						oldVy[k] += 0.001*oldVy[i];
						oldVy[i] -= 0.001*oldVy[i];
					}
				}



				//obie leca w prawo
				if (oldVx[i] > 0 && oldVx[k]>0)
				{ 
					if (oldVx[i] > oldVx[k])
					{
						oldVx[i] -= (oldVx[i] * 0.001);
						oldVx[k] += (oldVx[k] * 0.001);
					}
					else
					{
						oldVx[i] += (oldVx[i] * 0.001);
						oldVx[k] -= (oldVx[k] * 0.001);
					}
				}
				if (oldVx[i] < 0 && oldVx[k]<0)
				{
					if (oldVx[i] < oldVx[k])
					{
						oldVx[i] += (oldVx[i] * 0.001);
						oldVx[k] -= (oldVx[k] * 0.001);
					}
					else
					{
						oldVx[i] -= (oldVx[i] * 0.001);
						oldVx[k] += (oldVx[k] * 0.001);
					}
				}
		/*		if (oldVx[i] < 0 && oldVx[k]>0)
				{
					oldVx[i] = -0.8f*oldVx[i];
					oldVx[k] = -0.8f*oldVx[k];
				}
				if (oldVx[i] > 0 && oldVx[k]<0)
				{
					oldVx[i] = -0.8f*oldVx[i];
					oldVx[k] = -0.8f*oldVx[k];
				}*/



		}
	}

	if (oldX[i] > Xmax) {
		oldX[i] = Xmax;
		oldVx[i] = -oldVx[i];
	}
	else if (oldX[i] < Xmin) {
		oldX[i] = Xmin;
		oldVx[i] = -oldVx[i];
	}
	if (oldY[i] > Ymax) 
	{
		oldY[i] = Ymax;
	
		oldVy[i] = -oldVy[i]*0.8;
	}
	else if (oldY[i] < Ymin) {
		oldY[i] = Ymin;
		oldVy[i] = -oldVy[i];
	}


	oldVy[i] -= 0.0001f;

	if (oldVx[i]> 0)
	{
	oldVx[i] -= 0.00001f;
	}
	else
	oldVx[i] += 0.00001f;
}






void CudaBalls(float* oldX, float* oldY, float * oldVx, float* oldVy)
{


	int blockSize = size;
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;
	gridSize = (size + blockSize - 1) / blockSize;
	float* dev_oldX = 0;
	float* dev_oldY = 0;
	float* dev_oldVx = 0;
	float* dev_oldVy = 0;






	cudaMalloc((void**)&dev_oldX, size * sizeof(float));
	cudaMalloc((void**)&dev_oldY, size * sizeof(float));
	cudaMalloc((void**)&dev_oldVx, size * sizeof(float));
	cudaMalloc((void**)&dev_oldVy, size * sizeof(float));



	cudaMemcpy(dev_oldX, oldX, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_oldY, oldY, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_oldVx, oldVx, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_oldVy, oldVy, size * sizeof(float), cudaMemcpyHostToDevice);


	float d_XMin;
	float d_XMax;
	float d_YMin;
	float d_YMax;


	cudaMalloc((void**)&d_XMin, sizeof(float));
	cudaMalloc((void**)&d_XMax, sizeof(float));
	cudaMalloc((void**)&d_YMin, sizeof(float));
	cudaMalloc((void**)&d_YMax, sizeof(float));


	cudaMemcpy(&d_XMin, &XMin, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_XMax, &XMax, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_YMin, &YMin, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_YMax, &d_YMax, sizeof(float), cudaMemcpyHostToDevice);

	calculateBalls << <gridSize, blockSize >> >(dev_oldX, dev_oldY, dev_oldVx, dev_oldVy, XMax, XMin, YMax, YMin,size);


	cudaMemcpy(oldX, dev_oldX, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldY, dev_oldY, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldVx, dev_oldVx, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldVy, dev_oldVy, size * sizeof(float), cudaMemcpyDeviceToHost);

	OldPositionX = oldX;
	OldPositionY = oldY;
	OldVelocityX = oldVx;
	OldVelocityY = oldVy;


	cudaFree(dev_oldX);
	cudaFree(dev_oldY);
	cudaFree(dev_oldVx);
	cudaFree(dev_oldVy);

	cudaFree(&d_XMax);
	cudaFree(&d_XMin);
	cudaFree(&d_YMax);
	cudaFree(&d_YMin);
	
}


void cpuballs(float* oldX, float * oldY, float * oldVx, float* oldVy, float Xmax, float Xmin, float Ymax, float Ymin, int size)
{
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int i = 0;
	for (i = 0; i < size; i++)
	{
		float newVx, newVy;
		newVx = oldX[i] + oldVx[i];
		newVy = oldY[i] + oldVy[i];

		//	Ymax = 0.0f;
		oldX[i] = newVx;
		oldY[i] = newVy;

		int k = 0;
		for (k = 0; k < size; k++)
		{
			bool collided = cpuballColision(oldX[i], oldY[i], oldX[k], oldY[k]);
			if (collided == true)
			{
				//
				if (oldVy[k] >0 && oldVy[i] < oldVy[k])
				{
					//k leci w dó³, i leci w góre 
					if (oldVy[i] < 0)
					{
						oldVy[i] = -(oldVy[i] * 0.8);
						//		oldVy[k] -= 0.001f;
						oldVy[k] -= (oldVy[i] * 0.8);
					}
					//obie lec¹ w dó³
					if (oldVy[i] > 0)
					{
						oldVy[i] += 0.001f;
						oldVy[k] -= 0.001f;
					}
					if (oldVy[i] == 0)
					{
						oldVy[k] = 0.0f;
					}
				}

				if (oldVx[k] > 0 && oldVx[i] < oldVx[k])
				{
					if (oldVx[i] < 0)
					{
						oldVx[i] += 0.0001f;
						oldVx[k] -= 0.0001f;
					}

					if (oldVx[i] > 0)
					{
						//	oldVx[i]+=0.0001f;
						//	oldVx[k]-=0.0001f;
					}

				}

			}
		}





		//borders
		if (oldX[i] > Xmax) {
			oldX[i] = Xmax;
			oldVx[i] = -oldVx[i];
		}
		else if (oldX[i] < Xmin) {
			oldX[i] = Xmin;
			oldVx[i] = -oldVx[i];
		}
		if (oldY[i] > Ymax)
		{
			oldY[i] = Ymax;
			oldVy[i] = 0.0f;
			//oldVy[i] = -oldVy[i];
		}
		else if (oldY[i] < Ymin) {
			oldY[i] = Ymin;
			oldVy[i] = -oldVy[i];
		}
		//spadek

		oldVy[i] -= 0.0001f;

		if (oldVx[i]> 0)
		{
			oldVx[i] -= 0.00001f;
		}
		else
			oldVx[i] += 0.00001f;

	}
}




cudaError_t cudamovebirdies(float* oldX, float* oldY, float * oldVx, float* oldVy, float* newX, float* newY, float* newVx, float* newVy,
	int size)
{
	float xd = oldX[0];

	xd++;
	float* dev_oldX = 0;
	float* dev_oldY = 0;
	float* dev_oldVx = 0;
	float* dev_oldVy = 0;

	float* dev_newX = 0;
	float* dev_newY = 0;
	float* dev_newVx = 0;
	float* dev_newVy = 0;

	float* alignment = new float[2];
	float* cohesion = new float[2];
	float* separation = new float[2];

	alignment[0] = 0;
	alignment[1] = 0;
	cohesion[0] = 0;
	cohesion[1] = 0;
	separation[0] = 0;
	separation[1] = 0;

	float* dev_alignment = 0;
	float* dev_cohesion = 0;
	float* dev_separation = 0;

	float d_XMin;
	float d_XMax;
	float d_YMin;
	float d_YMax;


	cudaMalloc((void**)&d_XMin,  sizeof(float));
	cudaMalloc((void**)&d_XMax,  sizeof(float));
	cudaMalloc((void**)&d_YMin,  sizeof(float));
	cudaMalloc((void**)&d_YMax,  sizeof(float));


	cudaMemcpy(&d_XMin, &XMin,  sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_XMax, &XMax, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_YMin, &YMin, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_YMax, &d_YMax, sizeof(float), cudaMemcpyHostToDevice);



	//float ab[20];
	//int i;

	//for (i = 0; i<20; i++)
	//{
	//	ab[i] = oldX[i];
	//}


	int blockSize = size;
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;
	cudaError_t cudaStatus;
	// Round up according to array size 
	gridSize = (size + blockSize - 1) / blockSize;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaMalloc((void**)&dev_oldX, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaMalloc((void**)&dev_oldY, size * sizeof(float));
	cudaMalloc((void**)&dev_oldVx, size * sizeof(float));
	cudaMalloc((void**)&dev_oldVy, size * sizeof(float));



	cudaMemcpy(dev_oldX, oldX, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_oldY, oldY, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_oldVx, oldVx, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_oldVy, oldVy, size * sizeof(float), cudaMemcpyHostToDevice);


	cudaMalloc((void**)&dev_newX, size * sizeof(float));
	cudaMalloc((void**)&dev_newY, size * sizeof(float));
	cudaMalloc((void**)&dev_newVx, size * sizeof(float));
	cudaMalloc((void**)&dev_newVy, size * sizeof(float));

	cudaMemcpy(dev_newX, newX, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_newY, newY, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_newVx, newVx, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_newVy, newVy, size * sizeof(float), cudaMemcpyHostToDevice);


	cudaMalloc((void**)&dev_alignment, 2 * sizeof(float));
	cudaMalloc((void**)&dev_cohesion, 2 * sizeof(float));
	cudaMalloc((void**)&dev_separation, 2 * sizeof(float));

	cudaMemcpy(dev_alignment, alignment, 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cohesion, cohesion, 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_separation, separation, 2 * sizeof(float), cudaMemcpyHostToDevice);

	float ab[20];
	int i = 0;
	{
		cudasetup << <gridSize, blockSize >> >(dev_oldX, dev_oldY, dev_oldVx, dev_oldVy, dev_newX, dev_newY, dev_newVx, dev_newVy, size, dev_alignment, dev_cohesion, dev_separation, XMax, XMin, YMax, YMin);
		//cudasetup << <gridSize, blockSize >> >(dev_oldX, dev_oldY, dev_oldVx, dev_oldVy, dev_newX, dev_newY, dev_newVx, dev_newVy, dev_newX[i], dev_newY[i], dev_oldX[i], dev_oldY[i], dev_newVx[i], dev_newVy[i], dev_oldVx[i], dev_oldVy[i], size);
	}
	for (i = 0; i<20; i++)
	{
		ab[i] = newX[i];
	}
	

	update << <gridSize, blockSize >> >(dev_oldX, dev_oldY, dev_oldVx, dev_oldVy, dev_newX, dev_newY, dev_newVx, dev_newVy, size);

	//back to cpu mermory
	cudaMemcpy(oldX, dev_oldX, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldY, dev_oldY, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldVx, dev_oldVx, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldVy, dev_oldVy, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(newX, dev_newX, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(newY, dev_newY, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(newVx, dev_newVx, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(newVy, dev_newVx, size * sizeof(float), cudaMemcpyDeviceToHost);


	for (i = 0; i<20; i++)
	{
		ab[i] = newX[i];
	}

	cudaFree(dev_oldX);
	cudaFree(dev_oldY);
	cudaFree(dev_oldVx);
	cudaFree(dev_oldVy);

	cudaFree(dev_newX);
	cudaFree(dev_newY);
	cudaFree(dev_newVx);
	cudaFree(dev_newVy);

	cudaFree(dev_alignment);
	cudaFree(dev_cohesion);
	cudaFree(dev_separation);

	return cudaStatus;
}


/* Initialize OpenGL Graphics */
void initGL() {
	glClearColor(0.0, 0.0, 0.0, 1.0); // Set background (clear) color to black
}


static double second(void)
{
	LARGE_INTEGER t;
	static double oofreq;
	static int checkedForHighResTimer;
	static BOOL hasHighResTimer;

	if (!checkedForHighResTimer) {
		hasHighResTimer = QueryPerformanceFrequency(&t);
		oofreq = 1.0 / (double)t.QuadPart;
		checkedForHighResTimer = 1;
	}
	if (hasHighResTimer) {
		QueryPerformanceCounter(&t);
		return (double)t.QuadPart * oofreq;
	}
	else {
		return (double)GetTickCount() / 1000.0;
	}
}


void display() 
{


	glClear(GL_COLOR_BUFFER_BIT);  // Clear the color buffer
	glMatrixMode(GL_MODELVIEW);    // To operate on the model-view matrix
	glLoadIdentity();        // Reset model-view matrix



	//cudamovebirdies(OldPositionX, OldPositionY, OldVelocityX, OldVelocityY, NewPositionX, NewPositionY, NewVelocityX, NewVelocityY, size);


	float hostTime;
	double startTime, stopTime, elapsed;
	startTime = second();
	

	CudaBalls(OldPositionX, OldPositionY, OldVelocityX, OldVelocityY);

//	cpuballs(OldPositionX, OldPositionY, OldVelocityX, OldVelocityY, XMax, XMin, YMax, YMin, size);
	
	stopTime = second();
	hostTime = (stopTime - startTime) * 1000;
	fpsCurrent = 1000 / hostTime;
	fpsTotal += fpsCurrent;
	counter++;
	mean = fpsTotal / counter;




	for (int j = 0; j < size; j++)
	{
		// Use triangular segments to form a circle
		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.0f, 0.0f, 1.0f);  // Blue
		glVertex2f(OldPositionX[j], OldPositionY[j]);       // Center of circle
		int numSegments = 15;
		GLfloat angle;
		for (int i = 0; i <= numSegments; i++) { // Last vertex same as first vertex
			angle = i * 2.0f * PI / numSegments;  // 360 deg for all segments
			glVertex2f((cos(angle) * ballRadius) + OldPositionX[j], OldPositionY[j] + (sin(angle) * ballRadius));
		}
		glEnd();

		}

	glutSwapBuffers();

}


/* Call back when the windows is re-sized */
void reshape(GLsizei width, GLsizei height) {
	// Compute aspect ratio of the new window
	if (height == 0) height = 1;                // To prevent divide by 0
	GLfloat aspect = (GLfloat)width / (GLfloat)height;

	// Set the viewport to cover the new window
	glViewport(0, 0, width, height);

	// Set the aspect ratio of the clipping area to match the viewport
	glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
	glLoadIdentity();             // Reset the projection matrix
	if (width >= height) {
		clipAreaXLeft = -1.0 * aspect;
		clipAreaXRight = 1.0 * aspect;
		clipAreaYBottom = -1.0;
		clipAreaYTop = 1.0;
	}
	else {
		clipAreaXLeft = -1.0;
		clipAreaXRight = 1.0;
		clipAreaYBottom = -1.0 / aspect;
		clipAreaYTop = 1.0 / aspect;
	}
	gluOrtho2D(clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop);
	XMin = clipAreaXLeft;
	XMax = clipAreaXRight;
	YMin = clipAreaYBottom;
	YMax = clipAreaYTop;
}

/* Called back when the timer expired */
void Timer(int value) {
	glutPostRedisplay();    // Post a paint request to activate display()
	glutTimerFunc(refreshMillis, Timer, 0); // subsequent timer call at milliseconds
}



int main()
{
	int i;
	//srand(time(NULL));
	OldPositionX = new float[size];
	OldPositionY = new float[size];
	OldVelocityX = new float[size];
	OldVelocityY = new float[size];

	NewPositionX = new float[size];
	NewPositionY = new float[size];
	NewVelocityX = new float[size];
	NewVelocityY = new float[size];

	srand((unsigned int)time(NULL));

	float a = 50;
	for (i = 0; i < size; i++)
	{
	
		
		float r1 = -2.0f + (rand() / (float)RAND_MAX * 4.2f);
		float r2 = -3.0f + (rand() / (float)RAND_MAX * 4.2f);
		float v = -0.01f+ (rand() / (float)RAND_MAX * 0.01f);
		OldPositionX[i] = r1;
		OldPositionY[i] = r2;
		OldVelocityX[i] = v;
		OldVelocityY[i] = v;
		NewPositionX[i] = 0;
		NewPositionY[i] = 0;
		NewVelocityX[i] = 0;
		NewVelocityY[i] = 0;
	}
	//	int d =  OldPositionX[0];




	char fakeParam[] = "fake";
	char *fakeargv[] = { fakeParam, NULL };
	int fakeargc = 1;

	glutInit(&fakeargc, fakeargv);            // Initialize GLUT
	glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
	glutInitWindowSize(windowWidth, windowHeight);  // Initial window width and height
	glutInitWindowPosition(windowPooldVx, windowPooldVy); // Initial window top-left corner (x, y)
	glutCreateWindow(title);      // Create window with given title
	glutDisplayFunc(display);     // Register callback handler for window re-paint
	glutReshapeFunc(reshape);     // Register callback handler for window re-shape
	glutTimerFunc(0, Timer, 0);   // First timer call immediately
	initGL();                     // Our own OpenGL initialization
	glutMainLoop();
	system("pause");
	return 0;

}