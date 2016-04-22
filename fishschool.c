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
int windowPosX = 50;      // Windowed mode's top-left corner x
int windowPosY = 50;      // Windowed mode's top-left corner y


static double mytime = 0;

static float fpsCurrent = 0;
static float fpsTotal = 0;
static int counter = 0;
static float mean = 0;

float XMax, XMin, YMax, YMin; // Ball's center (x, y) bounds
GLfloat xSpeed = 0.02f;      // Ball's speed in x and y directions
GLfloat ySpeed = 0.007f;
int refreshMillis = 50;      // Refresh period in milliseconds
float Angle = atan2(-xSpeed, ySpeed) * 180 / PI;

const int size = 800;

float placeX=1;
float placeY = 1;

int rulemultiplier = 1;

// Projection clipping area
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;

static float OldPositionX[size];
static float OldPositionY[size];

static float NewPositionX[size];
static float NewPositionY[size];

static float OldVelocityX[size];
static float NewVelocityX[size];

static float OldVelocityY[size];
static float NewVelocityY[size];








__global__ void update(float* oldX, float* oldY, float * oldVx, float* oldVy, float* newX, float* newY, float* newVx, float* newVy, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	oldX[i] = newX[i];
	oldY[i] = newY[i];
	oldVx[i] = newVx[i];
	oldVy[i] = newVy[i];

}





__global__ void cudasetup(float* oldX, float* oldY, float * oldVx, float* oldVy, float* newX, float* newY, float* newVx, float* newVy,
	int size, float* alignment2, float* cohesion2, float* separation2, float Xmax, float Xmin, float Ymax, float Ymin, float placeX, float placeY, int M)
{
	int j = threadIdx.x;
	int neighborCount = 0;

	float alignment[2];
	float separation[2];
	float cohesion[2];

	alignment[0] = 0;
	alignment[1] = 0;
	cohesion[0] = 0;
	cohesion[1] = 0;
	separation[0] = 0;
	separation[1] = 0;
//
//	float placeX =1;
//	float placeY = 1;

	float CurrentX, CurrentY, CurrentVx, CurrentVy, oldCurrentX, oldCurrentY, oldCurrentVx, oldCurrentVy;

	/*CurrentX = newX[j];
	CurrentY = newY[j];
	CurrentVx = newVx[j];
	CurrentVy = newVy[j];*/

	CurrentX = oldX[j];
	CurrentY = oldY[j];
	CurrentVx = oldVx[j];
	CurrentVy = oldVy[j];


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
	if (distance < 0.015f)
	{

	alignment[0] += oldVx[i];
	alignment[1] += oldVy[i];
	neighborCount++;
	}
	}
	}
	if (neighborCount != 0)
	{
	//	alignment[0] /= neighborCount;
	//alignment[1] /= neighborCount;
	alignment[0] /= (size-1);
	alignment[1] /= (size-1);

	alignment[0] = (alignment[0] - oldCurrentVx) / 10;
	alignment[1] = (alignment[1] - oldCurrentVy) / 10;

	}




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
	if (distance < 0.018f)
	{
	separation[0] -= (oldX[i] - oldCurrentX);
	separation[1] -= (oldY[i] - oldCurrentY);
	neighborCount++;
	}
	}
	}
	//separation[0] *= -1;
	//separation[1] *= -1;
	if (neighborCount != 0)
	{
	separation[0] /= neighborCount;
	separation[1] /= neighborCount;
	}


	//Cohesion  na stronie kolejnosc jest odwrotna yo.
	neighborCount = 0;

	for (i = 0; i < size; i++)
	{

	float x, y;
	x = oldCurrentX - oldX[i]; //tu 
	y = oldCurrentY - oldY[i];
	float	distance = sqrt(x*x + y*y);
	if (distance != 0)
	{
	if (distance <0.015f)
	{
	cohesion[0] += oldX[i];
	cohesion[1] += oldY[i];
	neighborCount++;
	}
	}

	}
	if (neighborCount != 0)
	{
	//cohesion[0] /= neighborCount;
	//cohesion[1] /= neighborCount;

	cohesion[0] /= (size-1);
	cohesion[1] /= (size - 1);
	
	cohesion[0] = (cohesion[0] - oldCurrentX) / 100;
	cohesion[1] = (cohesion[1] - oldCurrentY) / 100;

	}


	//bounding them.

	float boundaryX = 0;
	float boundaryY = 0;

	if (CurrentX < Xmin)
	boundaryX = 0.0001f;
	if (CurrentX > Xmax)
	boundaryX = -0.0001f;


	if (CurrentY < Ymin)
	boundaryY = 0.0001f;
	if (CurrentY > Ymax)
	boundaryY = -0.0001f;




	placeX = (placeX - CurrentX)/100;
	placeY = (placeY - CurrentY)/100;









	//adding all up



	//CurrentVx += alignment[0] + separation[0] + cohesion[0];
//	CurrentVy += alignment[1] + separation[1] + cohesion[1];

	//CurrentVx +=   separation[0] + alignment[0];
	//CurrentVy +=	separation[1] + alignment[1];

	//CurrentVx += cohesion[0];
//	CurrentVy += cohesion[1];
	CurrentVx += M*(alignment[0] + separation[0] + cohesion[0]) + placeX+ boundaryX;
	CurrentVy += M*(alignment[1] + separation[1] + cohesion[1]) + placeY + boundaryY;

	//CurrentVx += 0.001f;
	//CurrentVy += 0.001f;
	//CurrentVx++;
	//CurrentVy++;
	//CurrentVx = CurrentVx;
	//CurrentVy = CurrentVy;

	CurrentX += CurrentVx*0.3f;
	CurrentY += CurrentVy*0.3f;

//	CurrentX = CurrentX;
	


	//if (CurrentX> 1.0f)
	//{
	//	CurrentVx = -CurrentVx;
	//	CurrentX = 1.0f;

	//}
	//else if (CurrentX < -1.0f) {
	//	CurrentVx = -CurrentVx;
	//	CurrentX = -1.0f;

	//}
	//if (CurrentY > 1.0f) {
	//	CurrentVy = -CurrentVy;
	//	CurrentY = 1.0f;
	//}
	//else if (CurrentY<-1.0f) {
	//	CurrentVy = -CurrentVy;
	//	CurrentY = -1.0f;

	//}



	newX[j] = CurrentX;
	newY[j] = CurrentY;
	newVx[j] = CurrentVx;
	newVy[j] = CurrentVy;
	//oldVx[j] = oldCurrentVx;
	//oldVy[j] = oldCurrentVy;
	//oldX[j] = oldCurrentX;
	//oldY[j] = oldCurrentY;

}



void CpuBirds(float* oldX, float* oldY, float * oldVx, float* oldVy, float* newX, float* newY, float* newVx, float* newVy,
	int size,  float Xmax, float Xmin, float Ymax, float Ymin, float placeX, float placeY, int M)
{

	int j = 0;
	for (j = 0; j < size; j++)
	{
	int neighborCount = 0;

	float alignment[2];
	float separation[2];
	float cohesion[2];

	alignment[0] = 0;
	alignment[1] = 0;
	cohesion[0] = 0;
	cohesion[1] = 0;
	separation[0] = 0;
	separation[1] = 0;
	//
	//	float placeX =1;
	//	float placeY = 1;

	float CurrentX, CurrentY, CurrentVx, CurrentVy, oldCurrentX, oldCurrentY, oldCurrentVx, oldCurrentVy;

	/*CurrentX = newX[j];
	CurrentY = newY[j];
	CurrentVx = newVx[j];
	CurrentVy = newVy[j];*/

	CurrentX = oldX[j];
	CurrentY = oldY[j];
	CurrentVx = oldVx[j];
	CurrentVy = oldVy[j];


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
	if (distance < 0.015f)
	{

	alignment[0] += oldVx[i];
	alignment[1] += oldVy[i];
	neighborCount++;
	}
	}
	}
	if (neighborCount != 0)
	{
	//	alignment[0] /= neighborCount;
	//alignment[1] /= neighborCount;
	alignment[0] /= (size - 1);
	alignment[1] /= (size - 1);

	alignment[0] = (alignment[0] - oldCurrentVx) / 7;
	alignment[1] = (alignment[1] - oldCurrentVy) / 7;

	}




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
	if (distance < 0.017f)
	{
	separation[0] -= (oldX[i] - oldCurrentX);
	separation[1] -= (oldY[i] - oldCurrentY);
	neighborCount++;
	}
	}
	}
	//separation[0] *= -1;
	//separation[1] *= -1;
	if (neighborCount != 0)
	{
	separation[0] /= neighborCount;
	separation[1] /= neighborCount;
	}


	//Cohesion  na stronie kolejnosc jest odwrotna yo.
	neighborCount = 0;

	for (i = 0; i < size; i++)
	{

	float x, y;
	x = oldCurrentX - oldX[i]; //tu 
	y = oldCurrentY - oldY[i];
	float	distance = sqrt(x*x + y*y);
	if (distance != 0)
	{
	if (distance < 0.015f)
	{
	cohesion[0] += oldX[i];
	cohesion[1] += oldY[i];
	neighborCount++;
	}
	}

	}
	if (neighborCount != 0)
	{
	//cohesion[0] /= neighborCount;
	//cohesion[1] /= neighborCount;

	cohesion[0] /= (size - 1);
	cohesion[1] /= (size - 1);

	cohesion[0] = (cohesion[0] - oldCurrentX) / 100;
	cohesion[1] = (cohesion[1] - oldCurrentY) / 100;

	}


	//bounding them.

	float boundaryX = 0;
	float boundaryY = 0;

	if (CurrentX < Xmin)
	boundaryX = 0.0001f;
	if (CurrentX > Xmax)
	boundaryX = -0.0001f;


	if (CurrentY < Ymin)
	boundaryY = 0.0001f;
	if (CurrentY > Ymax)
	boundaryY = -0.0001f;




	placeX = (placeX - CurrentX) / 100;
	placeY = (placeY - CurrentY) / 100;









	//adding all up



	//CurrentVx += alignment[0] + separation[0] + cohesion[0];
	//	CurrentVy += alignment[1] + separation[1] + cohesion[1];

	//CurrentVx +=   separation[0] + alignment[0];
	//CurrentVy +=	separation[1] + alignment[1];

	//CurrentVx += cohesion[0];
	//	CurrentVy += cohesion[1];
	CurrentVx += M*(alignment[0] + separation[0] + cohesion[0]) + placeX + boundaryX;
	CurrentVy += M*(alignment[1] + separation[1] + cohesion[1]) + placeY + boundaryY;

	//CurrentVx += 0.001f;
	//CurrentVy += 0.001f;
	//CurrentVx++;
	//CurrentVy++;
	//CurrentVx = CurrentVx;
	//CurrentVy = CurrentVy;

	float maxVelocity = 0.000001f;
	if (abs(CurrentVx) > maxVelocity)
	{
	CurrentVx = (CurrentVx / abs(CurrentVx)) * maxVelocity;
	}

	if (abs(CurrentVy) > maxVelocity)
	{
	CurrentVy = (CurrentVy / abs(CurrentVy)) * maxVelocity;
	}

	CurrentX += CurrentVx*0.3f;
	CurrentY += CurrentVy*0.3f;

	//	CurrentX = CurrentX;



	//if (CurrentX> 1.0f)
	//{
	//	CurrentVx = -CurrentVx;
	//	CurrentX = 1.0f;

	//}
	//else if (CurrentX < -1.0f) {
	//	CurrentVx = -CurrentVx;
	//	CurrentX = -1.0f;

	//}
	//if (CurrentY > 1.0f) {
	//	CurrentVy = -CurrentVy;
	//	CurrentY = 1.0f;
	//}
	//else if (CurrentY<-1.0f) {
	//	CurrentVy = -CurrentVy;
	//	CurrentY = -1.0f;

	//}



	newX[j] = CurrentX;
	newY[j] = CurrentY;
	newVx[j] = CurrentVx;
	newVy[j] = CurrentVy;
	//oldVx[j] = oldCurrentVx;
	//oldVy[j] = oldCurrentVy;
	//oldX[j] = oldCurrentX;
	//oldY[j] = oldCurrentY;
	}
	int i = 0;
	for (i = 0; i < size; i++)
	{
	oldX[i] = newX[i];
	oldY[i] = newY[i];
	oldVx[i] = newVx[i];
	oldVy[i] = newVy[i];
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


	cudaMalloc((void**)&d_XMin, sizeof(float));
	cudaMalloc((void**)&d_XMax, sizeof(float));
	cudaMalloc((void**)&d_YMin, sizeof(float));
	cudaMalloc((void**)&d_YMax, sizeof(float));


	cudaMemcpy(&d_XMin, &XMin, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_XMax, &XMax, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_YMin, &YMin, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_YMax, &d_YMax, sizeof(float), cudaMemcpyHostToDevice);


	int blockSize = size;
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;
	cudaError_t cudaStatus;
	// Round up according to array size 
	gridSize = (size + blockSize - 1) / blockSize;
//	gridSize = size / 1000;
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

	//float ab[20];
	//int i = 0;
	{
	cudasetup << <1, 1000 >> >(dev_oldX, dev_oldY, dev_oldVx, dev_oldVy, dev_newX, dev_newY, dev_newVx, dev_newVy, size, dev_alignment, dev_cohesion, dev_separation, XMax, XMin, YMax, YMin, placeX, placeY, rulemultiplier);
	}
	//for (i = 0; i<20; i++)
	//{
	//	ab[i] = newX[i];
	//}


	update << <1, 1000 >> >(dev_oldX, dev_oldY, dev_oldVx, dev_oldVy, dev_newX, dev_newY, dev_newVx, dev_newVy, size);

	//back to cpu mermory
	cudaMemcpy(oldX, dev_oldX, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldY, dev_oldY, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldVx, dev_oldVx, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldVy, dev_oldVy, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(newX, dev_newX, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(newY, dev_newY, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(newVx, dev_newVx, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(newVy, dev_newVx, size * sizeof(float), cudaMemcpyDeviceToHost);


	//for (i = 0; i<20; i++)
	//{
	//	ab[i] = newX[i];
	//}

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




float ballRadius = 0.01f;
/* Initialize OpenGL Graphics */
void initGL() {
	glClearColor(0.0, 0.0, 0.0, 1.0); // Set background (clear) color to black
}

void display() {


	glClear(GL_COLOR_BUFFER_BIT);  // Clear the color buffer
	glMatrixMode(GL_MODELVIEW);    // To operate on the model-view matrix
	glLoadIdentity();        // Reset model-view matrix


	float hostTime;
	double startTime, stopTime, elapsed;
	startTime = second();



	cudamovebirdies(OldPositionX, OldPositionY, OldVelocityX, OldVelocityY, NewPositionX, NewPositionY, NewVelocityX, NewVelocityY, size);
	//CpuBirds(OldPositionX, OldPositionY, OldVelocityX, OldVelocityY, NewPositionX, NewPositionY, NewVelocityX, NewVelocityY, size, XMax, XMin, YMax, YMin, placeX, placeY, rulemultiplier);


	stopTime = second();
	hostTime = (stopTime - startTime) * 1000;
	fpsCurrent = 1000 / hostTime;
	fpsTotal += fpsCurrent;
	counter++;
	mean = fpsTotal / counter;

	printf("Fps: %f \n", mean);

	//for (int i = 0; i < size; i++)
	//{
	//	float x;
	//	float y;
	//	x = NewPositionX[i];
	//	y = NewPositionY[i];

	//	float Vx = NewVelocityX[i];
	//	float Vy = NewVelocityY[i];

	//	Angle = atan2(-Vx, Vy) * 180 / PI;

	//	//	glTranslatef(x, y, 0.0f);
	//	//glRotatef(Angle, x, y, 0.0f);

	//	////glColor3f(0.0f + 0.2f*i, 0.0f + 0.1f*i, 1.0f);
	//	//glBegin(GL_TRIANGLES);
	//	//glColor3f(0.0f, 0.0f, 1.0f);
	//	//glVertex2f(0.003f + x, y - 0.02f);
	//	//glVertex2f(0.03f + x, y - 0.02f);
	//	//glVertex2f(0.02f + x, y + 0.03f);


	//	glEnd();

	//}
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
void getMouse (int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) 
	{ // Pause/resume
	placeX = x / (float)windowWidth;
	placeY = 1- y / (float)windowHeight;
	printf("PlaceX : %f\n", placeX);
	printf("PlaceY : %f\n", placeY);
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
	if (rulemultiplier == 1)
	rulemultiplier = -1;
	else
	rulemultiplier = 1;
	}
}


int main()
{
	int i;
	//srand(time(NULL));


	srand((unsigned int)time(NULL));

	float a = 50;
	for (i = 0; i < size; i++)
	{


	float r1 = -0.5f + (rand() / (float)RAND_MAX * 0.5f);
	float r2 = -0.5f + (rand() / (float)RAND_MAX * 0.5f);
	float vX = 0 + (rand() / (float)RAND_MAX * 0.0001f);
	float vY = 0 + (rand() / (float)RAND_MAX * 0.0001f);
	OldPositionX[i] = r1;
	OldPositionY[i] = r2;
	OldVelocityX[i] = vX;
	OldVelocityY[i] = vY;
	NewPositionX[i] = r1;
	NewPositionY[i] = r2;
	NewVelocityX[i] = vX;
	NewVelocityY[i] = vY;
	}
	//	int d =  OldPositionX[0];




	char fakeParam[] = "fake";
	char *fakeargv[] = { fakeParam, NULL };
	int fakeargc = 1;

	glutInit(&fakeargc, fakeargv);            // Initialize GLUT
	glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
	glutInitWindowSize(windowWidth, windowHeight);  // Initial window width and height
	glutInitWindowPosition(windowPosX, windowPosY); // Initial window top-left corner (x, y)
	glutCreateWindow(title);      // Create window with given title
	glutDisplayFunc(display);     // Register callback handler for window re-paint
	glutReshapeFunc(reshape);     // Register callback handler for window re-shape
	glutTimerFunc(0, Timer, 0);   // First timer call immediately
	glutMouseFunc(getMouse);
	initGL();                     // Our own OpenGL initialization
	glutMainLoop();
	system("pause");
	return 0;

}