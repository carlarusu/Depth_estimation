// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <queue>
#include <stack>
#include <random>
#include <math.h>
using namespace cv;
using namespace std;

uint64_t*** processed_sums;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

bool isInside(Mat img, int i, int j) {
	if (i < img.rows && i >= 0 && j < img.cols && j >= 0)
		return true;
	return false;
}

Mat_<float> convolve(Mat_<float> srcf, Mat_<float> kernel) {
	Mat_<float> dstf(srcf.rows, srcf.cols);
	for (int i = 0; i < srcf.rows; i++) {
		for (int j = 0; j < srcf.cols; j++) {
			float center = 0.0;
			for (int u = 0; u < kernel.rows; u++) {
				for (int v = 0; v < kernel.cols; v++) {
					//check if inside the image
					if (isInside(srcf, i + u - kernel.rows / 2, j + v - kernel.cols / 2)) {
						center += srcf(i + u - kernel.rows / 2, j + v - kernel.cols / 2) * kernel(u, v);
					}
				}
			}
			dstf(i, j) = center;
		}
	}
	return dstf;
}

Mat_<uchar> normalize(Mat_<float> srcf, Mat_<float> kernel, int type = 0) {
	float max = 0;
	float min = 0;
	Mat_<uchar> dst(srcf.rows, srcf.cols);

	for (int u = 0; u < kernel.rows; u++) {
		for (int v = 0; v < kernel.cols; v++) {
			if (kernel(u, v) > 0)
				max += kernel(u, v);
			else
				min += kernel(u, v);
		}
	}
	max *= 255;
	min *= 255;

	for (int i = 0; i < srcf.rows; i++) {
		for (int j = 0; j < srcf.cols; j++) {
			dst(i, j) = round(255 * (srcf(i, j) - min) / (max - min));

		}
	}

	return dst;
}

Mat_<uchar> spatial_filter(Mat_<uchar> src, Mat_<float> kernel) {
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	Mat_<float> dstf = convolve(srcf, kernel);
	Mat_<uchar> dst = normalize(dstf, kernel);

	return dst;
}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat_<float> process_mag(Mat img) {
	Mat_<float> dst(img.rows, img.cols);
	float max = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			dst.at<float>(i, j) = log(img.at<float>(i, j) + 1);
			if (dst(i, j) > max)
				max = dst(i, j);
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			dst(i, j) = dst(i, j) / max;
		}
	}

	imshow("processed magnitude", dst);
	return dst;
}

Mat_<float> ideal_low_pass(Mat_<float> mag) {
	Mat_<float> dst(mag.rows, mag.cols);
	int R = 20;
	float H = mag.rows;
	float W = mag.cols;

	for (int u = 0; u < mag.rows; u++) {
		for (int v = 0; v < mag.cols; v++) {
			if (((H / 2 - u) * (H / 2 - u) + (W / 2 - v) * (W / 2 - v)) <= R * R)
				dst(u, v) = mag(u, v);
			else
				dst(u, v) = 0;
		}
	}

	return dst;
}

Mat_<float> ideal_high_pass(Mat_<float> mag) {
	Mat_<float> dst(mag.rows, mag.cols);
	int R = 20;
	float H = mag.rows;
	float W = mag.cols;

	for (int u = 0; u < mag.rows; u++) {
		for (int v = 0; v < mag.cols; v++) {
			if (((H / 2 - u) * (H / 2 - u) + (W / 2 - v) * (W / 2 - v)) > R* R)
				dst(u, v) = mag(u, v);
			else
				dst(u, v) = 0;
		}
	}

	return dst;
}

Mat_<float> gaussian_low_pass(Mat_<float> mag) {
	Mat_<float> dst(mag.rows, mag.cols);
	int A = 20;
	float H = mag.rows;
	float W = mag.cols;

	for (int u = 0; u < mag.rows; u++) {
		for (int v = 0; v < mag.cols; v++) {
			dst(u, v) = mag(u, v) * exp(-(((H / 2 - u) * (H / 2 - u) + (W / 2 - v) * (W / 2 - v)) / (A * A)));
		}
	}

	return dst;
}

Mat_<float> gaussian_high_pass(Mat_<float> mag) {
	Mat_<float> dst(mag.rows, mag.cols);
	int A = 20;
	float H = mag.rows;
	float W = mag.cols;

	for (int u = 0; u < mag.rows; u++) {
		for (int v = 0; v < mag.cols; v++) {
			dst(u, v) = mag(u, v) * (1.0 - (float)exp(-(((H / 2 - u) * (H / 2 - u) + (W / 2 - v) * (W / 2 - v)) / (A * A))));
		}
	}

	return dst;
}

Mat generic_frequency_domain_filter(Mat src) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	//display the phase and magnitude images here
	Mat_<float> processed_mag = process_mag(mag);

	//insert filtering operations on Fourier coefficients here
	Mat_<float> filtered_mag = ideal_low_pass(mag);
	//Mat_<float> filtered_mag = ideal_high_pass(mag);
	//Mat_<float> filtered_mag = gaussian_low_pass(mag);
	//Mat_<float> filtered_mag = gaussian_high_pass(mag);

	//store in real part in channels[0] and imaginary part in channels[1]
	for (int u = 0; u < filtered_mag.rows; u++) {
		for (int v = 0; v < filtered_mag.cols; v++) {
			channels[0].at<float>(u, v) = filtered_mag(u, v) * cos(phi.at<float>(u, v));
			channels[1].at<float>(u, v) = filtered_mag(u, v) * sin(phi.at<float>(u, v));
		}
	}

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);
	return dst;
}

Mat_<uchar> median_filter(int w, Mat_<uchar> src) {
	Mat_<uchar> dst(src.rows, src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			vector<int> v;
			for (int di = -w / 2; di <= w / 2; di++) {
				for (int dj = -w / 2; dj <= w / 2; dj++) {
					if (isInside(src, i + di, j + dj)) {
						v.push_back(src(i + di, j + dj));
					}
				}
			}
			sort(v.begin(), v.end());
			dst(i, j) = v[v.size() / 2];
		}
	}

	return dst;
}

Mat_<float> gaussian_2d_filter(int w) {
	Mat_<float> filter(w, w);
	double sigma = w / 6.0;
	int x0 = w / 2;
	int y0 = w / 2;

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < w; y++) {
			filter(x, y) = exp(-((((x - x0) * (x - x0)) + ((y - y0) * (y - y0))) / (2 * sigma * sigma))) / (2 * PI * sigma * sigma);
		}
	}

	return filter;
}

Mat_<float> gaussian_1d_filter_gx(int w) {
	Mat_<float> filter(1, w);
	double sigma = w / 6.0;
	int y0 = w / 2;

	for (int y = 0; y < w; y++) {
		filter(0, y) = exp(-(((y - y0) * (y - y0)) / (2 * sigma * sigma))) / sqrt(2 * PI * sigma * sigma);
	}

	return filter;
}

Mat_<float> gaussian_1d_filter_gy(int w) {
	Mat_<float> filter(w, 1);
	double sigma = w / 6.0;
	int x0 = w / 2;

	for (int x = 0; x < w; x++) {
		filter(x, 0) = exp(-(((x - x0) * (x - x0)) / (2 * sigma * sigma))) / sqrt(2 * PI * sigma * sigma);
	}

	return filter;
}

// step 1.
uint64_t** census_transform(Mat_<uchar> img, int size) {
	// create an 64 bit H*W matrix to store the census codes
	uint64_t** dst;
	dst = new uint64_t * [img.rows];
	for (int i = 0; i < img.rows; i++) {
		dst[i] = new uint64_t[img.cols]();
	}

	// take a size * size window for each pixel 
	for (int i = size / 2; i < img.rows - size / 2; i++) {
		for (int j = size / 2; j < img.cols - size / 2; j++) {
			// census will contain the code
			uint64_t census = 0ULL;
			for (int u = -size / 2; u <= size / 2; u++) {
				for (int v = -size / 2; v <= size / 2; v++) {
					// for every pixel that is not the center of the window
					if ((u != 0 || v != 0)) {
						// shift left by one bit
						census <<= 1ULL;
						// census code is 1 if the neighbouring bit is smaller than the center
						if (img(i + u, j + v) < img(i, j))
							census += 1ULL;
						// and 0 if it is bigger
						else
							census += 0ULL;
					}
				}
			}
			//cout << census << " ";
			dst[i][j] = census;
		}
	}

	return dst;
}

// step 2. helper function
int hamming_distance(uint64_t x, uint64_t y) {
	// hamming distance is amount of differing bits in census codes
	// xor leaves only differing bits
	uint64_t xor = (uint64_t)(x ^ y);
	uint64_t ones = 0ULL;

	// count the ones (differing bits)
	// until xor is 0
	while (xor) {
		// iteratively count a bit
		++ones;
		// then subtract 1 and perform bitwise & to update value
		xor &= xor - 1ULL;
	}

	return ones;
}

// step 2.
int*** hamming_cost(int H, int W, uint64_t** left, uint64_t** right, int range) {
	int*** C; // cost array H x W x D
	C = new int** [H];
	for (int i = 0; i < H; i++) {
		C[i] = new int* [W];
		for (int j = 0; j < W; j++) {
			C[i][j] = new int[range]();
		}
	}

	// for each pixel compute hamming distance to the left in interval [0, range]
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			for (int d = 0; d < range; d++) {
				// if inside the image
				if (j - d > 0) {
					C[i][j][d] = hamming_distance(left[i][j], right[i][j - d]);
					//cout << C[i][j][d] << " ";
				}
			}
		}
	}

	return C;
}

// no longer used. optimized version below
uint64_t sum_costs(int*** C, int i, int j, int w, int d) {
	uint64_t sum = 0;

	for (int x = i - w; x <= i + w; x++) {
		for (int y = j - w; y <= j + w; y++) {
			sum += C[x][y][d];
		}
	}

	return sum;
}

// step 3.
uint64_t*** preprocess(int*** C, int H, int W, int D) {
	uint64_t*** sum;
	sum = new uint64_t * *[H];
	for (uint64_t i = 0; i < H; i++) {
		sum[i] = new uint64_t * [W];
		for (uint64_t j = 0; j < W; j++) {
			sum[i][j] = new uint64_t[D]();
		}
	}

	// process all d's for i,j = 0 (first pixel)
	for (int d = 0; d < D; d++)
		sum[0][0][d] = C[0][0][d];

	// process first row, all d's (first row except first pixel)
	for (int j = 1; j < W; j++)
		for (int d = 0; d < D; d++)
			sum[0][j][d] = C[0][j][d] + sum[0][j - 1][d];

	// process first column, all d's (first column except first pixel)
	for (int i = 1; i < H; i++)
		for (int d = 0; d < D; d++)
			sum[i][0][d] = C[i][0][d] + sum[i - 1][0][d];

	// process the rest
	for (int i = 1; i < H; i++) {
		for (int j = 1; j < W; j++) {
			for (int d = 0; d < D; d++) {
				sum[i][j][d] = C[i][j][d] + sum[i - 1][j][d] + sum[i][j - 1][d] - sum[i - 1][j - 1][d];
			}
		}
	}

	return sum;
}

// step 3. helper function. optimized constant time sum computation
uint64_t submatrix_sum(int p, int q, int r, int s, int d) {
	// pq top left, rs bottom right
	uint64_t total = processed_sums[r][s][d];

	if (q - 1 >= 0)
		total -= processed_sums[r][q - 1][d];
	if (p - 1 >= 0)
		total -= processed_sums[p - 1][s][d];
	if (p - 1 >= 0 && q - 1 >= 0)
		total += processed_sums[p - 1][q - 1][d];

	return total;
}

// step 3.
int*** hamming_sum(int*** C, int H, int W, int D, int w) {
	int*** S; // cost array H x W x D

	S = new int** [H];
	for (int i = 0; i < H; i++) {
		S[i] = new int* [W];
		for (int j = 0; j < W; j++) {
			S[i][j] = new int[D]();
			for (int d = 0; d < D; d++) {
				S[i][j][d] = C[i][j][d];
			}
		}
	}

	int p, q, r, s;

	// compute hamming sums for [-w,+w] range
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			for (int d = 0; d < D; d++) {
				// w^2 version
				//S[i][j][d] = sum_costs(C, i, j, w, d);

				// constant time version

				if (i - w <= 0)
					p = 0;
				else
					p = i - w;

				if (j - w <= 0)
					q = 0;
				else
					q = j - w;

				// check if inside. if not compute for righmost value
				if (i + w >= H)
					r = i + (H - i - 1);
				else
					r = i + w;

				// check if inside. if not compute for bottommost value
				if (j + w >= W)
					s = j + (W - j - 1);
				else
					s = j + w;

				S[i][j][d] = submatrix_sum(p, q, r, s, d);
			}
		}
	}

	return S;
}

// step 4. compute the disparity maps
Mat_<uchar> disparity_map(int*** S, int H, int W, int D) {
	Mat_<uchar> dst(H, W);
	int d, disparity;
	int val;
	int max = 0;

	// for each pixel find minimum disparity based on hamming sums
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			int minimum = MAXINT;
			disparity = 0;
			for (int d = 0; d < D; d++) {
				if (S[i][j][d] < minimum) {
					minimum = S[i][j][d];
					disparity = d;
					if (d >= max)
						max = d;
				}
			}
			 dst(i, j) = disparity;
			// normalize for visualisation
			// dst(i, j) = disparity * 255 / D;
		}
	}

	// optionally apply a median filter
	//dst = median_filter(3, dst);

	return dst;
}

// no longer used. better version above
Mat_<uchar> disparities(Mat_<uchar> left, Mat_<uchar> right) {
	Mat_<uchar> dst(left.rows, left.cols);
	int d, disparity;
	int val, D = 75;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < left.rows; i++) {
		for (int j = 0; j < left.cols; j++) {
			uint64_t minimal = MAXUINT64;
			disparity = 0;
			for (int d = 0; d <= D; d++) {
				val = 0;
				for (int k = 0; k < 8; k++) {
					if (isInside(right, i + di[k], j + dj[k] - d) && isInside(left, i + di[k], j + dj[k])) {
						val += hamming_distance(left(i + di[k], j + dj[k]), right(i + di[k], j + dj[k] - d));
					}
				}
				if (val < minimal) {
					minimal = val;
					disparity = d;
				}
			}
			dst(i, j) = disparity * 255 / D;
			//cout << val << " ";
		}
	}

	dst = median_filter(3, dst);
	dst = median_filter(3, dst);

	return dst;
}


void error(Mat_<uchar> img, Mat_<uchar> res, int T) {
	int total_pixels = img.rows * img.cols;
	int different_pixels = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j)/4 < res(i, j) - T || img(i, j)/4 > res(i, j) + T)
				different_pixels++;
		}
	}

	float percentage = different_pixels * 100 / total_pixels;

	cout << "error w.r.t ground truth: " << percentage << "%" << endl;
}

int main() {

	/*******************************************************************************/
	//PROJECT 

	// teddy
	Mat_<uchar> img_left = imread("Images/Data Set/teddy/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img_right = imread("Images/Data Set/teddy/im6.png", CV_LOAD_IMAGE_GRAYSCALE);


	// cones
	//Mat_<uchar> img_left = imread("Images/Data Set/cones/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img_right = imread("Images/Data Set/cones/im6.png", CV_LOAD_IMAGE_GRAYSCALE);


	// barn1
	//Mat_<uchar> img_left = imread("Images/Data Set/barn1/im2.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img_right = imread("Images/Data Set/barn1/im6.ppm", CV_LOAD_IMAGE_GRAYSCALE);

	// art
	//Mat_<uchar> img_left = imread("Images/Data Set/art/view1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img_right = imread("Images/Data Set/art/view5.png", CV_LOAD_IMAGE_GRAYSCALE);

	// tsukuba
	//Mat_<uchar> img_left = imread("Images/Data Set/tsukuba/scene3.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img_right = imread("Images/Data Set/tsukuba/scene5.ppm", CV_LOAD_IMAGE_GRAYSCALE);

	// poster
	//Mat_<uchar> img_left = imread("Images/Data Set/poster/im2.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img_right = imread("Images/Data Set/poster/im6.ppm", CV_LOAD_IMAGE_GRAYSCALE);

	//imshow("img left", img_left);
	//imshow("img right", img_right);

	// set distance on which to compute hamming costs
	int D = 60;
	// set distance on which to compute hamming sums
	int w = 5;

	// get current time
	double t1 = (double)getTickCount();

	// step 1. apply census transform
	uint64_t** left = census_transform(img_left, 8);
	uint64_t** right = census_transform(img_right, 8);

	//imshow("census left", left);
	//imshow("census right", right);

	// step 2. compute hamming costs
	int*** cost = hamming_cost(img_left.rows, img_left.cols, left, right, D);

	// preprocess all hamming cost sums to optimize time (global variable)
	processed_sums = preprocess(cost, img_left.rows, img_left.cols, D);

	// step 3. compute hamming sums
	int*** sum = hamming_sum(cost, img_left.rows, img_left.cols, D, w);

	// step 4. compute disparity map
	Mat_<uchar> disparity = disparity_map(sum, img_left.rows, img_left.cols, D);
	imshow("disparity", disparity);

	// measure time
	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	printf("Time for depth estimation algorithm = %.3f [ms]\n", t1 * 1000);

	// teddy
	Mat_<uchar> ground_truth = imread("Images/Data Set/teddy/disp2.png", CV_LOAD_IMAGE_GRAYSCALE);

	// cones
	//Mat_<uchar> ground_truth = imread("Images/Data Set/cones/disp2.png", CV_LOAD_IMAGE_GRAYSCALE);

	// barn1
	//Mat_<uchar> ground_truth = imread("Images/Data Set/barn1/disp2.pgm", CV_LOAD_IMAGE_GRAYSCALE);

	// art
	//Mat_<uchar> ground_truth = imread("Images/Data Set/art/disp1.png", CV_LOAD_IMAGE_GRAYSCALE);

	// tsukuba
	//Mat_<uchar> ground_truth = imread("Images/Data Set/tsukuba/disp3.pgm", CV_LOAD_IMAGE_GRAYSCALE);

	// poster
	//Mat_<uchar> ground_truth = imread("Images/Data Set/poster/disp2.pgm", CV_LOAD_IMAGE_GRAYSCALE);

	imshow("ground truth", ground_truth);

	int T = 4;
	error(ground_truth, disparity, T);

	//imwrite("Images/Data Set/teddy/teddy_D60_w5.jpg",disparity);
	//imwrite("Images/Data Set/cones/cones_D60_w5.jpg", disparity);
	//imwrite("Images/Data Set/barn1/barn1_D30_w5.jpg", disparity);
	//imwrite("Images/Data Set/art/art_D85_w6.jpg", disparity);
	//imwrite("Images/Data Set/tsukuba/tsukuba_D30_w6.jpg", disparity);
	//imwrite("Images/Data Set/poster/poster_D30_w6.jpg", disparity);

	waitKey();

	/*******************************************************************************/
}