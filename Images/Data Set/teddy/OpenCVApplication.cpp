// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <queue>
#include <random>
using namespace cv;
using namespace std;


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

void negative_image() {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img2(img.rows, img.cols, CV_8UC1);
	Mat_<uchar> img2(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//img2.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
			img2(i, j) = 255 - img(i, j);
		}
	}
	imshow("image", img);
	imshow("negative image", img2);
	waitKey(0);
}

void additive_image(int x) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) + x < 0)
				img2(i, j) = 0;
			else if (img(i, j) + x > 255)
				img2(i, j) = 255;
			else
				img2(i, j) = img(i, j) + x;
		}
	}
	imshow("image", img);
	imshow("additive image", img2);
	waitKey(0);
}

void multiplicative_image(float x) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) * x < 0)
				img2(i, j) = 0;
			else if (img(i, j) * x > 255)
				img2(i, j) = 255;
			else
				img2(i, j) = img(i, j) * x;
		}
	}
	imshow("image", img);
	imshow("multiplicative image", img2);
	imwrite("Images/mimage.bmp", img2);
	waitKey(0);
}

void color_image() {
	Mat_<Vec3b> img(256, 256);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (i < 128 && j < 128)
			{
				img(i, j)[0] = 255; //blue
				img(i, j)[1] = 255; //green
				img(i, j)[2] = 255; //red
			}
			else if (i < 128 && j >= 128)
			{
				img(i, j)[0] = 0; //blue
				img(i, j)[1] = 0; //green
				img(i, j)[2] = 255; //red
			}
			else if (i >= 128 && j < 128)
			{
				img(i, j)[0] = 0; //blue
				img(i, j)[1] = 255; //green
				img(i, j)[2] = 0; //red
			}
			else
			{
				img(i, j)[0] = 0; //blue
				img(i, j)[1] = 255; //green
				img(i, j)[2] = 255; //red
			}
		}
	}
	imshow("color_image", img);
	waitKey(0);
}

void float_mat()
{
	float vals[9] = { 2,1,1,3,2,1,2,1,2 };
	Mat_<float> img(3, 3, vals);
	std::cout << img << std::endl;
	std::cout << img.inv() << std::endl;
	std::cout << img * img.inv() << std::endl;
}

void rotate_image() {
	Mat_<uchar> img = imread("Images/moon.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2(img.cols, img.rows);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img2(img2.rows - 1 - j, i) = img(i, j);
		}
	}
	imshow("image", img);
	imshow("rotated image", img2);
	waitKey(0);
}

void rgb_separation() {
	Mat_<Vec3b> img = imread("Images/flowers_24bits.bmp");
	Mat_<uchar> r_ch(img.rows, img.cols);
	Mat_<uchar> g_ch(img.rows, img.cols);
	Mat_<uchar> b_ch(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			r_ch(i, j) = img(i, j)[2];
			g_ch(i, j) = img(i, j)[1];
			b_ch(i, j) = img(i, j)[0];
		}
	}
	imshow("image", img);
	imshow("red", r_ch);
	imshow("green", g_ch);
	imshow("blue", b_ch);
	waitKey(0);
}

void color_to_grayscale() {
	Mat_<Vec3b> img = imread("Images/flowers_24bits.bmp");
	Mat_<uchar> grayscale(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			grayscale(i, j) = (img(i, j)[0] + img(i, j)[1] + img(i, j)[2]) / 3;
		}
	}
	imshow("image", img);
	imshow("grayscale", grayscale);
	waitKey(0);
}

void grayscale_to_binary(int threshold) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> bin(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) < threshold)
				bin(i, j) = 0;
			else
				bin(i, j) = 255;
		}
	}
	imshow("image", img);
	imshow("binary", bin);
	waitKey(0);
}

float min3(float x, float y, float z) {
	if (x < y)
		if (x < z)
			return x;
		else
			return z;
	else
		if (y < z)
			return y;
		else
			return z;
}

float max3(float x, float y, float z) {
	if (x > y)
		if (x > z)
			return x;
		else
			return z;
	else
		if (y > z)
			return y;
		else
			return z;
}

void rgb_to_hsv() {
	Mat_<Vec3b> img = imread("Images/flowers_24bits.bmp");
	Mat_<Vec3b> hsv(img.rows, img.cols);
	Mat_<Vec3b> rgb(img.rows, img.cols);
	float r, g, b, M, m, C, H, S, V;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			r = (float)img(i, j)[2] / 255;
			g = (float)img(i, j)[1] / 255;
			b = (float)img(i, j)[0] / 255;
			M = max3(r, g, b);
			m = min3(r, g, b);
			C = M - m;
			//value
			V = M;
			//saturation
			if (V != 0)
				S = C / V;
			else
				S = 0;
			//hue
			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else
				H = 0;
			if (H < 0)
				H = H + 360;
			//hsv image
			hsv(i, j)[0] = H / 2;
			hsv(i, j)[1] = S * 255;
			hsv(i, j)[2] = V * 255;
		}
	}
	cvtColor(hsv, rgb, COLOR_HSV2BGR);
	imshow("image", img);
	imshow("HSV", hsv);
	imshow("RGB", rgb);
	//imshow("H", h);
	//imshow("S", s);
	//imshow("V", v);
	waitKey(0);
}

bool isInside(Mat img, int i, int j) {
	if (i < img.rows && i >= 0 && j < img.cols && j >= 0)
		return true;
	return false;
}

void compute_histogram(int* hist) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//int hist[256] = {};
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img(i, j)]++;
		}
	}

	/*for (int i = 0; i < 256; i++) {
		std::cout << hist[i] << " ";
	}
	imshow("image", img);
	waitKey(0);*/
}

void compute_pdf(int* hist, float* pdf) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//int hist[256] = {};
	//float pdf[256] = {};
	int M = img.rows * img.cols;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img(i, j)]++;
		}
	}

	//std::cout << "histogram: ";
	for (int i = 0; i < 256; i++) {
		//std::cout << hist[i] << " ";
		pdf[i] = (float)hist[i] / M;
	}

	/*std::cout << std::endl << std::endl;
	std::cout << "pdf: ";
	float sum = 0;
	for (int i = 0; i < 256; i++) {
		std::cout << pdf[i] << " ";
		sum += pdf[i];
	}
	std::cout << std::endl << std::endl << "sum is: " << sum;

	imshow("image", img);
	waitKey(0);*/
}

void compute_histo_bins(int m, int* hist) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//std::cout << "histo on " << m << "bins: ";
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			(hist[img(i, j) / (256 / m)])++;
			//std::cout << hist2[i] << " ";
		}
	}
}

void multilevel_thresholding(int WH, float TH) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2(img.rows, img.cols);

	std::vector<int> maxima_vec;
	maxima_vec.push_back(0);
	float pdf[256] = {};
	int hist[256] = {};
	compute_pdf(hist, pdf);
	//showHistogram("histo", hist, 256, 200);
	//waitKey(0);

	//int WH = 5;
	//float TH = 0.0003;

	for (int k = 0 + WH; k <= 255 - WH; k++) {
		float v = 0;
		float max = pdf[k];
		for (int i = k - WH; i <= k + WH; i++) {
			v += pdf[i];
			if (pdf[i] > max)
				max = pdf[i];
		}
		v = v / (2 * WH + 1);
		if ((pdf[k] > v + TH) && (pdf[k] >= max))
		{
			maxima_vec.push_back(k);
			std::cout << k << " ";
		}
	}
	maxima_vec.push_back(255);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float absolute = 100000;
			float value;
			for (std::vector<int>::iterator it = maxima_vec.begin(); it != maxima_vec.end(); it++) {
				if (absolute > abs(*it - img(i, j))) {
					absolute = abs(*it - img(i, j));
					value = *it;
				}
			}
			img2(i, j) = value;
		}
	}

	imshow("image", img);
	imshow("image2", img2);
	waitKey(0);
}

void onMouse(int event, int x, int y, int flags, void* param) {
	if (event == EVENT_LBUTTONDOWN) {
		Mat_<Vec3b> img = *(Mat_<Vec3b>*)param;
		//y - row, x - col
		Vec3b color = img(y, x);

		int area = 0;
		double ri = 0;
		double ci = 0;

		double nom = 0;
		double denom = 0;
		double phi = 0;
		int perimeter = 0;
		bool contour = false;
		double T = 0;
		int imin = 100000, imax = 0, jmin = 100000, jmax = 0;
		double R = 0;
		int* h, * v;
		h = (int*)malloc(sizeof(int) * img.rows + 1);
		v = (int*)malloc(sizeof(int) * img.cols + 1);
		for (int i = 0; i < img.rows; i++)
			h[i] = 0;
		for (int j = 0; j < img.cols; j++)
			v[j] = 0;

		//show features on separate image
		Mat_<Vec3b> clone = img.clone();
		Mat_<Vec3b> img2(img.rows, img.cols);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img(i, j) == color) {
					//area
					area++;

					//center of mass
					ri += i;
					ci += j;

					//perimeter
					//down
					contour = false;
					if (isInside(img, i - 1, j) && img(i - 1, j) != color || !isInside(img, i - 1, j))
						contour = true;
					//up
					if (isInside(img, i + 1, j) && img(i + 1, j) != color || !isInside(img, i + 1, j))
						contour = true;
					//left
					if (isInside(img, i, j - 1) && img(i, j - 1) != color || !isInside(img, i, j - 1))
						contour = true;
					//right
					if (isInside(img, i, j + 1) && img(i, j + 1) != color || !isInside(img, i, j + 1))
						contour = true;
					//down left
					if (isInside(img, i - 1, j - 1) && img(i - 1, j - 1) != color || !isInside(img, i - 1, j - 1))
						contour = true;
					//down right
					if (isInside(img, i - 1, j + 1) && img(i - 1, j + 1) != color || !isInside(img, i - 1, j + 1))
						contour = true;
					//up left
					if (isInside(img, i + 1, j - 1) && img(i + 1, j - 1) != color || !isInside(img, i + 1, j - 1))
						contour = true;
					//up right
					if (isInside(img, i + 1, j + 1) && img(i + 1, j + 1) != color || !isInside(img, i + 1, j + 1))
						contour = true;
					//contour drawing and perimeter computation
					if (contour) {
						perimeter++;
						clone(i, j)[0] = 0;
						clone(i, j)[1] = 0;
						clone(i, j)[2] = 0;
					}

					//aspect ratio
					if (i < imin)
						imin = i;
					if (i > imax)
						imax = i;
					if (j < jmin)
						jmin = j;
					if (j > jmax)
						jmax = j;

					//projection
					img2(i, h[i]) = color;
					img2(v[j], j) = color;
					h[i]++;
					v[j]++;
				}
			}
		}

		ri /= area;
		ci /= area;

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img(i, j) == color) {
					//axis of elongation
					nom += (i - ri) * (j - ci);
					denom += (j - ci) * (j - ci) - (i - ri) * (i - ri);
				}
			}
		}
		phi = atan2(2 * nom, denom) / 2;

		//thiness ratio
		T = 4 * PI * (area / (double)(perimeter * perimeter));

		//aspect ratio
		R = (jmax - jmin + 1) / (double)(imax - imin + 1);

		std::cout << "Center of mass: " << ri << " " << ci << endl;
		std::cout << "Area: " << area << endl;
		std::cout << "Angle of axis of elongation: " << phi << endl;
		std::cout << "Perimeter: " << perimeter << endl;
		std::cout << "Thiness ratio: " << T << endl;
		std::cout << "Aspect ratio: " << R << endl;
		std::cout << endl;

		line(clone, Point(ci - 10, ri), Point(ci + 10, ri), Vec3b(0, 0, 0));
		line(clone, Point(ci, ri - 10), Point(ci, ri + 10), Vec3b(0, 0, 0));
		//axis of elongation
		line(clone, Point(ci, ri), Point(ci + cos(phi) * 100, ri + sin(phi) * 100), Vec3b(0, 0, 0), 1);
		//aspect ratio
		line(clone, Point(jmin, imin), Point(jmin, imax), Vec3b(255, 255, 0));
		line(clone, Point(jmax, imin), Point(jmax, imax), Vec3b(255, 255, 0));
		line(clone, Point(jmin, imin), Point(jmax, imin), Vec3b(255, 255, 0));
		line(clone, Point(jmin, imax), Point(jmax, imax), Vec3b(255, 255, 0));

		imshow("geom. features", clone);
		imshow("projection", img2);
		waitKey();
	}
}

void label_to_img(Mat_<int> labels, int label) {
	default_random_engine gen;
	uniform_int_distribution<int> d(0, 255);

	Mat_<Vec3b> img(labels.rows, labels.cols);
	Vec3b* color = (Vec3b*)malloc(sizeof(Vec3b) * (label + 1));

	for (int i = 0; i < label; i++) {
		color[i][0] = d(gen);
		color[i][1] = d(gen);
		color[i][2] = d(gen);
	}

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (labels(i, j) == 0) {
				img(i, j)[0] = 255;
				img(i, j)[1] = 255;
				img(i, j)[2] = 255;
			}
			else {
				img(i, j) = color[labels(i, j) - 1];
			}
		}
	}
	imshow("label image", img);
}

void bfs(int N) {
	if (!(N == 4 || N == 8))
		return;
	int di[8] = { -1,0,1,0,-1,-1,1,1 };
	int dj[8] = { 0,-1,0,1,-1,1,-1,1 };

	Mat_<uchar> img = imread("Images/lab5/letters.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int label = 0;
	Mat_<int> labels(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			labels(i, j) = 0;
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {
				label++;
				queue<Point> Q;
				labels(i, j) = label;
				Q.push(Point(i, j));
				while (!Q.empty()) {
					Point q = Q.front();
					Q.pop();
					for (int k = 0; k < N; k++) {
						int i2 = q.x + di[k];
						int j2 = q.y + dj[k];
						if (isInside(img, i2, j2)) {
							if (img(i2, j2) == 0 && labels(i2, j2) == 0) {
								labels(i2, j2) = label;
								Q.push(Point(i2, j2));
							}
						}
					}
				}
			}
		}
	}
	label_to_img(labels, label);
	imshow("image", img);
	waitKey(0);
}

void two_pass() {
	int di[4] = { -1,0,-1,-1 };
	int dj[4] = { 0,-1,-1,1 };

	Mat_<uchar> img = imread("Images/lab5/letters.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int label = 0;
	Mat_<int> labels(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			labels(i, j) = 0;
		}
	}

	vector<vector<int>> edges(img.rows * img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {
				vector<int> L;
				for (int k = 0; k < 4; k++) {
					int i2 = i + di[k];
					int j2 = j + dj[k];
					if (isInside(img, i2, j2))
						if (labels(i2, j2) > 0)
							L.push_back(labels(i2, j2));
				}
				if (L.size() == 0) {
					label++;
					labels(i, j) = label;
				}
				else {
					int x = 999999;
					for (int k = 0; k < L.size(); k++) {
						if (L[k] < x)
							x = L[k];
					}
					labels(i, j) = x;
					for (int y : L) {
						if (y != x) {
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}
			}
		}
	}

	label_to_img(labels, label);
	waitKey();

	int newLabel = 0;
	int* newLabels = (int*)malloc(sizeof(int) * (label + 1));
	for (int i = 0; i < (label + 1); i++) {
		newLabels[i] = 0;
	}

	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);
			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			labels(i, j) = newLabels[labels(i, j)];
		}
	}

	label_to_img(labels, label);
	imshow("image", img);
	waitKey(0);
}

void border_tracing() {
	Mat_<uchar> img = imread("Images/lab6/triangle_up.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2(img.rows, img.cols);

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int dir = 7; //previous direction
	int dir0;
	vector<Point> pts;
	vector<int> dirs;
	vector<int>deriv;
	int posi, posj;
	int ok = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				posi = i;
				posj = j;
				pts.push_back(Point(posi, posj));
				ok = 1;
				break;
			}
		}
		if (ok) break;
	}

	while (1) {
		//starting direction
		if (dir % 2 == 1)
			dir0 = (dir + 6) % 8;
		else
			dir0 = (dir + 7) % 8;

		for (int k = 0; k < 8; k++) {
			int dir_now = (dir0 + k) % 8;
			int i2 = posi + di[dir_now];
			int j2 = posj + dj[dir_now];

			if (img(i2, j2) == 0) {
				pts.push_back(Point(i2, j2));
				dirs.push_back(dir_now);
				dir = dir_now;
				posi = i2;
				posj = j2;
				break;
			}
		}
		int size = pts.size();
		if (size > 2) {
			if (pts[0] == pts[size - 2] && pts[1] == pts[size - 1]) {
				break;
			}
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img2(i, j) = 255;
		}
	}

	for (int i = 0; i < pts.size() - 1; i++) {
		img2(pts[i].x, pts[i].y) = 0;
	}

	//chain codes extraction
	int val;
	int size_dir = dirs.size();
	std::cout << "code:	";
	for (int i = 0; i < size_dir - 1; i++) {
		val = (dirs[i + 1] - dirs[i]) % 8;
		if (val < 0)
			val += 8;
		deriv.push_back(val);

		std::cout << dirs[i] << " ";
	}
	val = (dirs[0] - dirs[size_dir - 1]) % 8;
	if (val < 0)
		val += 8;
	deriv.push_back(val);
	std::cout << dirs[size_dir - 1] << endl;

	std::cout << "derivative:	";
	int size_deriv = deriv.size();
	for (int i = 0; i < size_deriv; i++) {
		std::cout << deriv[i] << " ";
	}

	imshow("image", img);
	imshow("contour", img2);
	waitKey(0);
}

void reconstruct() {
	Mat_<uchar> img = imread("Images/lab6/gray_background.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	std::ifstream myfile("Images/lab6/reconstruct.txt");
	int i, j;
	int x;
	std::vector<int> dirs;
	if (myfile.is_open())
	{
		myfile >> i >> j >> x;
		while (myfile >> x) {
			dirs.push_back(x);
		}
		myfile.close();
	}

	img(i, j) = 0;
	int size = dirs.size();

	for (int n = 0; n < size; n++) {
		i += di[dirs[n]];
		j += dj[dirs[n]];
		img(i, j) = 0;
	}
	imshow("reconstructed image", img);
	waitKey(0);
}

Mat_<uchar> dilate(Mat_<uchar> img, Mat_<uchar> B) {
	Mat_<uchar> dst = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//if img(i, j) is black
			if (img(i, j) == 0) {
				for (int u = 0; u < B.rows; u++) {
					for (int v = 0; v < B.cols; v++) {
						if (isInside(img, i + u - B.rows / 2, j + v - B.cols / 2) && B(u, v) == 0) {
							dst(i + u - B.rows / 2, j + v - B.cols / 2) = 0;
						}
					}
				}
			}
		}
	}

	return dst;
}

Mat_<uchar> erode(Mat_<uchar> img, Mat_<uchar> B) {
	Mat_<uchar> dst = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//if img(i ,j) is black
			if (img(i, j) == 0) {
				int ok = 0;
				for (int u = 0; u < B.rows; u++) {
					for (int v = 0; v < B.cols; v++) {
						if (isInside(img, i + u - B.rows / 2, j + v - B.cols / 2) && B(u, v) == 0) {
							if (img(i + u - B.rows / 2, j + v - B.cols / 2) != 0)
								ok = 1;
						}
					}
				}
				if (ok)
					dst(i, j) = 255;
			}
		}
	}

	return dst;
}

Mat_<uchar> open(Mat_<uchar> img, Mat_<uchar> B) {
	Mat_<uchar> dst = img.clone();

	dst = erode(dst, B);
	dst = dilate(dst, B);

	return dst;
}

Mat_<uchar> close(Mat_<uchar> img, Mat_<uchar> B) {
	Mat_<uchar> dst = img.clone();

	dst = dilate(dst, B);
	dst = erode(dst, B);

	return dst;
}

Mat_<uchar> repeat_alg(Mat_<uchar> img, Mat_<uchar> B, int n, Mat_<uchar>(*f)(Mat_<uchar>, Mat_<uchar>)) {
	Mat_<uchar> dst = img.clone();
	Mat_<uchar> buffer = img.clone();

	for (int i = 0; i < n; i++) {
		buffer = (*f)(dst, B);
		dst = buffer;
	}

	return dst;
}

Mat_<uchar> boundary_extraction(Mat_<uchar> img, Mat_<uchar> B) {
	Mat_<uchar> dst = img.clone();

	Mat_<uchar> eroded = erode(img, B);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			dst(i, j) = 255;
			if (img(i, j) == 0 && eroded(i, j) == 255)
				dst(i, j) = 0;
		}
	}

	return dst;
}

Mat_<uchar> dilate_intersect(Mat_<uchar> prev, Mat_<uchar> B, Mat_<uchar> complement) {
	Mat_<uchar> current(prev.rows, prev.cols);

	//dilate 
	Mat_<uchar> dilated = dilate(prev, B);

	//intersect dilation with complement
	for (int i = 0; i < prev.rows; i++) {
		for (int j = 0; j < prev.cols; j++) {
			if (complement(i, j) == 0 && dilated(i, j) == 0)
				current(i, j) = 0;
			else
				current(i, j) = 255;
		}
	}

	return current;
}

int equal_images(Mat_<uchar> img1, Mat_<uchar> img2) {
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			if (img1(i, j) != img2(i, j))
				return 0;
		}
	}
	return 1;
}

Mat_<uchar> region_filling(Mat_<uchar> img, Mat_<uchar> B) {
	Mat_<uchar> complement = img.clone();
	Mat_<uchar> prev = img.clone();
	Mat_<uchar> current = img.clone();
	int ok = 1;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//create complement
			if (img(i, j) == 0)
				complement(i, j) = 255;
			else
				complement(i, j) = 0;
			prev(i, j) = 255;
			if (isInside(img, i - 1, j) && isInside(img, i, j - 1)) {
				if (ok && img(i, j) == 255 && img(i - 1, j) == 0 && img(i, j - 1) == 0) {
					ok = 0;
					prev(i, j) = 0;
				}
			}
		}
	}

	while (1) {
		current = dilate_intersect(prev, B, complement);
		if (equal_images(prev, current)) {
			break;
		}
		prev = current;
	}

	return prev;
}

void calchist(Mat_<uchar> img, int* hist, float* pdf) {
	int M = img.rows * img.cols;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img(i, j)]++;
		}
	}

	for (int i = 0; i < 256; i++) {
		pdf[i] = (float)hist[i] / M;
	}
}

double mean(int M, int* hist) {
	double mean = 0.0;
	for (int g = 0; g < 256; g++) {
		mean += (double)g * (double)hist[g];
	}
	mean /= M;
	return mean;
}

double standard_deviation(int M, int* hist, float* pdf) {
	double sigma = 0.0;
	double mean_val = mean(M, hist);
	for (int g = 0; g < 256; g++) {
		sigma += (g - mean_val) * (g - mean_val) * (double)pdf[g];
	}
	sigma = sqrt(sigma);
	return sigma;
}

int compute_threshold(Mat_<uchar> img) {
	int hist[256] = {};
	float pdf[256];
	calchist(img, hist, pdf);

	double imin = 0.0;
	double imax = 0.0;
	int ok = 1;
	double t, t2;
	double error = 0.1;

	for (int g = 0; g < 256; g++) {
		if (hist[g] > 0) {
			imax = g;
			if (ok) {
				imin = g;
				ok = 0;
			}
		}
	}
	t = (imin + imax) / 2.0;

	double mean1, mean2;
	double n1, n2;

	while (1) {
		n1 = 0.0;
		n2 = 0.0;
		mean1 = 0.0;
		mean2 = 0.0;

		for (int g = imin; g < imax; g++) {
			if (g <= t) {
				//compute n1
				n1 += hist[g];
				//compute mean1
				mean1 += g * (double)hist[g];
			}
			else {
				//compute n2
				n2 += hist[g];
				//compute mean2
				mean2 += g * (double)hist[g];
			}
		}

		mean1 /= n1;
		mean2 /= n2;

		t2 = (mean1 + mean2) / 2.0;
		if (abs(t2 - t) < error) {
			t = t2;
			break;
		}
		t = t2;
	}

	t = round(t);

	Mat_<uchar> img2(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) <= t)
				img2(i, j) = 0;
			else
				img2(i, j) = 255;
		}
	}

	imshow("initial image", img);
	imshow("thresholded image", img2);
	waitKey();

	return t;
}

void shrink_stretch(Mat_<uchar> img, int gout_min, int gout_max) {
	//compute histogram
	int hist[256] = {};
	float pdf[256];
	calchist(img, hist, pdf);

	//get gin min and max
	double gin_min = 0.0;
	double gin_max = 0.0;
	int ok = 1;

	for (int g = 0; g < 256; g++) {
		if (hist[g] > 0) {
			gin_max = g;
			if (ok) {
				gin_min = g;
				ok = 0;
			}
		}
	}

	//stretch/shrink
	int val;
	Mat_<uchar> img2(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			val = round((gout_min + (img(i, j) - gin_min) * (double)((gout_max - (double)gout_min) / (double)(gin_max - gin_min))));
			if (val > 255)
				img2(i, j) = 255;
			else if (val < 0)
				img2(i, j) = 0;
			else
				img2(i, j) = val;
		}
	}

	int hist2[256] = {};
	float pdf2[256];
	calchist(img2, hist2, pdf2);

	//show initial histogram
	showHistogram("initial histogram", hist, 256, 200);
	//show new histogram
	showHistogram("new histogram", hist2, 256, 200);

	imshow("initial image", img);
	imshow("thresholded image", img2);
	waitKey();
}

void gamma_correction(Mat_<uchar> img, float gamma) {
	Mat_<uchar> img2(img.rows, img.cols);
	double val;
	double l = 255;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			val = round((pow((img(i, j) / (double)l), gamma) * l));
			if (val > 255)
				img2(i, j) = 255;
			else if (val < 0)
				img2(i, j) = 0;
			else
				img2(i, j) = val;
		}
	}

	imshow("initial image", img);
	imshow("gamma corrected image", img2);
	waitKey();
}

void compute_cpdf(Mat_<uchar> img, float* cpdf) {
	int hist[256] = {};
	float pdf[256];
	calchist(img, hist, pdf);

	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < i + 1; j++) {
			cpdf[i] += pdf[j];
		}
	}
}

void histogram_equalization(Mat_<uchar> img) {
	Mat_<uchar> img2(img.rows, img.cols);

	//compute pdf
	int hist[256] = {};
	float pdf[256];
	calchist(img, hist, pdf);

	//compute cpdf
	float cpdf[256] = {};
	compute_cpdf(img, cpdf);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img2(i, j) = round(((double)255 * cpdf[img(i, j)]));
		}
	}

	int hist2[256] = {};
	float pdf2[256];
	calchist(img2, hist2, pdf2);

	//show initial histogram
	showHistogram("initial histogram", hist, 256, 200);
	//show new histogram
	showHistogram("equalized histogram", hist2, 256, 200);

	imshow("initial image", img);
	imshow("equalized image", img2);
	waitKey();
}

void showHistogramFloat(const std::string& name, float* hist, const int  hist_cols, const int hist_height)
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

Mat_<uchar> census_transform(Mat_<uchar> img, int size) {
	Mat_<uchar> dst(img.rows, img.cols);

	for (int i = size / 2; i < img.rows - size / 2; i++) {
		for (int j = size / 2; j < img.cols - size / 2; j++) {
			uint64_t census = 0ULL;
			for (int u = -size / 2; u <= size / 2; u++) {
				for (int v = -size / 2; v <= size / 2; v++) {
					if (isInside(img, i + u, j + v) && (u != 0 || v != 0)) {
						census <<= 1ULL;
						if (img(i + u, j + v) < img(i, j))
							census += 1ULL;
						else
							census += 0ULL;
					}
				}
			}
			//cout << census << " ";
			dst(i, j) = census;
		}
	}

	return dst;
}

int hamming_distance(uint64_t x, uint64_t y) {
	uint64_t xor = (uint64_t)(x ^ y);
	uint64_t ones = 0ULL;

	while (xor) {
		++ones;
		xor &= xor -1ULL;
	}

	return ones;
}

int*** hamming_cost(Mat_<uchar> left, Mat_<uchar> right, int range) {
	int*** C; // cost array H x W x D
	C = new int** [left.rows];
	for (int i = 0; i < left.rows; i++) {
		C[i] = new int* [left.cols];
		for (int j = 0; j < left.cols; j++) {
			C[i][j] = new int[range]();
		}
	}
	//cout << left.rows << " " << left.cols << " " << range << endl;

	for (int i = 0; i < left.rows; i++) {
		for (int j = 0; j < left.cols; j++) {
			for (int d = 0; d < range; d++) {
				if (isInside(right, i, j - d)) {
					C[i][j][d] = hamming_distance(left(i, j), right(i, j - d));
					//cout << C[i][j][d] << " ";
				}
			}
		}
	}

	return C;
}

uint64_t sum_costs(int*** C, int i, int j, int w, int d) {
	uint64_t sum = 0;

	for (int x = i - w; x < i + w; x++) {
		for (int y = j - w; y < j + w; y++) {
			sum += C[x][y][d];
		}
	}

	return sum;
}

int*** hamming_sum(int*** C, int H, int W, int D, int w) {
	int*** S; // cost array H x W x D
	/*int H = sizeof(C) / sizeof(C[0]);
	int W = sizeof(C[0]) / sizeof(C[0][0]);
	int D = sizeof(C[0][0]) / sizeof(C[0][0][0]);*/

	S = new int **[H];
	for (int i = 0; i < H; i++) {
		S[i] = new int* [W];
		for (int j = 0; j < W; j++) {
			S[i][j] = new int[D]();
			for (int d = 0; d < D; d++) {
				S[i][j][d] = C[i][j][d];
			}
		}
	}

	//cout << H << " " << W << " " << D << endl;

	for (int i = w; i < H - w; i++) {
		for (int j = w; j < W - w; j++) {
			for (int d = 0; d < D; d++) {
				S[i][j][d] = sum_costs(C, i, j, w, d);
				//cout << S[i][j][d] << " ";
			}
		}
	}

	return S;
}

Mat_<uchar> disparity_map(int*** S, int H, int W, int D) {
	Mat_<uchar> dst(H, W);
	int d, disparity;
	int val;

	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			int minimal = MAXINT;
			disparity = 0;
			for (int d = 0; d < D; d++) {
				if (S[i][j][d] < minimal) {
					minimal = S[i][j][d];
					disparity = d;
				}
			}
			dst(i, j) = disparity*255/D;
			//cout << disparity << " ";
		}
	}

	//dst = median_filter(3, dst);

	return dst;
}

Mat_<uchar> disparities(Mat_<uchar> left, Mat_<uchar> right) {
	Mat_<uchar> dst(left.rows, left.cols);
	int d, disparity;
	int val, D=75;
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
			dst(i, j) = disparity * 255/D;
			//cout << val << " ";
		}
	}

	dst = median_filter(3, dst);
	dst = median_filter(3, dst);

	return dst;
}

int main() {

	/*******************************************************************************/
	//PROJECT 

	//Mat_<uchar> img_left = imread("Images/Data Set/teddy/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img_right = imread("Images/Data Set/teddy/im6.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat_<uchar> img_left = imread("Images/Data Set/cones/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img_right = imread("Images/Data Set/cones/im6.png", CV_LOAD_IMAGE_GRAYSCALE);

	//imshow("img left", img_left);
	//imshow("img right", img_right);

	Mat_<uchar> left;
	//left = spatial_filter(img_left, gaussian_2d_filter(7));
	left = census_transform(img_left, 3);
	Mat_<uchar> right;
	//right = spatial_filter(img_right, gaussian_2d_filter(7));
	right = census_transform(img_right, 3);

	//imshow("census left", left);
	//imshow("census right", right);

	int D = 75;

	int*** cost = hamming_cost(left, right, D);
	int*** sum = hamming_sum(cost, left.rows, left.cols, D, 5);
	Mat_<uchar> disparity = disparity_map(sum, left.rows, left.cols, D);
	imshow("disparity", disparity);

	Mat_<uchar> disparity2 = disparities(left,right);
	imshow("disparity2", disparity2);


	/*******************************************************************************/


	//LAB1
	//negative_image();
	//additive_image(-50);
	//multiplicative_image(5);
	//color_image();
	//float_mat();
	//rotate_image();


	/*******************************************************************************/


	//LAB2
	//rgb_separation();
	//color_to_grayscale();
	/*int c;
	std::cin >> c;
	grayscale_to_binary(c);
	*/
	//rgb_to_hsv();


	/*******************************************************************************/


	//LAB3
	//int hist[256] = {};
	//compute_histogram(hist);
	//float pdf[256] = {};
	//compute_pdf(hist, pdf);
	//showHistogram("histogram", hist, 256, 200);
	//waitKey();
	//int hist2[128] = {};
	//compute_histo_bins(128, hist2);
	//showHistogram("histogram bins", hist2, 128, 200);
	//waitKey();
	//multilevel_thresholding(5, 0.0003);


	/*******************************************************************************/


	//LAB4
	/*Mat_<Vec3b> img = imread("Images/lab4/trasaturi_geom.bmp");
	imshow("img", img);
	cvSetMouseCallback("img", onMouse, &img);
	waitKey();
	return 0;*/


	/*******************************************************************************/


	//LAB5
	//bfs(8);
	//two_pass();


	/*******************************************************************************/


	//LAB6
	//border_tracing();
	//reconstruct();


	/*******************************************************************************/


	//LAB7
	/*uchar b8Vals[9] = { 0,0,0,0,0,0,0,0,0};
	Mat_<uchar> B8(3, 3, b8Vals);

	uchar b4Vals[9] = { 255,0,255,0,0,0,255,0,255 };
	Mat_<uchar> B4(3, 3, b4Vals);

	Mat_<uchar> img1 = imread("Images/Morphological_Op_Images/1_Dilate/wdg2ded1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2 = imread("Images/Morphological_Op_Images/2_Erode/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img3 = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img4 = imread("Images/Morphological_Op_Images/4_Close/phn1thr1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img5 = imread("Images/Morphological_Op_Images/5_BoundaryExtraction/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img6 = imread("Images/Morphological_Op_Images/6_RegionFilling/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	*/
	/*int n;
	cout << "How many times do you want to repeat the algorithms? N = ";
	cin >> n;

	Mat_<uchar> dilated;
	//dilated = dilate(img1, B4);
	dilated = repeat_alg(img1, B4, n, dilate);

	Mat_<uchar> eroded;
	//eroded = erode(img2, B4);
	eroded = repeat_alg(img2, B4, n, erode);

	Mat_<uchar> opened;
	//opened = open(img3, B4);
	opened = repeat_alg(img3, B4, n, open);

	Mat_<uchar> closed;
	//closed = close(img4, B4);
	closed = repeat_alg(img4, B4, n, close);

	imshow("to be dilated", img1);
	imshow("dilated", dilated);

	imshow("to be eroded", img2);
	imshow("eroded", eroded);

	imshow("to be opened", img3);
	imshow("opened", opened);

	imshow("to be closed", img4);
	imshow("closed", closed);*/

	/*Mat_<uchar> boundary;
	boundary = boundary_extraction(img5, B4);

	imshow("to extract boundary", img5);
	imshow("boundary extracted", boundary);*/

	/*Mat_<uchar> region;
	region = region_filling(img6, B4);

	imshow("to fill region", img6);
	imshow("region filled", region);*/


	/*******************************************************************************/


	//LAB8
	/*int hist[256] = {};
	float pdf[256];
	Mat_<uchar> img = imread("Images/lab8/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int M = img.rows * img.cols;
	calchist(img, hist, pdf);

	//EX1

	//display histogram
	//showHistogram("histogram", hist, 256, 200);

	//mean value
	double mean_val = mean(M, hist);
	cout << "mean value: "<< mean_val << endl;

	//standard deviation
	double sigma = standard_deviation(M, hist, pdf);
	cout << "standard deviation: " << sigma << endl;

	//show cpdf
	Mat_<uchar> img8 = imread("Images/lab8/wheel.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float cpdf[256] = {};
	compute_cpdf(img8, cpdf);
	showHistogramFloat("cpdf", cpdf, 256, 200);


	//EX2

	//threshold
	Mat_<uchar> img2 = imread("Images/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int threshold = compute_threshold(img2);
	cout << "threshold: " << threshold << endl;*/


	//EX3

	//stretching/shrinking
	/*Mat_<uchar> img3 = imread("Images/lab8/Hawkes_Bay_NZ.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img4 = imread("Images/lab8/wheel.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int gout_min, gout_max;
	cout << "gout min value = " << endl;
	cin >> gout_min;
	cout << "gout max value = " << endl;
	cin >> gout_max;
	shrink_stretch(img4, gout_min, gout_max);*/

	//gamma correction
	/*Mat_<uchar> img5 = imread("Images/lab8/wilderness.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float gamma;
	cout << "gamma = " << endl;
	cin >> gamma;
	gamma_correction(img5, gamma);*/


	//EX4

	//histogram equalization
	/*Mat_<uchar> img6 = imread("Images/lab8/Hawkes_Bay_NZ.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img7 = imread("Images/lab8/wheel.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	histogram_equalization(img7);*/


	/*******************************************************************************/


	//LAB9

	// Part I
	// spatial domain filters

	//Mat_<uchar> img = imread("Images/lab9/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);	

	// low pass filters

	/*float mean_filter_3[9] = {1,1,1,1,1,1,1,1,1};
	float mean_filter_5[25] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	float gaussian_filter_3[9] = { 1,2,1,2,4,2,1,2,1 };

	Mat_<float> mean_3(3, 3, mean_filter_3);
	Mat_<float> mean_5(5, 5, mean_filter_5);
	Mat_<float> gaussian_3(3, 3, gaussian_filter_3);

	Mat_<uchar> img_mean3 = spatial_filter(img, mean_3);
	Mat_<uchar> img_mean5 = spatial_filter(img, mean_5);
	Mat_<uchar> img_gaussian = spatial_filter(img, gaussian_3);

	//imshow("img", img);

	imshow("mean3", img_mean3);
	imshow("mean5", img_mean5);
	imshow("gaussian", img_gaussian);

	// high pass filters

	float laplace_filter3_1[9] = { 0,-1,0,-1,4,-1,0,-1,0 };
	float laplace_filter3_2[9] = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
	float highpass_filter3_1[9] = { 0,-1,0,-1,5,-1,0,-1,0 };
	float highpass_filter3_2[9] = { -1,-1,-1,-1,9,-1,-1,-1,-1 };

	Mat_<float> laplace3_1(3, 3, laplace_filter3_1);
	Mat_<float> laplace3_2(3, 3, laplace_filter3_2);
	Mat_<float> highpass3_1(3, 3, highpass_filter3_1);
	Mat_<float> highpass3_2(3, 3, highpass_filter3_2);

	Mat_<uchar> img_laplace3_1 = spatial_filter(img, laplace3_1);
	Mat_<uchar> img_laplace3_2 = spatial_filter(img, laplace3_2);
	Mat_<uchar> img_mean_laplace = spatial_filter(img_mean3, laplace3_2);
	Mat_<uchar> img_highpass3_1 = spatial_filter(img, highpass3_1);
	Mat_<uchar> img_highpass3_2 = spatial_filter(img, highpass3_2);

	imshow("img", img);

	imshow("laplace3_1", img_laplace3_1);
	imshow("laplace3_2", img_laplace3_2);
	imshow("mean laplace", img_mean_laplace);
	imshow("highpass3_1", img_highpass3_1);
	imshow("highpass3_2", img_highpass3_2);*/

	// Part II
	// frequency domain filters
	/*Mat_<uchar> img = imread("Images/lab9/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img2 = imread("Images/lab9/circle.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat_<uchar> dst = generic_frequency_domain_filter(img);
	Mat_<uchar> dst2 = generic_frequency_domain_filter(img2);

	imshow("img", img);
	imshow("dst", dst);
	imshow("dst2", dst2);*/


	/*******************************************************************************/


	//LAB10

	// median filter
	/*Mat_<uchar> img = imread("Images/lab10/balloons_Salt&Pepper.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	double t1 = (double)getTickCount();
	Mat_<uchar> filter3 = median_filter(3, img);
	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	printf("w = 3; Time for median filter = %.3f [ms]\n", t1 * 1000);
	imshow("median filter w=3", filter3);

	double t2 = (double)getTickCount();
	Mat_<uchar> filter5 = median_filter(5, img);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();
	printf("w = 5; Time for median filter = %.3f [ms]\n", t2 * 1000);
	imshow("median filter w=5", filter5);

	double t3 = (double)getTickCount();
	Mat_<uchar> filter7 = median_filter(7, img);
	t3 = ((double)getTickCount() - t3) / getTickFrequency();
	printf("w = 5; Time for median filter = %.3f [ms]\n", t3 * 1000);
	imshow("median filter w=7", filter7);

	imshow("img", img);*/

	// 2d gaussian filter
	/*Mat_<uchar> img = imread("Images/lab10/portrait_Gauss1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("img", img);

	double t = (double)getTickCount();
	Mat_<uchar> filter = spatial_filter(img, gaussian_2d_filter(3));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 3; Time for 2D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("2d gaussian filter w=3", filter);

	t = (double)getTickCount();
	filter = spatial_filter(img, gaussian_2d_filter(5));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 5; Time for 2D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("2d gaussian filter w=5", filter);

	t = (double)getTickCount();
	filter = spatial_filter(img, gaussian_2d_filter(7));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 7; Time for 2D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("2d gaussian filter w=7", filter);*/

	// 1d gaussian filter
	/*Mat_<uchar> img = imread("Images/lab10/portrait_Gauss1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("img", img);
	double t;
	Mat_<uchar> filter2d;
	Mat_<uchar> filter1d;

	t = (double)getTickCount();
	filter2d = spatial_filter(img, gaussian_2d_filter(3));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 3; Time for 2D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("2d gaussian filter w=3", filter2d);

	t = (double)getTickCount();
	filter1d = spatial_filter(spatial_filter(img, gaussian_1d_filter_gy(3)), gaussian_1d_filter_gx(3));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 3; Time for 1D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("1d gaussian filter w=3", filter1d);

	cout << endl;

	t = (double)getTickCount();
	filter2d = spatial_filter(img, gaussian_2d_filter(5));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 5; Time for 2D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("2d gaussian filter w=5", filter2d);

	t = (double)getTickCount();
	filter1d = spatial_filter(spatial_filter(img, gaussian_1d_filter_gy(5)), gaussian_1d_filter_gx(5));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 5; Time for 1D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("1d gaussian filter w=5", filter1d);

	cout << endl;

	t = (double)getTickCount();
	filter2d = spatial_filter(img, gaussian_2d_filter(7));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 7; Time for 2D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("2d gaussian filter w=7", filter2d);

	t = (double)getTickCount();
	filter1d = spatial_filter(spatial_filter(img, gaussian_1d_filter_gy(7)), gaussian_1d_filter_gx(7));
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("w = 7; Time for 1D Gaussian filter = %.3f [ms]\n", t * 1000);
	imshow("1d gaussian filter w=7", filter1d);*/


	/*******************************************************************************/



	waitKey();
}