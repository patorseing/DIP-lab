//
//  main.cpp
//  OpenCVTest
//
//  Created by Napatchol Thaipanich on 25/9/18.
//  Copyright © 2018 Napatchol Thaipanich. All rights reserved.
//
//https://github.com/samidalati/OpenCV-Entropy/blob/master/histGray.cpp
//https://www.mathworks.com/help/images/ref/entropy.html
//http://imagej.net/plugins/download/Entropy_Threshold.java
//https://stackoverflow.com/questions/1153548/minimum-double-value-in-c-c
//https://stackoverflow.com/questions/17698431/extracting-background-image-using-grabcut
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
// for lab 17
CvRect box;
bool drawing_box = true;
//
void test(){
    cout << CV_VERSION << endl;
}

//Lab1: Show Image
void lab1(const char * argv[]){
    Mat img;
    img = imread(argv[1]);
    namedWindow("Display", WINDOW_AUTOSIZE);
    imshow("Display", img);
    waitKey(0);
}

//Lab2: Convert Color Models
void lab2(const char * argv[]){
    Mat img;
    img = imread(argv[1]);
    namedWindow("Display", WINDOW_AUTOSIZE);
    imshow("Display", img);
    
    Mat img_grey;
    cvtColor(img, img_grey, COLOR_BGR2GRAY);
    namedWindow("Grey", WINDOW_AUTOSIZE);
    imshow("Grey", img_grey);
    
    waitKey(0);
}

//Lab3: Smoothing Image using Gaussian Blur
void lab3(const char * argv[]){
    Mat img, out;
    img = imread(argv[1]);
    GaussianBlur(img, out, Size(11,11), 0,0);
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    
    waitKey(0);
}

//Lab4: Smoothing Image using Gaussian Blur
void lab4(const char * argv[]){
    Mat img, out, out2, out3;
    img = imread(argv[1]);
    blur(img, out, Size(9,9));
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("blur", WINDOW_AUTOSIZE);
    imshow("blur", out);
    GaussianBlur(img, out2, Size(9,9), 0,0);
    namedWindow("GaussianBlur", WINDOW_AUTOSIZE);
    imshow("GaussianBlur", out2);
    medianBlur(img, out3, 9);
    namedWindow("medianBlur", WINDOW_AUTOSIZE);
    imshow("medianBlur", out3);
    waitKey(0);
}

//Lab5: Convolutions
void lab5(const char * argv[]){
    Mat img, out;
    img = imread(argv[1]);
    //Mat kernal  = (Mat_<double>(3,3) << 1,-2,1,2,-4,2,1,-2,1);
    Mat kernal  = (Mat_<double>(3,3) << -1,-1,-1,-1,8,-1,-1,-1,-1);
    Point anchor;
    double delta;
    int ddepth;
    anchor = Point(-1,-1);
    delta = 0;
    ddepth = -1;
    filter2D(img, out, ddepth, kernal, anchor, delta, BORDER_DEFAULT);
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    waitKey(0);
}

//Lab6: Erode (การกร่อน ขนาดบริเวณขอบของวัตถุ)
void lab6(const char * argv[]){
    //Rectangular box: MORPH_RECT
    //Cross: MORPH_CROSS
    //Ellipse: MORPH_ELLIPSE
    Mat img, out;
    img = imread(argv[1]);
    cvtColor(img, img, CV_RGB2GRAY);
    threshold( img, img, 128,255,THRESH_BINARY );
    //increase erosion_size the white will smaller
    int erosion_size = 4;
    Mat element = getStructuringElement(MORPH_CROSS,
                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        Point(erosion_size, erosion_size) );
    
    // Apply erosion or dilation on the image
    erode(img,out,element);  // dilate(image,dst,element);
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    waitKey(0);
}

//Lab7: Dilate (การขยาย)
void lab7(const char * argv[]){
    Mat img, out;
    img = imread(argv[1]);
    cvtColor(img, img, CV_RGB2GRAY);
    threshold( img, img, 128,255,THRESH_BINARY );
    //decrease the dilation_size white will the decrease too
    int dilation_size = 4;
    Mat element = getStructuringElement(MORPH_CROSS,
                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                        Point(dilation_size, dilation_size) );
    //cout << Point(dilation_size, dilation_size) << "\n";
    //cout << Size(2 * dilation_size + 1, 2 * dilation_size + 1) << "\n";
    //cout << element << "\n";
    // Apply erosion or dilation on the image
    //erode(img,out,element);
    dilate(img,out,element);
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    waitKey(0);
}

//Lab8: Opening
void lab8(const char * argv[]){
    //Rectangular box: MORPH_RECT
    //Cross: MORPH_CROSS
    //Ellipse: MORPH_ELLIPSE
    Mat img, out, out2;
    img = imread(argv[3]);
    cvtColor(img, img, CV_RGB2GRAY);
    threshold( img, img, 128,255,THRESH_BINARY );
    int dilation_size = 4;
    Mat element = getStructuringElement(MORPH_CROSS,
                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1));
    //cout << Point(dilation_size, dilation_size) << "\n";
    //cout << Size(2 * dilation_size + 1, 2 * dilation_size + 1) << "\n";
    //cout << element << "\n";
    // Apply erosion or dilation on the image
    //erode(img,out,element);
    //dilate(out,out2,element);
    morphologyEx(img, out, MORPH_OPEN, element);
    //erode(out,out2,element);
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    //namedWindow("Opening", WINDOW_AUTOSIZE);
    //imshow("Opening", out2);
    waitKey(0);
}

//Lab9: Closing
void lab9(const char * argv[]){
    //Rectangular box: MORPH_RECT
    //Cross: MORPH_CROSS
    //Ellipse: MORPH_ELLIPSE
    Mat img, out, out2;
    img = imread(argv[3]);
    cvtColor(img, img, CV_RGB2GRAY);
    threshold( img, img, 128,255,THRESH_BINARY );
    int dilation_size = 10;
    Mat element = getStructuringElement(MORPH_CROSS,
                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1));
    //cout << Point(dilation_size, dilation_size) << "\n";
    //cout << Size(2 * dilation_size + 1, 2 * dilation_size + 1) << "\n";
    //cout << element << "\n";
    // Apply erosion or dilation on the image
    //erode(img,out,element);
    //dilate(img,out,element);
    //erode(out,out2,element);
    morphologyEx(img, out, MORPH_CLOSE, element);
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    //namedWindow("Closing", WINDOW_AUTOSIZE);
    //imshow("Closing", out2);
    waitKey(0);
}

//Lab10: Histogram Equalization
void lab10(const char * argv[]){
    Mat img, out;
    img = imread(argv[1]);
    /// Convert to grayscale
    cvtColor( img, img, CV_BGR2GRAY );
    
    /// Apply Histogram Equalization
    equalizeHist( img, out );
    namedWindow("In", WINDOW_AUTOSIZE);
    imshow("In", img);
    namedWindow("Out", WINDOW_AUTOSIZE);
    imshow("Out", out);
    waitKey(0);
}

//Lab11: Image segmentation: thresholding
void lab11(const char * argv[]){
    Mat img = imread(argv[2]);
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    /// Convert to grayscale
    Mat img_gray;
    cvtColor( img, img_gray, CV_BGR2GRAY );
    
    Mat seg;
    threshold(img_gray, seg, 200, 255, CV_THRESH_BINARY);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", seg);
    waitKey(0);
}

 //Lab12: Thresholding – mean (don't use adaptiveThreshold)
void lab12(const char * argv[]){
    Mat img = imread(argv[2]);
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    /// Convert to grayscale
    Mat img_gray;
    cvtColor( img, img_gray, CV_BGR2GRAY );
    
    Mat seg;
    /*
    int total = 0;
    for (int i=0; i<img_gray.rows; i++) {
        for (int j=0; j<img_gray.cols; j++) {
            int val = img_gray.at<uchar>(i,j);
            total += val;
        }
    }
    int mean = total/(img_gray.rows*img_gray.cols);
    */
    Scalar meanScalar = mean(img_gray);
    float mean = meanScalar.val[0];
    cout << mean << endl;
    //adaptiveThreshold(img_gray, seg, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);
    threshold(img_gray, seg, mean, 255, CV_THRESH_BINARY);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", seg);
    waitKey(0);
}

//Lab13: Image segmentation: fixed-ratio
void lab13(const char * argv[]){
    Mat img = imread(argv[2],IMREAD_COLOR);
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    /// Convert to grayscale
    Mat img_gray;
    cvtColor( img, img_gray, CV_BGR2GRAY );
    
    // create histogram
    int histSize = 256;
    
    float range[] = {0,255};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat hist;
    calcHist(&img_gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    
    //Normalize histogram
    Scalar sumScalar = sum(hist);
    float total = sumScalar.val[0];
    hist /= total;
    
    //Find  threshold
    //float ratio = 0.50; //158
    //float ratio = 0.80; //239
    float ratio = 0.1; //108
    float count = 0;
    int i;
    for (i = 0 ; i<histSize; i++) {
        count += hist.at<float>(i);
        if (count>=ratio) {
            break;
        }
    }
    cout << i << endl;
    Mat seg;
    threshold(img_gray, seg, i, 255, CV_THRESH_BINARY);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", seg);
    waitKey(0);
}

//Lab14: Image segmentation: iterative selection and use the mean as the initial threshold
void lab14(const char * argv[]){
    Mat img = imread(argv[2]);
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    /// Convert to grayscale
    Mat img_gray;
    cvtColor( img, img_gray, CV_BGR2GRAY );
    
    Mat seg;
    /*
    int total = 0;
    for (int i=0; i<img_gray.rows; i++) {
        for (int j=0; j<img_gray.cols; j++) {
            int val = img_gray.at<uchar>(i,j);
            total += val;
        }
    }
    int mean = total/(img_gray.rows*img_gray.cols);
    */
    
    // create histogram
    int histSize = 256;
    
    float range[] = {0,255};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat hist;
    calcHist(&img_gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    
    //Normalize histogram
    Scalar sumScalar = sum(hist);
    float total = sumScalar.val[0];
    hist /= total;
    
    Scalar meanScalar = mean(img_gray);
    int mean = meanScalar.val[0];
    
    float w = 0,b = 0,Tw,Tb,Cw = 0,Cb = 0, old = mean, count=0;
    cout << mean << endl;
    mean = 0;
    while (old != mean && count < 10) {
        count++;
        old = mean;
        Cw = 0;
        Cb = 0;
        w = 0;
        b = 0;
        for (int i=0; i<img_gray.rows; i++) {
            for (int j=0; j<img_gray.cols; j++) {
                int val = img_gray.at<uchar>(i,j);
                if (val > mean) {
                    w += val;
                    Cw++;
                } else {
                    b += val;
                    Cb++;
                }
            }
        }
        if (Cw == 0) {
            Tw = w;
        } else {
            Tw = w/Cw;
        }
        if (Cb == 0) {
            Tb = b;
        } else {
            Tb = b/Cb;
        }
        /*
        for (int i = 0; i < mean+1; i++) {
            Cb += i*hist.at<float>(0, i);
            b += hist.at<float>(0, i);
        }
        for (int i = mean+1; i < histSize; i++) {
            Cw += i*hist.at<float>(0, i);
            w += hist.at<float>(0, i);
        }
         mean = Cb/ 2*b + Cw/ 2*w;
         */
        mean = (Tw + Tb)/2;
        cout << count << endl;
        //cout << "Tw: "<< Tw << endl;
        //cout << "Tb: "<< Tb << endl;
        cout << "mean: "<< mean << endl;
    }
     
    //adaptiveThreshold(img_gray, seg, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);
    threshold(img_gray, seg, old, 255, CV_THRESH_BINARY);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", seg);
    waitKey(0);
}

//Lab15: Write a program to segment an image by using Entropy technique
void lab15(const char * argv[]){
    Mat img = imread(argv[5]);
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    
    /// Convert to grayscale
    Mat img_gray;
    cvtColor( img, img_gray, CV_BGR2GRAY );
    
    Mat seg;
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    
    Mat hist;
    
    /// Compute the histograms:
    calcHist( &img_gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    
    
    //Normalize histogram
    Scalar sumScalar = sum(hist);
    float total = sumScalar.val[0];
    hist /= total;
    //
    double pT[histSize];
    pT[0] = hist.at<float>(0, 0);
    for (int i = 1; i < histSize; i++) {
        pT[i] = pT[i - 1] + hist.at<float>(0, i);
    }
    // Entropy for black and white parts of the histogram
    double epsilon = DBL_MIN;
    double hB[histSize], hW[histSize];
    for (int t = 0; t < histSize; t++) {
        // Black entropy
        if (pT[t] > epsilon) {
            double hhB = 0;
            for (int i = 0; i <= t; i++) {
                if (hist.at<float>(0, i) > epsilon) {
                    hhB -= hist.at<float>(0, i) / pT[t] * log(hist.at<float>(0, i) / pT[t]);
                }
            }
            hB[t] = hhB;
        } else {
            hB[t] = 0;
        }
        
        // White  entropy
        double pTW = 1 - pT[t];
        if (pTW > epsilon) {
            double hhW = 0;
            for (int i = t + 1; i < histSize; ++i) {
                if (hist.at<float>(0, i) > epsilon) {
                    hhW -= hist.at<float>(0, i) / pTW * log(hist.at<float>(0, i) / pTW);
                }
            }
            hW[t] = hhW;
        } else {
            hW[t] = 0;
        }
    }
    // Find histogram index with maximum entropy
    double HMax = hB[0] + hW[0];
    int tMax = 0;
    for (int t = 1; t < histSize; ++t) {
        double H = hB[t] + hW[t];
        if (H > HMax) {
            HMax = H;
            tMax = t;
        }
    }
    cout << tMax<<endl;
    threshold(img_gray, seg, tMax, 255, CV_THRESH_BINARY);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", seg);
    waitKey(0);
}

//Lab16: Thresholding using mean + Watershed
void lab16(const char * argv[]){
    Mat img = imread(argv[2],IMREAD_COLOR);
    namedWindow("input");
    imshow("input", img);
    
    //Thresholding using mean
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    Scalar meanScalar = mean(img_gray);
    float mean = meanScalar.val[0];
    Mat thres;
    threshold(img_gray, thres, mean+20, 255, CV_THRESH_BINARY);
    namedWindow("Threshold");
    imshow("Threshold", thres);
    
    // Erode + Dilate
    Mat fg, bg;
    int size = 1;
    Mat element = getStructuringElement(MORPH_RECT, Size(2*size+1,2*size+1), Point(size,size));
    erode(thres, fg, element);
    dilate(thres, bg, element);
    namedWindow("erode");
    imshow("erode", fg);
    namedWindow("dilate");
    imshow("dilate", bg);
    
    // find markers
    Mat marker;
    threshold(bg, bg, 1, 128, CV_THRESH_BINARY_INV);
    namedWindow("bg");
    imshow("bg", bg);
    add(fg, bg, marker);
    namedWindow("marker");
    imshow("marker", marker);
    
    // Watershed
    Size sz = img.size();
    int height = sz.height;
    int width = sz.width;
    Mat output(height,width,CV_32SC1);
    marker.convertTo(output, CV_32SC1);
    watershed(img, output);
    output.convertTo(marker, CV_8UC1);
    namedWindow("watershed");
    imshow("watershed", marker);
    waitKey(0);
}

void my_mouse_callback(int event, int x, int y, int flag, void* param){
    IplImage* image = (IplImage*) param;
    switch (event) {
        case CV_EVENT_LBUTTONDOWN:
            drawing_box = true;
            box = cvRect(x, y, 0, 0);
            break;
            
        case CV_EVENT_MOUSEMOVE:
            if (drawing_box) {
                box.width = x - box.x;
                box.height = y-box.y;
            }
            break;
        case CV_EVENT_LBUTTONUP:
            drawing_box = false;
            if (box.width < 0) {
                box.x += box.width;
                box.width*=-1;
            }
            if (box.height < 0) {
                box.y += box.height;
                box.height*=-1;
            }
            cvRectangle(image, cvPoint(box.x, box.y), cvPoint(box.x+box.width, box.y+box.height),CV_RGB(255,0,0));
            break;
    }
}

// Alpha blending using multiply and add functions
Mat& blend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
    Mat fore, back;
    multiply(alpha, foreground, fore);
    multiply(Scalar::all(1.0)-alpha, background, back);
    add(fore, back, outImage);
    
    return outImage;
}

//Lab17: GrabCut with Rect
void lab17(const char * argv[]){
    
    
    /*Mat img = imread(argv[6],IMREAD_COLOR);
    Rect rectangle(25,70,img.cols -80, img.rows-120);
    Mat result;
    Mat fg,bg;
    grabCut(img, result, rectangle, bg, fg,1, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    Mat foreground(img.size(),CV_8UC3, Scalar(0,0,0));
    img.copyTo(foreground,result);
    cv::rectangle(img, rectangle, Scalar(0,0,255), 1);
    namedWindow("input");
    imshow("input", img);
    namedWindow("output");
    imshow("output", foreground);
    
    //Lab18: Scene Mapping
    //https://github.com/spmallick/learnopencv/tree/master/AlphaBlending
    Mat back = imread(argv[7],IMREAD_COLOR);
    namedWindow("inputbg");
    imshow("inputbg", back);

    waitKey(0);*/
    int num = 6;
    box = cvRect(-1, -1, 0, 0);
    IplImage* image = cvLoadImage(argv[num]);
    IplImage* temp = cvCloneImage(image);
    cvNamedWindow("input");
    cvSetMouseCallback("input", my_mouse_callback,(void*) image);
    while (1) {
        cvCopy(image,temp);
        if(drawing_box){
            cvRectangle(temp, cvPoint(box.x, box.y), cvPoint(box.x+box.width, box.y+box.height),CV_RGB(255,0,0));
        }
        cvShowImage("input", temp);
        if (cvWaitKey(15) == 27) {
            cout << box.x <<endl;
            cout << box.y <<endl;
            cout << box.width <<endl;
            cout << box.height <<endl;
            break;
        }
    }
    Mat img = imread(argv[num],IMREAD_COLOR);
    Rect rectangle(box.x,box.y,box.width,box.height);
    Mat result;
    Mat fg,bg;
    grabCut(img, result, rectangle, bg, fg,1, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    Mat foreground(img.size(),CV_8UC3, Scalar(0,0,0));
    img.copyTo(foreground,result);
    cv::rectangle(img, rectangle, Scalar(0,0,255), 1);
    namedWindow("output");
    imshow("output", foreground);
    Mat background = imread(argv[num+1],IMREAD_COLOR);
    namedWindow("back");
    imshow("back", background);

    Mat tmp,alphaF_inv,alphaF,alphaB;
    cvtColor(foreground,tmp,CV_BGR2GRAY);
    threshold(tmp,alphaF,0,255,THRESH_BINARY);
    Mat rgb[3];
    split(foreground,rgb);
    Mat rgba[4] = {rgb[0],rgb[1],rgb[2],alphaF};
    merge(rgba,4,foreground);
    imwrite("dst.png",foreground);
    
    split(background,rgb);
    cvtColor(background,tmp,CV_BGR2GRAY);
    threshold(tmp,alphaB,0,255,THRESH_BINARY);
    Mat Brgba[4] = {rgb[0],rgb[1],rgb[2],alphaB};
    merge(Brgba,4,background);
    
    cout<< alphaF<<endl;
    foreground.copyTo(background.rowRange(background.rows-foreground.rows, background.rows).colRange(0, foreground.cols),alphaF);
    cout<<background.channels()<<endl;
    //foreground.copyTo(background(Rect(background.rows-foreground.rows,0, background.rows, foreground.cols)),alphaF);
    namedWindow("final");
    imshow("final", background);
    waitKey(0);
}

//Lab 19: Sobel operation
void lab19(const char * argv[]){
    Mat img = imread(argv[6],IMREAD_COLOR);
    namedWindow("Input");
    imshow("Input", img);
    
    Mat sobel_x, sobel_y;
    Mat abs_sobel_x, abs_sobel_y;
    
    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    
    //Gradient X
    Sobel(img_gray, sobel_x, CV_16S, 1, 0,3);
    convertScaleAbs(sobel_x, abs_sobel_x);
    namedWindow("sobel_x");
    imshow("sobel_x", abs_sobel_x);
    
    //Gradient Y
    Sobel(img_gray, sobel_y, CV_16S, 0, 1,3);
    convertScaleAbs(sobel_y, abs_sobel_y);
    namedWindow("sobel_y");
    imshow("sobel_y", abs_sobel_y);
    
    //Gradient Y and X
    Sobel(img_gray, sobel_y, CV_16S, 1, 1,3);
    convertScaleAbs(sobel_y, abs_sobel_y);
    namedWindow("sobel_y_x");
    imshow("sobel_y_x", abs_sobel_y);
    
    waitKey(0);
}

//Lab20: Write a program to find a gradient (i.e. magnitude) of an image based on Sobel operation
void lab20(const char * argv[]){
    Mat img = imread(argv[2],IMREAD_COLOR);
    namedWindow("Input");
    imshow("Input", img);
    
    Mat sobel_x, sobel_y;
    Mat abs_sobel_x, abs_sobel_y;
    
    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    
    //Gradient X
    Sobel(img_gray, sobel_x, CV_16S, 1, 0,3);
    convertScaleAbs(sobel_x, abs_sobel_x);
    namedWindow("sobel_x");
    imshow("sobel_x", abs_sobel_x);
    
    //Gradient Y
    Sobel(img_gray, sobel_y, CV_16S, 0, 1,3);
    convertScaleAbs(sobel_y, abs_sobel_y);
    namedWindow("sobel_y");
    imshow("sobel_y", abs_sobel_y);
    
    Mat m(img.rows,img.cols,double(0));
    double c = 2;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            double x = abs_sobel_x.at<double>(i, j);
            double y = abs_sobel_y.at<double>(i, j);
            double x_2 = pow(x,c);
            double y_2 = pow(y,c);
            m.at<double>(i, j) = (double)sqrt((x_2+y_2)/c);
        }
    }
    namedWindow("result");
    imshow("result", m);
    
    waitKey(0);
}

//Lab21: Create a histogram
void lab21(const char * argv[]){
    Mat img = imread(argv[2], IMREAD_COLOR);
    
    // Separate the image in 3 places (B,G,R)
    vector<Mat> img_bgr;
    split(img, img_bgr);
    
    // Establish the number of bits
    int histSize = 256;
    
    // Set the ranges
    float range[]  = {0,256};
    const float* histRange = {range};
    
    bool uniform = true;
    bool accumulate = false;
    
    Mat b_hist, g_hist, r_hist;
    
    // Compute the histogram
    calcHist(&img_bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&img_bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&img_bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
    
    // Draw the histogram for B, G, and R
    int hist_w = 516; // 2 pixels-width for each bin display
    int hist_h = 400;
    int bin_w = cvRound((double)hist_h/ histSize); // = 2
    
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));
    
    // Normalize the result to [0, histImage.rows i.e. hist_h]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    
    // Draw for each channel
    for (int i = 1; i < histSize; i++) {
        line(histImage,
             Point(bin_w*(i-1),hist_h-cvRound(b_hist.at<float>(i-1))),
             Point(bin_w*(i),hist_h-cvRound(b_hist.at<float>(i))),
             Scalar(255,0,0),2,8,0);
        line(histImage,
             Point(bin_w*(i-1),hist_h-cvRound(g_hist.at<float>(i-1))),
             Point(bin_w*(i),hist_h-cvRound(g_hist.at<float>(i))),
             Scalar(0,255,0),2,8,0);
        line(histImage,
             Point(bin_w*(i-1),hist_h-cvRound(r_hist.at<float>(i-1))),
             Point(bin_w*(i),hist_h-cvRound(r_hist.at<float>(i))),
             Scalar(0,0,255),2,8,0);
    }
    
    // Display
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    namedWindow("Hist", WINDOW_AUTOSIZE);
    imshow("Hist", histImage);
    
    waitKey(0);
}

//Lab22: Compare Histogram
void lab22(const char * argv[]){
    Mat img1, hsv1;
    Mat img2, hsv2;
    
    // 2 images with different sizes
    img1 = imread(argv[2], IMREAD_COLOR);
    img2 = imread(argv[3], IMREAD_COLOR);
    
    // CONVERT TO HSV
    cvtColor(img1, hsv1, COLOR_BGR2HSV);
    cvtColor(img2, hsv2, COLOR_BGR2HSV);
    
    // Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = {h_bins,s_bins};
    
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    
    const float* ranges[] = {h_ranges, s_ranges};
    
    // Use the 0-th (i.e. H) and the 1-th (i.e. S) channels
    int channels[] = {0,1};
    
    // Histograms
    MatND hist1;
    MatND hist2;
    
    // Calculate the histograms for the HSV images
    calcHist(&hsv1, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    
    calcHist(&hsv2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
    
    // Apply the histogram comparison method
    double sim = compareHist(hist1, hist2, CV_COMP_CORREL);
    cout << "The correlation similarity is " << sim << endl;
    
}

//Lab23: Find the most similar image
void lab23(const char * argv[]){
    Mat img1, hsv1;
    Mat img2, hsv2;
    Mat img3, hsv3;
    
    // 2 images with different sizes
    img1 = imread(argv[8], IMREAD_COLOR);
    img2 = imread(argv[9], IMREAD_COLOR);
    img3 = imread(argv[10], IMREAD_COLOR);
    
    // CONVERT TO HSV
    cvtColor(img1, hsv1, COLOR_BGR2HSV);
    cvtColor(img2, hsv2, COLOR_BGR2HSV);
    cvtColor(img3, hsv3, COLOR_BGR2HSV);
    
    // Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = {h_bins,s_bins};
    
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    
    const float* ranges[] = {h_ranges, s_ranges};
    
    // Use the 0-th (i.e. H) and the 1-th (i.e. S) channels
    int channels[] = {0,1};
    
    // Histograms
    MatND hist1;
    MatND hist2;
    MatND hist3;
    
    // Calculate the histograms for the HSV images
    calcHist(&hsv1, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    
    calcHist(&hsv2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
    
    calcHist(&hsv3, 1, channels, Mat(), hist3, 2, histSize, ranges, true, false);
    normalize(hist3, hist3, 0, 1, NORM_MINMAX, -1, Mat());
    
    // Apply the histogram comparison method
    double sim = compareHist(hist1, hist2, CV_COMP_CORREL);
    double sim2 = compareHist(hist1, hist3, CV_COMP_CORREL);
    
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img1);
    namedWindow("Input2", WINDOW_AUTOSIZE);
    imshow("Input2", img2);
    namedWindow("Input3", WINDOW_AUTOSIZE);
    imshow("Input3", img3);
    cout << "I1 vs I2 is " << sim << endl;
    cout << "I1 vs I3 is " << sim2 << endl;
    waitKey(0);
}

float lbp(Mat img, int i, int j){
    float result = 0;
    for (int x = 0; x < 8; x++) {
        switch (x) {
            case 0:
                if (i-1 < 0 || j - 1< 0 ||i-1 > img.rows || j - 1> img.cols ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i-1, j-1) >= img.at<float>(i, j)) {
                        result += 128;
                    }
                }
                break;
                
            case 1:
                if (i-1 < 0 || i-1 > img.rows ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i, j-1) >= img.at<float>(i, j)) {
                        result += 64;
                    }
                }
                break;
                
            case 2:
                if (i-1 < 0 || j + 1< 0 ||i-1 > img.rows || j + 1> img.cols ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i-1, j+1) >= img.at<float>(i, j)) {
                        result += 32;
                    }
                }
                break;
                
            case 3:
                if (j + 1< 0 ||j + 1> img.cols ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i, j+1) >= img.at<float>(i, j)) {
                        result += 16;
                    }
                }
                break;
                
            case 4:
                if (i+1 < 0 || j + 1< 0 ||i+1 > img.rows || j + 1> img.cols ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i+1, j+1) >= img.at<float>(i, j)) {
                        result += 8;
                    }
                }
                break;
                
            case 5:
                if (i+1 < 0 ||i+1 > img.rows ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i+1, j) >= img.at<float>(i, j)) {
                        result += 4;
                    }
                }
                break;
                
            case 6:
                if (i+1 < 0 || j - 1< 0 ||i+1 > img.rows || j - 1> img.cols ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i+1, j-1) >= img.at<float>(i, j)) {
                        result += 2;
                    }
                }
                break;
                
            case 7:
                if (j - 1< 0 ||j - 1> img.cols ) {
                    result += 0;
                }
                else {
                    if (img.at<float>(i, j-1) >= img.at<float>(i, j)) {
                        result += 2;
                    }
                }
                break;
            default:
                break;
        }
    }
    //cout << " "<<abs(img.at<float>(i, j));
    return result;
}

//Lab24: (Local Binary Pattern (LBP)) Write the program to create and visualize the normalized LBP histogram of the texture images
void lab24(const char * argv[]){
    Mat img;
    
    // Load the color image
    img = imread(argv[2], IMREAD_COLOR);
    
    // Convert to grayscale image
    cvtColor(img, img, CV_BGR2GRAY);
    
    // Calculate the LBP mask
    Mat mask(img.rows,img.cols,double(0));
    
    for (int i = 0; i <img.rows; i++) {
        for (int j = 0; j< img.cols; j++) {
            mask.at<float>(i, j) = lbp(img, i, j);
        }
    }
    // Display
    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", img);
    namedWindow("markLBP", WINDOW_AUTOSIZE);
    imshow("markLBP", mask);
    
    // Calculate the LBP histogram
    // Establish the number of bits
    int histSize = 256;
    
    float range[] = {0,255};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat hist, Ohist;
    calcHist(&img, 1, 0, Mat(), Ohist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&mask, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    
    //Normalize the LBP histogram
    /*Scalar sumScalar = sum(hist);
    float total = sumScalar.val[0];
    hist /= total;*/
    
    // Draw the histogram for B, G, and R
    int hist_w = 256; // 2 pixels-width for each bin display
    int hist_h = 256;
    int bin_w = cvRound((double)hist_h/ histSize); // = 2
    
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(Ohist, Ohist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    
    for (int i = 1; i < histSize; i++) {
        /*rectangle( histImage,
                  Point(bin_w*(i-1),hist_h-cvRound(hist.at<float>(i-1))),
                  Point(bin_w*(i),hist_h-cvRound(hist.at<float>(i))),
                  Scalar(255,0,0),5,10,0);*/
        line(histImage,
             Point(bin_w*(i-1),hist_h-cvRound(hist.at<float>(i-1))),
             Point(bin_w*(i),hist_h-cvRound(hist.at<float>(i))),
             Scalar(255,0,0),2,8,0);
        /*line(histImage,
             Point(bin_w*(i-1),hist_h-cvRound(Ohist.at<float>(i-1))),
             Point(bin_w*(i),hist_h-cvRound(Ohist.at<float>(i))),
             Scalar(0,0,255),2,8,0);*/
    }
    namedWindow("Hist", WINDOW_AUTOSIZE);
    imshow("Hist", histImage);
    waitKey(0);
}

int main(int argc, const char * argv[]){
    int x = 0;
    /* For single input */
    cout << "Enter a number: ";
    cin >> x;
    switch (x) {
        case 1:
            lab1(argv);
            break;
            
        case 2:
            lab2(argv);
            break;
            
        case 3:
            lab3(argv);
            break;
            
        case 4:
            lab4(argv);
            break;
            
        case 5:
            lab5(argv);
            break;
            
        case 6:
            lab6(argv);
            break;
            
        case 7:
            lab7(argv);
            break;
            
        case 8:
            lab8(argv);
            break;
            
        case 9:
            lab9(argv);
            break;
        
        case 10:
            lab10(argv);
            break;
            
        case 11:
            lab11(argv);
            break;
            
        case 12:
            lab12(argv);
            break;
            
        case 13:
            lab13(argv);
            break;
            
        case 14:
            lab14(argv);
            break;
            
        case 15:
            lab15(argv);
            break;
            
        case 16:
            lab16(argv);
            break;
            
        case 17:
            lab17(argv);
            break;
        
        case 19:
            lab19(argv);
            break;
            
        case 20:
            lab20(argv);
            break;
            
        case 21:
            lab21(argv);
            break;
            
        case 22:
            lab22(argv);
            break;
            
        case 23:
            lab23(argv);
            break;
            
        case 24:
            lab24(argv);
            break;
                
        default:
            test();
            break;
    }
    return 0;
}
