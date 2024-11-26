#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

string detectTrafficLights(Mat frame) {
    // 1. Preprocessing
    Mat blurred;
    GaussianBlur(frame, blurred, Size(5, 5), 0);

    // 2. Convert to HSV
    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    // 3. Brightness thresholding for ROI
    Mat gray;
    cvtColor(blurred, gray, COLOR_BGR2GRAY);
    Mat thresh;
    threshold(gray, thresh, 150, 255, THRESH_BINARY); // Adjust threshold as needed
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat opening, closing;
    morphologyEx(thresh, opening, MORPH_OPEN, kernel);
    morphologyEx(opening, closing, MORPH_CLOSE, kernel);
    vector<vector<Point>> contours;
    findContours(closing, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the largest contour (assumed to be the traffic light)
    vector<Point> largestContour;
    double maxArea = 0;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            largestContour = contour;
        }
    }

    if (largestContour.empty()) {
        return "NOT"; // No bright region found
    }

    Rect roiRect = boundingRect(largestContour);
    Mat roi = hsv(roiRect);

    // 4. Color detection in ROI (handling multiple colors)
    Scalar lowerRed = {0, 140, 150};
    Scalar upperRed = {10, 255, 255};
    Scalar lowerRed2 = {120, 100, 100};
    Scalar upperRed2 = {130, 255, 255};
    Scalar lowerGreen = {40, 120, 120};
    Scalar upperGreen = {80, 255, 255};
    Scalar lowerYellow = {20, 120, 120};
    Scalar upperYellow = {30, 255, 255};

    Mat maskRed, maskGreen, maskYellow;
    inRange(roi, lowerRed, upperRed, maskRed);
    inRange(roi, lowerRed2, upperRed2, Mat temp);
    maskRed += temp;
    inRange(roi, lowerGreen, upperGreen, maskGreen);
    inRange(roi, lowerYellow, upperYellow, maskYellow);

    int redPixels = countNonZero(maskRed);
    int greenPixels = countNonZero(maskGreen);
    int yellowPixels = countNonZero(maskYellow);

    vector<string> results;
    int pixelThreshold = 80; // Adjust as needed

    if (redPixels > pixelThreshold) {
        results.push_back("RED");
    }
    if (greenPixels > pixelThreshold) {
        results.push_back("GREEN");
    }
    if (yellowPixels > pixelThreshold) {
        results.push_back("YELLOW");
    }

    if (results.empty()) {
        return "NOT";
    } else {
        string resultString;
        for (size_t i = 0; i < results.size(); ++i) {
            resultString += results[i];
            if (i < results.size() - 1) {
                resultString += ", ";
            }
        }
        return resultString;
    }
}

int main() {
    string videoPath = ""; // Replace with your video path
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    while (cap.isOpened()) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        string lightStatus = detectTrafficLights(frame);
        putText(frame, lightStatus, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        imshow("Traffic Light Detection", frame);

        if (waitKey(25) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
