#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"

#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);
int gpumain();
void cpumain(const char** argv);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
double fps = 0;
// !! Modify the path with opencv drirectory that contains the cascade data
String opencv_path = "./work/opencv";
// Set the gloabal variable to true inorder to use the GPU
// or false to use only CPU
bool gpu_en = false;

void cpumain(const char** argv)
{
    // Using milli as time unit for fps calculation
    using milli = std::chrono::milliseconds;
    String str(argv[1]);
    int camera_device = 0;
    VideoCapture capture(argv[1]);
    // Read the video stream
    
    capture.open(argv[1],CAP_ANY);
    // check if open succeeded
    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open videoFile\n";
        }
        //capture.VideoCapture(argv[1]);
    // Get input FPS from video capture
    double frames_per_second = capture.get(CAP_PROP_FPS);
    Mat frame;
    double millisec, total_milli = 0;
    int count = 0;
    cout << "Processing frames on a CPU for:"<<argv[1];
    cout << " FPS : " << frames_per_second << endl;
    while (capture.read(frame))
    {
        // Start time for fps calculation
        auto start = std::chrono::high_resolution_clock::now();
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        // Apply the classifier to the frame
        detectAndDisplay(frame);
        // End time for fps calculation
        auto finish = std::chrono::high_resolution_clock::now();
        millisec = std::chrono::duration_cast<milli>(finish - start).count();
        total_milli = total_milli + millisec;
        count = count + 1;
        if (count % (int)frames_per_second == 0)
        {
            // Calculate output FPS based on the time taken to compute input FPS
            fps = (double)(frames_per_second * 1000 / total_milli);
            cout << " FPS : " << fps << endl;
            total_milli = 0;
        }
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
}

void detectAndDisplay(Mat frame)
{
    // !! Ensure the path is correct
    // !! Else the program fails.
    String eyes_cascade_name = opencv_path + "/data/haarcascades/haarcascade_eye.xml";
    String face_cascade_name = opencv_path + "/data/haarcascades/haarcascade_frontalface_alt.xml";
    face_cascade.load(face_cascade_name);
    eyes_cascade.load(eyes_cascade_name);
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    for (size_t i = 0; i < faces.size(); i++)
    {
        rectangle(frame, faces[i], Scalar(255, 0, 0), 5);
        Mat faceROI = frame_gray(faces[i]);
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        for (size_t j = 0; j < eyes.size(); j++)
        {
            rectangle(frame, eyes[j] + Point(faces[i].x, faces[i].y), Scalar(0, 255, 255), 5);
        }
        putText(frame, "FPS : " + to_string(fps), Point(0, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 8);
    }
    //-- Show what you got
    imshow("Capture - Face detection", frame);
}

// Part of code that runs on GPU
int gpumain()
{
    // Using milli as time unit for fps calculation
    using milli = std::chrono::milliseconds;
    double millisec, total_milli = 0;
    double fps = 0;
    int count = 0;
    String eye_cascade_name = opencv_path + "/data/haarcascades_cuda/haarcascade_eye.xml";
    String face_cascade_name = opencv_path + "/data/haarcascades_cuda/haarcascade_frontalface_alt.xml";
    
    // For Using video file, uncomment the below two lines and comment "VideoCapture cap(0);"
    // string filename = "videoplayback.mp4";
    // VideoCapture cap(filename);
    
    // open the Webcam
    VideoCapture cap(0);
    // if not success, exit program
    if (cap.isOpened() == false)
    {
        cout << "Cannot open Webcam" << endl;
        return -1;

    }

    // Note the cuda apis for GPU are different
    // Ensure that only the cuda cascades are used
    Ptr<cuda::CascadeClassifier> eye_cascade = cuda::CascadeClassifier::create(eye_cascade_name);
    Ptr<cuda::CascadeClassifier> face_cascade = cuda::CascadeClassifier::create(face_cascade_name);

    // Allocate variables in GPU memory
    cuda::GpuMat d_image;
    cuda::GpuMat roi_image;
    cuda::GpuMat d_buf;
    cuda::GpuMat faces_buf;
    std::vector<Rect> faces;
    std::vector<Rect> detections;


    //get the frames rate of the video from webcam
    double frames_per_second = cap.get(CAP_PROP_FPS);
    cout << "Frames per seconds : " << frames_per_second << endl;
    cout << "Press Q to Quit" << endl;
    String win_name = "Webcam Video";
    namedWindow(win_name); //create a window
    Mat frame_gray;
    cout << "Processing frames on a GPU\n";
    while (true)
    {

        Mat frame;
        Mat faceROI;
        bool flag = cap.read(frame); 
        // read a new frame from video
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // Time start for GPU compute
        auto start = std::chrono::high_resolution_clock::now();
        // Copies image data array from CPU to GPU memory
        d_image.upload(frame_gray);
        // Fills face buffers from images
        face_cascade->detectMultiScale(d_image, faces_buf);
        // Gets faces from face buffers
        face_cascade->convert(faces_buf, faces);
        for (int i = 0; i < faces.size(); i++)
        {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 5);
            faceROI = frame_gray(faces[i]);
            roi_image.upload(faceROI);
            // Fills eye buffers from images
            eye_cascade->detectMultiScale(roi_image, d_buf);
            // Gets eye data from eye buffers
            eye_cascade->convert(d_buf, detections);
            for (int j = 0; j < detections.size(); j++)
            {
                // Iterates for each face and identifies eyes.
                rectangle(frame, detections[j] + Point(faces[i].x, faces[i].y), Scalar(0, 255, 255), 5);
            }
            // Printing the fps on the image
            putText(frame, "FPS : " + to_string(fps), Point(0, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 8);
            imshow(win_name, frame);
        }
        // Time end for GPU compute
        auto finish = std::chrono::high_resolution_clock::now();
        millisec = std::chrono::duration_cast<milli>(finish - start).count();
        total_milli = total_milli + millisec;
        count = count + 1;
        if (count % (int)frames_per_second == 0)
        {
            fps = (double)(frames_per_second * 1000 / total_milli);
            cout << " FPS : " << fps << endl;
            total_milli = 0;
        }
        
        if (waitKey(1) == 'q')
        {
            break;
        }
    }
    return 0;
}

int main(int argc, const char** argv)
{
    if (!gpu_en)
    {
        cpumain(argv);
    }
    else
    {
        gpumain();
    }
    return 0;
}
