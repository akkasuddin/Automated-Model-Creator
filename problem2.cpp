#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<vector>
#include "ME566.h"
#include"RobustMatcher.h"

using namespace std;
using namespace cv;


// Globals ----------------------------------------------------------------------------------------

int boardHeight = 3;
int boardWidth = 4;
Size cbSize = Size(boardHeight,boardWidth);

string filename = "out_camera_data_1.xml";

//default image size
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

Mat intrinsics, distortion;
Mat r_descriptors;
vector<Point3f> r_points;

void addToList(vector<Point3f> c_points,cv::Mat descriptors_){

    int i = 0;
    if(r_points.size()==0){
         r_descriptors=descriptors_.clone();
        for(vector<Point3f>::iterator it = c_points.begin();it!=c_points.end();++it,++i){
            r_points.push_back(*it);
        }
    }
    else
    for(vector<Point3f>::iterator it = c_points.begin();it!=c_points.end();++it,++i){
        vector<Point3f>::iterator itt = r_points.begin();
        for(;itt!=r_points.end();++itt){
            if(dist(*itt,*it)<0.0001)
                break;
        }
        if(itt==r_points.end()){
            r_descriptors.push_back(descriptors_.row(i).clone());
            r_points.push_back(*it);
        }


    }
}


void writePoints(string path,vector<Point3f>& points){
    FileStorage storage(path,FileStorage::WRITE);
    storage<<"points"<<points;
    storage.release();

}

/** Save a CSV file and fill the object mesh */
void save(const std::string path)
{
    Point3f center;
    float n = (float) r_points.size();
    for(vector<Point3f>::iterator it = r_points.begin();it!=r_points.end();++it){
        center.x = center.x + (*it).x/n;
        center.y = center.y + (*it).y/n;
        center.z = center.z + (*it).z/n;

    }

    for(vector<Point3f>::iterator it = r_points.begin();it!=r_points.end();++it){
       *it = *it - center;
    }
    cout<<r_points;
    writePoints("points.yml",r_points);

  cv::Mat points3dmatrix = cv::Mat(r_points);
  //cv::Mat points2dmatrix = cv::Mat(list_points2d_in_);
  //cv::Mat keyPointmatrix = cv::Mat(list_keypoints_);

  cv::FileStorage storage(path, cv::FileStorage::WRITE);
  storage << "points_3d" << points3dmatrix;
  //storage << "points_2d" << points2dmatrix;
  //storage << "keypoints" << list_keypoints_;
  storage << "descriptors" << r_descriptors;


  storage.release();
  fstream f;
  f.open("matrix.m",ios_base::out);
  f<<"A = [";
  for(int i=0;i<r_points.size();i++){
      f<<r_points[i].x<<" "<<r_points[i].y<<" "<<r_points[i].z<<endl;
  }
  f<<"];"<<endl<<"scatter3(A(:,1),A(:,2),A(:,3))";
  f.close();
  cout<<endl<<"rpoints "<<r_points.size()<<"Descriptors: "<<r_descriptors.size()<<endl;

  //saveMatToCsv(points3dmatrix,"3dpoints");
}

//bool twoimages(Mat& images1,Mat& images2,Mat& cam0,Mat& cam1,Mat& descriptors,vector<Point3f>& keypoints,RobustMatcher& rmatcher, Point3d center, double radius){

 bool twoimages(Mat& images1,Mat& images2,Mat& cam0,Mat& cam1,Mat& descriptors,vector<Point3f>& keypoints,RobustMatcher& rmatcher, Point3d strt, Point3d end, bool&saveimages){

    //radius = radius*radius;
    vector<DMatch> goodmatches;
    Mat descriptorCandidates;
   //compute matches
    vector<KeyPoint>kps1,kps2;

    Mat disp1 = images1.clone();
    Mat disp2 = images2.clone();
    ;
    rmatcher.computeKeyPoints(images1,kps1);
    rmatcher.computeDescriptors(images1,kps1,descriptorCandidates);

    rmatcher.robustMatch(images2,goodmatches,kps2,descriptorCandidates);

    vector<Point2f>pts1,pts2;

    for(int i=0;i<goodmatches.size();i++){
        //if(kps1[goodmatches[i].trainIdx].pt.y>images[1].rows/2 || kps2[goodmatches[i].queryIdx].pt.y>images[1].rows/2 )
          //  continue;
        pts1.push_back(kps1[goodmatches[i].trainIdx].pt);
        pts2.push_back(kps2[goodmatches[i].queryIdx].pt);
        circle(disp1, kps1[goodmatches[i].trainIdx].pt,3,Scalar(0,0,255),2);
        circle(disp2, kps2[goodmatches[i].queryIdx].pt,3,Scalar(0,0,255),2);
    }
    imshow("img1",disp1);
    imshow("img2",disp2);
    if(saveimages){
    imwrite("img1.png",disp1);
    imwrite("img2.png",disp2);
    }
    if(pts1.size()==0){
        cout<<"No points found"<<endl;return false;
    }

    Mat pts3D_h;

    ;

    Mat normalizedpoints1,normalizedpoints2;
    undistortPoints(pts1,normalizedpoints1,intrinsics,distortion);
    undistortPoints(pts2,normalizedpoints2,intrinsics,distortion);

    triangulatePoints(cam0,cam1,normalizedpoints1,normalizedpoints2,pts3D_h);
   // cout<<pts3D_h;
    pts3D_h = pts3D_h.t();

    vector<Point3f> candidatePts;
    convertPointsHomogeneous(pts3D_h.reshape(4,1),candidatePts);

   // cout<<endl<<"points"<<pt_3d<<endl;
/*
    framePoints.push_back(Point3d(0,0,0));
    framePoints.push_back(Point3d(1,0,0));
    framePoints.push_back(Point3d(0,1,0));
    framePoints.push_back(Point3d(0,0,1));
*/

    Mat rvecLeft;
    Rodrigues( cam0(cv::Range(0,3), cv::Range(0,3)), rvecLeft);
    Mat tvecLeft(cam0(cv::Range(0,3), cv::Range(3,4)).t());

    vector<Point2f> projectedOnLeft;
    projectPoints(candidatePts, rvecLeft, tvecLeft, intrinsics, distortion, projectedOnLeft);




    Mat rvecRight;
    Rodrigues( cam1(cv::Range(0,3), cv::Range(0,3)), rvecRight);
    Mat tvecRight(cam1(cv::Range(0,3), cv::Range(3,4)).t());

    vector<Point2f> projectedOnRight;
    projectPoints(candidatePts, rvecRight, tvecRight, intrinsics, distortion, projectedOnRight);


    for(int j=0;j<projectedOnRight.size();j++){
        bool xbounds = (strt.x<candidatePts[j].x)&&(candidatePts[j].x<end.x);
        bool ybounds = (strt.y<candidatePts[j].y)&&(candidatePts[j].y<end.y);
        bool zbounds = (strt.z<candidatePts[j].z)&&(candidatePts[j].z<end.z);
        if(xbounds&&ybounds&&zbounds){

            descriptors.push_back(descriptorCandidates.row(j).clone());

            keypoints.push_back(candidatePts[j]);
            circle(disp1, projectedOnLeft[j],3,Scalar(0,255,0),2);
            circle(disp2, projectedOnRight[j],3,Scalar(0,255,0),2);
        }
    }
    cout<<keypoints;
    imshow("Reprojected1",disp2);

    imshow("Reprojected0",disp1);
    if(saveimages){
        imwrite("Reprojected1.png",disp2);
        imwrite("Reprojected0.png",disp1);
        saveimages = false;

    }
    //waitKey(0);
    //exit(0);

}

void initMatcher(RobustMatcher& rmatcher){
    int numKeyPoints = 10000;

    Ptr<FeatureDetector> detector =  ORB::create(numKeyPoints);
    rmatcher.setFeatureDetector(detector);
    rmatcher.setDescriptorExtractor(detector);

    Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1); // instantiate LSH index parameters
    Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);       // instantiate flann search parameters

    // instantiate FlannBased matcher
    Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    rmatcher.setDescriptorMatcher(matcher);                                                         // set matcher
    rmatcher.setRatio(0.7f);
}


int main(int argc, char** argv){
    set_working_path(argv[0]);

    FileStorage fs;
    fs.open(filename,FileStorage::READ);

    fs["camera_matrix"] >> intrinsics;
    fs["distortion_coefficients"] >> distortion;
    fs.release();
    vector<Mat> images;
    vector<Mat> camPose;
    Point3d center(0,9,6);
    double radius = 3;


    Mat frame,frameGr;
    Mat image1;



    vector<Point2d> imagePoints,imageFramePoints;
    vector<Point3d> boardPoints, framePoints;
    for (int i=0; i<boardWidth; i++)
        for (int j=0; j<boardHeight; j++)
            boardPoints.push_back( Point3d( double(i), double(j), 0.0) );

    framePoints.push_back(Point3d(0,0,0));
    framePoints.push_back(Point3d(1,0,0));
    framePoints.push_back(Point3d(0,1,0));
    framePoints.push_back(Point3d(0,0,1));


    RobustMatcher rmatcher;
    initMatcher(rmatcher);


/*
    for(int i=0;i<2;i++){

        Mat im = imread("img_"+toString(i)+".png");
        images.push_back(im.clone());
        Mat rvec, tvec;
        Mat R;
        Mat T(3,4,CV_64F);


        cvtColor(im,frameGr,COLOR_BGR2GRAY);
        findChessboardCorners(frameGr,Size(6,9),imagePoints,  CALIB_CB_NORMALIZE_IMAGE
                                                   + CALIB_CB_FAST_CHECK);
        solvePnP(Mat(boardPoints),Mat(imagePoints),intrinsics,distortion,rvec,tvec,false);

        Rodrigues(rvec,R);
        T( cv::Range(0,3), cv::Range(0,3) ) = R * 1; // copies R into T
        T( cv::Range(0,3), cv::Range(3,4) ) = tvec * 1; // copies tvec into T
        cout<<T<<endl;
        camPose.push_back(T.clone());

    }
    Mat descriptors;
    vector<Point3f> pt_3d;
    twoimages(images.rbegin()[0],images.rbegin()[1],camPose.rbegin()[0],images.rbegin()[1],descriptors,pt_3d, rmatcher, center,radius);
   // */
 //*
    VideoCapture vid(0);





    bool saveimages  = false;

    while(true){
        vid>>frame;
        cvtColor(frame,frameGr,COLOR_BGR2GRAY);
        bool found = findChessboardCorners(frameGr,Size(boardHeight,boardWidth),imagePoints,  CALIB_CB_ADAPTIVE_THRESH+ CALIB_CB_NORMALIZE_IMAGE
                                           + CALIB_CB_FAST_CHECK);
        if(found){
            Mat rvec, tvec,descriptors;
            vector<Point3f> pt_3d;
            solvePnP(Mat(boardPoints),Mat(imagePoints),intrinsics,distortion,rvec,tvec,false);
            projectPoints(framePoints,rvec,tvec,intrinsics,distortion,imageFramePoints);
            Mat R;
            Rodrigues(rvec,R);
            Mat T(3,4,R.type());
            T( cv::Range(0,3), cv::Range(0,3) ) = R * 1; // copies R into T
            T( cv::Range(0,3), cv::Range(3,4) ) = tvec * 1; // copies tvec into T


           if(images.size()>=1){
          //      twoimages(images.rbegin()[0],frame,camPose.rbegin()[0],T,descriptors,pt_3d, rmatcher, center,radius);
               twoimages(images.rbegin()[0],frame,camPose.rbegin()[0],T,descriptors,pt_3d, rmatcher, Point3d(0,4,0),Point3d(10,7,4),saveimages);
           //    twoimages(images.rbegin()[0],frame,camPose.rbegin()[0],T,descriptors,pt_3d, rmatcher, Point3d(-50,4,0),Point3d(50,50,10));

           }
           line(frame, imageFramePoints[0], imageFramePoints[1], CV_RGB(255,0,0), 2 );
           line(frame, imageFramePoints[0], imageFramePoints[2], CV_RGB(0,255,0), 2 );
           line(frame, imageFramePoints[0], imageFramePoints[3], CV_RGB(0,0,255), 2 );


           imshow("webcam",frame);

            int key = waitKey(10);
            if(key == 's'){
                images.push_back(frame.clone());
                camPose.push_back(T.clone());
                if(pt_3d.size()>1)
                    addToList(pt_3d,descriptors);
                cout<<r_points;
            }
            if(key == 'd')
                saveimages = true;

            else if(key==27)
                break;




        }




        imshow("webcam",frame);
        if(waitKey(1)==27)break;
    }
  // */
    save("model.yml");

  //  twoimages(images[0],images[1],camPose[0],camPose[1],rmatcher, center,radius);
}
