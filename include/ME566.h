#ifndef _ME566_H_
#define _ME566_H_

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;
#ifdef __linux__
#include <unistd.h>
#endif


inline bool set_working_path(char *argv)
{
#ifdef __linux__
    string ss = string(argv);
    chdir(ss.substr(0,ss.rfind('/')).c_str());

    return true;
#endif
   return false;
}


/**
 * @brief Converts a number to string object
 * @param any datatype supported by ostringstream
 */
template <typename T>
string toString(T number){
    ostringstream ss;
    ss<<number;
    return ss.str();

}



inline int dist(const KeyPoint &a, const KeyPoint &b)
{
    return (a.pt.x - b.pt.x)*(a.pt.x - b.pt.x) + (a.pt.y - b.pt.y)*(a.pt.y - b.pt.y);
}

inline double dist(const Point3d &a, const Point3d &b)
{

    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
}


/**
 * @brief Performs adaptive non maximal suppression on a vector of keypoints
 * @author Akkas Uddin Haque
 * @param keyPoints : vector of KeyPoint data to apply max suppression on
 * @param numberOfPoints : number of output KeyPoints
 *
 */
inline void anms(vector<KeyPoint> &keyPoints,int numberOfPoints)
{
    //sort keypoints according to response
    sort(keyPoints.begin(),keyPoints.end(),[](const KeyPoint&a,const KeyPoint&b){return a.response>b.response;});

    /*for(int i=0;i<keyPoints.size();i++)
    {
        cout<<keyPoints[i].response<<endl;

    }//*/

    //<distance,index_of_keypoint> vector to store minimum suppressed radii
    vector<pair<int,int> > radius;
    radius.reserve(keyPoints.size());

    //robust
    float Crobust = 0.9;
    //first keypoint is always in the final list
    radius.push_back(make_pair<int,int>(INT_MAX,0));


    for(int i=1;i<keyPoints.size()  ;i++)
    {
        //start with infinitely large radius
        int minRadius=INT_MAX;
        int idx=-1;
        //iterate from first keypoint in output list to current position
         for(int j = 0; j < i; j++ ){
             //calculate distance
            int distCalc = dist(keyPoints[i],keyPoints[j]);
            if(distCalc<minRadius && keyPoints[i].response<Crobust*keyPoints[j].response){
                idx = j;
                minRadius = distCalc;
            }

        }
        if(idx!=-1)
        radius.push_back(make_pair(minRadius,i));

    }

    //sort raduis descending
    sort(radius.begin(),radius.end(),[](const pair<int,int>&a,const pair<int,int>&b){return a.first>b.first;});


    //select only required number of non maximally suppressed keypoints
    vector<KeyPoint> filteredKeypoints;
    for(int i=0;i<numberOfPoints;i++)
       filteredKeypoints.push_back(keyPoints[radius[i].second]);

    keyPoints = filteredKeypoints;


}


#endif
