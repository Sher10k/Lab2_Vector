#include <QCoreApplication>

#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void ReadVideo( string &fn, Mat &img1, Mat &img2 )
{
    VideoCapture cap( fn );
    if( !cap.isOpened() )
            throw "Error when reading " + fn;
    cap.set( CAP_PROP_POS_MSEC, 3300 );
    
    Mat frameRAW;
    while(1)
    {
        if ( !cap.read( frameRAW ) ) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        imshow( "Real time", frameRAW );
        int button = waitKey();
        if ( button == 13 )                  // Take picture, press "Enter"
        {
            frameRAW.copyTo( img1 );
            cap.read( frameRAW );
            frameRAW.copyTo( img2 );
        }
        else if ( button == 27 ) 
        {
            cap.release();
            destroyWindow( "Real time" );
            break;      // Interrupt the cycle, press "ESC"
        }
    }
    
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
        
        // Read image
    string file_name = "video.avi";
    Mat img_in1, img_in2;
    ReadVideo( file_name, img_in1, img_in2 );
    imwrite( "img_in1.png", img_in1 );
    imwrite( "img_in2.png", img_in2 );
//    imshow( "img1", img_in1 );
//    imshow( "img2", img_in2 );
        // Convert to gray
    Mat img_grey1, img_grey2;
    cvtColor( img_in1, img_grey1, COLOR_BGR2GRAY );
    cvtColor( img_in2, img_grey2, COLOR_BGR2GRAY );
    imwrite( "img_grey1.png", img_grey1 );
    imwrite( "img_grey2.png", img_grey2 );
    
        // Grid
    Mat img_MSE = Mat::zeros( img_in1.size(), img_in1.type() );
    
    
    
    waitKey();
    return 0; // a.exec();
}
