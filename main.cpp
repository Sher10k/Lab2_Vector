#include <QCoreApplication>

#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>

using namespace std;
using namespace cv;

// Global variables
static int vx[5] = {0,  1,  0, -1,  0};     // Possible displacement vectors
static int vy[5] = {0,  0,  1,  0, -1};     // in a logarithmic search

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

void LogMSE( const Size MN, int dx, int dy, int vS, Mat *mse, Mat img1, Mat img2 )
{
    float square = MN.width * MN.height;            // square block
    *mse = Mat::ones( mse->size(), CV_32FC1 );
    *mse *= 65025;
    float mse0=0, mse1=0, mse2=0, mse3=0, mse4=0;
    for ( int i = 0; i < MN.width; i++ )        // cols
    {
        for ( int j = 0; j < MN.height; j++ )   // rows
        {
            mse0 += (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[0] + j, dx + vS*vx[0] + i)) * 
                    (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[0] + j, dx + vS*vx[0] + i));
            if ( dx + vS*vx[1] <= mse->cols )
                mse1 += (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[1] + j, dx + vS*vx[1] + i)) * 
                        (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[1] + j, dx + vS*vx[1] + i));
            if ( dy + vS*vy[2] <= mse->rows )
                mse2 += (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[2] + j, dx + vS*vx[2] + i)) * 
                        (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[2] + j, dx + vS*vx[2] + i));
            if ( dx + vS*vx[3] >= 0 )
                mse3 += (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[3] + j, dx + vS*vx[3] + i)) * 
                        (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[3] + j, dx + vS*vx[3] + i));
            if ( dy + vS*vy[4] >= 0 )
                mse4 += (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[4] + j, dx + vS*vx[4] + i)) * 
                        (img1.at< uchar >(dy + j, dx + i) - img2.at< uchar >(dy + vS*vy[4] + j, dx + vS*vx[4] + i));
        }
    }
    mse->at< float >(dy+vS*vy[0], dy+vS*vx[0]) = mse0 / square;
    mse->at< float >(dy+vS*vy[1], dy+vS*vx[1]) = mse1 / square;
    mse->at< float >(dy+vS*vy[2], dy+vS*vx[2]) = mse2 / square;
    mse->at< float >(dy+vS*vy[3], dy+vS*vx[3]) = mse3 / square;
    mse->at< float >(dy+vS*vy[4], dy+vS*vx[4]) = mse4 / square;
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
    cout << "img.cols (width) = " << img_in1.cols << endl;
    cout << "img.rows (height) = " << img_in1.rows << endl;
    
        // Input block size (M x N)
    Size S_block;
    cout << "input M= ";
    cin >> S_block.width;
    cout << "input N= ";
    cin >> S_block.height;
    cout << "Block(M x N)= " << S_block.width << "x" << S_block.height << endl;
    
        // MSE & optical vectors
    Mat flow = Mat::zeros( img_in1.rows / S_block.height, img_in1.cols / S_block.width, CV_32FC2 );
    Mat MSE = Mat::zeros( flow.size(), CV_32FC1 );
    
    int Dx = 0;             // Crosshair center offset
    int Dy = 0;
    int Size_vector = 5;    // Initial displacement vector
    float min_MSE = 0;
    int k = 0;              // Search hierarchy
    
    for ( int i = 0; i < flow.cols; i++ )
    {
        for ( int j = 0; j < flow.rows; j++ )
        {
            while ( Size_vector > 1 )
            {
                switch (k) 
                {
                    case 0:
                        break;
                    case 1:
                        break;
                    case 2:
                        break;
                    case 3:
                        break;
                    case 4:
                        break;
                }
            }
        }
    }
    
    
    waitKey();
    return 0; // a.exec();
}
