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

void LogMSE( const Size MN, vector< Point2i > V, Point2i bp, Point2i delta, int vS, Mat img1, Mat img2, Mat &mse )
{
    float square = MN.width * MN.height;            // square block
    mse = Mat::ones( mse.size(), CV_32FC1 );
    mse *= 65025;
    vector< float > mse_i(V.size(), 0.0);
    for ( size_t k = 0; k < V.size(); k++ )         // num block
    {
        if ( ( delta.x + vS*V[k].x <= mse.cols ) && ( delta.y + vS*V[k].y <= mse.rows ) && 
             ( delta.x + vS*V[k].x >= 0 ) && ( delta.y + vS*V[k].y >= 0 ) )     // border check
        {
            for ( int i = 0; i < MN.width; i++ )        // cols
            {
                for ( int j = 0; j < MN.height; j++ )   // rows
                {
                        mse_i[k] += (img1.at< uchar >(bp.y + j, bp.x + i) - img2.at< uchar >(delta.y + vS*V[k].y + j, delta.x + vS*V[k].x + i)) * 
                                    (img1.at< uchar >(bp.y + j, bp.x + i) - img2.at< uchar >(delta.y + vS*V[k].y + j, delta.x + vS*V[k].x + i));
                }
            }
            mse.at< float >(delta.y+vS*V[k].y, delta.x+vS*V[k].x) = mse_i[k] / square;
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
    
    int start_num = 4;
    int Size_vector = start_num;    // Initial displacement vector
    double min_MSE = 0, max_MSE = 65025;
    Point2i min_MSE_point, max_MSE_point;
    Point2i offset_point, block_point;
    
        // Possible displacement vectors in a logarithmic search
    const vector< Point2i > Vcross = { Point(0,0), Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
    const vector< Point2i > Vsquare = { Point(0,0), Point(1,0), Point(1,1), Point(0,1), Point(-1,1), 
                                        Point(-1,0), Point(-1,-1), Point(0,-1), Point(1,-1) };
    
    waitKey(100);
    for ( int i = 0; i < flow.cols; i++ )
    {
        for ( int j = 0; j < flow.rows; j++ )
        {
            block_point.x = i;
            block_point.y = j;
            offset_point.x = i;
            offset_point.y = j;
            while ( Size_vector > 1 )
            {
                LogMSE( S_block, Vcross, block_point, offset_point, Size_vector, img_grey1, img_grey2, MSE );
//                cout << "MSE( " << offset_point.x << ", " << offset_point.y << " ): \n" << MSE << endl;
                minMaxLoc( MSE, &min_MSE, &max_MSE, &min_MSE_point, &max_MSE_point );
//                cout << "min: " << min_MSE << endl;
//                cout << "min P: " << min_MSE_point << endl;
                
                if ( offset_point == min_MSE_point ) 
                {
                    Size_vector /= 2;
                }
                else
                {
                    offset_point = min_MSE_point;
                    Size_vector = start_num;
                }
            }
            LogMSE( S_block, Vsquare, block_point, offset_point, Size_vector, img_grey1, img_grey2, MSE );
//            cout << "MSE( " << offset_point.x << ", " << offset_point.y << " ): \n" << MSE << endl;
            minMaxLoc( MSE, &min_MSE, &max_MSE, &min_MSE_point, &max_MSE_point );
//            cout << "min: " << min_MSE << endl;
//            cout << "min P: " << min_MSE_point << endl;
            
            flow.at< Point2f >(j, i) = min_MSE_point;
            
            Size_vector = start_num;
        }
    }
    
    for ( int i = 0; i < flow.cols; i++ )
    {
        for ( int j = 0; j < flow.rows; j++ )
        {
            line( img_in2, 
                  Point( cvRound( S_block.width*(i+0.5f) ), cvRound( S_block.height*(j+0.5f) ) ),     // cvRound()
                  Point( cvRound( S_block.width*(flow.at<Point2f>(j,i).x+0.5f) ), cvRound( S_block.height*(flow.at<Point2f>(j,i).y+0.5f) ) ), 
                  Scalar(255, 100, 0) );        // H: 0-179, S: 0-255, V: 0-255
        }
    }
    imwrite( "Flow.png", img_in2 );
    
    waitKey();
    return 0; // a.exec();
}
