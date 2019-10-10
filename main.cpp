#include <QCoreApplication>

#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>

using namespace std;
using namespace cv;

#define RESIZE 1
#define W_img 640   // 640 320
#define H_img 480   // 480 240

void ReadVideo( string fn, Mat *img1, Mat *img2 )
{
    VideoCapture cap( fn );
    if( !cap.isOpened() )
        throw "Error when reading " + fn;
    cap.set( CAP_PROP_POS_MSEC, 309000 ); // 3300 136300 309000
    
    Mat frameRAW;
    while(1)
    {
        if ( !cap.read( frameRAW ) ) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
#if  (RESIZE == 1)
        resize( frameRAW, frameRAW, Size(W_img, H_img), 0, 0, INTER_LINEAR);
#endif
        imshow( "Real time", frameRAW );
        int button = waitKey();
        if ( button == 13 )                  // Take picture, press "Enter"
        {
            frameRAW.copyTo( *img1 );
            //for (int i=0; i<5; i++)
                cap.read( frameRAW );
#if  (RESIZE == 1)
            resize( frameRAW, frameRAW, Size(W_img, H_img), 0, 0, INTER_LINEAR);
#endif
            frameRAW.copyTo( *img2 );
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
    vector< float > mse_i(V.size(), 0.0f);
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
//                        mse_i[k] += abs(img1.at< uchar >(bp.y + j, bp.x + i) - img2.at< uchar >(delta.y + vS*V[k].y + j, delta.x + vS*V[k].x + i));
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
    //string file_name = "video.avi";
    string file_name = "/home/roman/Reconst_Stereo/zcm_logs/zcmlog-2019-09-23_L_05.avi";
    Mat img_in1, img_in2;
    ReadVideo( file_name, &img_in1, &img_in2 );
    imwrite( "img_in1.png", img_in1 );
    imwrite( "img_in2.png", img_in2 );
//    imshow( "img1", img_in1 );
//    imshow( "img2", img_in2 );
        // Convert to gray and blur
    Mat img_grey1, img_grey2;
    cvtColor( img_in1, img_grey1, COLOR_BGR2GRAY );
    cvtColor( img_in2, img_grey2, COLOR_BGR2GRAY );
    GaussianBlur( img_grey1, img_grey1, Size(3,3), 5, 0, BORDER_DEFAULT);
    GaussianBlur( img_grey2, img_grey2, Size(3,3), 5, 0, BORDER_DEFAULT);
    imwrite( "img_grey1.png", img_grey1 );
    imwrite( "img_grey2.png", img_grey2 );
    cout << "img.cols (width) = " << img_in1.cols << endl;
    cout << "img.rows (height) = " << img_in1.rows << endl;
    
        // Input block size (M x N)
    Size S_block;
    cout << "input M = ";
    cin >> S_block.width;
    cout << "input N = ";
    cin >> S_block.height;
    cout << "Block(M x N)= " << S_block.width << "x" << S_block.height << endl;
    
        // MSE & optical vectors
    Mat flow = Mat::zeros( img_in1.rows / S_block.height, img_in1.cols / S_block.width, CV_32FC2 );
    cout << "Flow.cols (width) = " << flow.cols << endl;
    cout << "Flow.rows (height) = " << flow.rows << endl;
    //Mat MSE = Mat::zeros( flow.size(), CV_32FC1 );
    Mat MSE = Mat::zeros( img_grey1.size(), CV_32FC1 );
    cout << "MSE.cols (width) = " << MSE.cols << endl;
    cout << "MSE.rows (height) = " << MSE.rows << endl;
    
    int start_num = 4;
    int Size_vector = start_num;    // Initial displacement vector
    double min_MSE = 0, max_MSE = 65025;
    Point2i min_MSE_point, max_MSE_point;
    Point2i offset_point, block_point;
    
        // Possible displacement vectors in a logarithmic search
    const vector< Point2i > Vcross = { Point(0,0), Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
    const vector< Point2i > Vsquare = { Point(0,0), Point(1,0), Point(1,1), Point(0,1), Point(-1,1), 
                                        Point(-1,0), Point(-1,-1), Point(0,-1), Point(1,-1) };

        // Calculate vectors
    imshow( "vec_flow", img_in2 );
    waitKey(50);
    //int but = waitKey(0);
    for ( int i = 0; i < flow.cols; i++ )
    {
        for ( int j = 0; j < flow.rows; j++ )
        {
            block_point.x = S_block.width * i;
            block_point.y = S_block.height * j;
            offset_point.x = S_block.width * i;
            offset_point.y = S_block.height * j;
            while ( Size_vector > 1 )
            {
                LogMSE( S_block, Vcross, block_point, offset_point, Size_vector, img_grey2, img_grey1, MSE );
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
                    Size_vector /= 2;
                }
            }
            LogMSE( S_block, Vsquare, block_point, offset_point, Size_vector, img_grey2, img_grey1, MSE );
//            cout << "MSE( " << offset_point.x << ", " << offset_point.y << " ): \n" << MSE << endl;
            minMaxLoc( MSE, &min_MSE, &max_MSE, &min_MSE_point, &max_MSE_point );
//            cout << "min: " << min_MSE << endl;
//            cout << "min P: " << min_MSE_point << endl;

            flow.at< Point2f >(j, i) = Point(min_MSE_point.x / S_block.width, min_MSE_point.y / S_block.height);    // Point2f-----!!!!!!!!!!!!!
            
            Size_vector = start_num;
        }
    }
    //cout << "Flow: " << endl << flow << endl;
    
        // Draw vectors
    Mat vectors = Mat::zeros(img_in1.size(), img_in1.type());
    for ( int i = 0; i < flow.cols; i++ )
    {
        for ( int j = 0; j < flow.rows; j++ )
        {
//            line( vectors,
//                  Point( cvRound( S_block.width*(i+0.5f) ), cvRound( S_block.height*(j+0.5f) ) ),     // cvRound()
//                  Point( cvRound( S_block.width*(flow.at<Point2f>(j,i).x+0.5f) ), cvRound( S_block.height*(flow.at<Point2f>(j,i).y+0.5f) ) ),
//                  Scalar(255, 100, 0) );        // H: 0-179, S: 0-255, V: 0-255
            line( vectors,
                  Point( int( S_block.width*(i+0.5f) ), int( S_block.height*(j+0.5f) ) ),     // cvRound()
                  Point( int( S_block.width*(flow.at<Point2f>(j,i).x+0.5f) ), int( S_block.height*(flow.at<Point2f>(j,i).y+0.5f) ) ),
                  Scalar(255, 100, 0) );        // H: 0-179, S: 0-255, V: 0-255
//            imwrite( "Flow.png", img_in2 );
        }
    }
    //click = waitKey(15);
    imwrite( "Flow.png", vectors );
    imshow( "vec_flow", vectors );
    cout << " --- vec_dif cmplited" << endl;

    
    
    
        // Recursive vector median filtering
    for ( int x = 0; x < flow.cols; x++ )
    {
        for ( int y = 0; y < flow.rows; y++ )
        {
            int num_block = 0;
            for ( size_t k = 1; k < 9; k++ )         // num block
            {
                if ( ( x + Vsquare[k].x <= flow.cols ) && ( y + Vsquare[k].y <= flow.rows ) && 
                     ( x + Vsquare[k].x >= 0 ) && ( y + Vsquare[k].y >= 0 ) )     // border check
                {
                    if ( !(flow.at< Point2f >(y, x) == Point2f(x, y)) )
                    {
                        
                        num_block ++;
                    }
                    
                    for ( int i = 0; i < 3; i++ )        // cols
                    {
                        for ( int j = 0; j < 3; j++ )   // rows
                        {
                            
                        }
                    }
                }
            }
        }
    }
    
    
    
    waitKey(0);
    MSE.release();
    flow.release();

    return 0; // a.exec();
}
