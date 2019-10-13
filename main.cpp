#include <QCoreApplication>

#include <vector>
#include <queue>
#include <math.h>
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

    // Структура для фильтрации векторов
struct L2_norm
{
    float L2;
    Point dxy;
    Point xy;
};
    // Перегруженный оператор для сортировки
struct less_then_L2
{
    inline bool operator() (const L2_norm& str1, const L2_norm& str2)
    {
        return (str1.L2 < str2.L2);
    }
};


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
        if ( (button == 13) || (button == 141) )                  // Take picture, press "Enter"
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

void filter_vectors( Mat &src, Mat &dst, vector< Point2i > V)
{
        // Recursive vector median filtering
    //dst = src.clone();
    for ( int x = 0; x < src.cols; x++ )
    {
        for ( int y = 0; y < src.rows; y++ )
        {
                // проверка нулевости вектора
            if ( !(src.at< Point2f >(y, x) == Point2f(x, y)) )
            {
                    // Колличество найденых ненулевых смещенных векторов в блоке
                int num_noZero = 0;
                vector< L2_norm > L2_block;
                for ( size_t k = 1; k < 9; k++ )         // num block, блоки смещения
                {
                    L2_norm Li;
                    if ( ( x + V[k].x <= src.cols ) && ( y + V[k].y <= src.rows ) && 
                         ( x + V[k].x >= 0 ) && ( y + V[k].y >= 0 ) )     // border check
                    {
                            // Проверка нулевости смещенного вектора
                        if ( !(src.at< Point2f >(y + V[k].y, x + V[k].x) == Point2f(x + V[k].x, y + V[k].y)) )    
                        {
                                // норма L2 между вектором и смещенным вектором 
                            Li.L2 = sqrt( (((x - src.at< Point2f >(y, x).x) - (V[k].x - src.at< Point2f >(y + V[k].y, x + V[k].x).x)) *
                                           ((x - src.at< Point2f >(y, x).x) - (V[k].x - src.at< Point2f >(y + V[k].y, x + V[k].x).x))) + 
                                          (((y - src.at< Point2f >(y, x).y) - (V[k].y - src.at< Point2f >(y + V[k].y, x + V[k].x).y)) *
                                           ((y - src.at< Point2f >(y, x).y) - (V[k].y - src.at< Point2f >(y + V[k].y, x + V[k].x).y))) );
                                // дельта координат меджу вектором и смещенным вектором
                            Li.dxy = V[k];
                                // значение смещенного вектора дижения 
                            Li.xy = src.at< Point2f >(y + V[k].y, x + V[k].x);
                            L2_block.push_back(Li);
                            num_noZero++ ;
                        }
                    }
                }
                if (num_noZero > 0)
                {
                        // Сортировка по норме L2
                    sort( L2_block.begin(), L2_block.end(), less_then_L2() );
                        // Расчет среднего элемента (вектора) и присвоение его вектору 
                    unsigned mid = unsigned( L2_block.size() / 2 );
                    float xdx = L2_block.at( mid ).xy.x - L2_block.at( mid ).dxy.x;
                    float ydy = L2_block.at( mid ).xy.y - L2_block.at( mid ).dxy.y;
                    if ( (xdx >= 0) && (xdx <= dst.cols) && (ydy >= 0) && (ydy <= dst.rows) )
                        dst.at< Point2f >(y, x) = Point2f( xdx, ydy );
                }
    //                else 
    //                {
    //                    dst.at< Point2f >(y, x) = Point2f(y, x);
    //                }
            }
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
    float progress = 0.0;
    int barWidth = 70;
    float step = 1.0f / flow.total();
    progress += step;
    cout << " --- Start vector calculation" << endl;
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
            
                // progress
            progress += step;
            std::cout << "[";
            int pos = int( float(barWidth) * progress );
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0f) << " %\r";
            std::cout.flush();
        }
    }
    cout << endl;
    
        // Draw vectors
    Mat vectors = Mat::zeros(img_in1.size(), img_in1.type());
    for ( int i = 0; i < flow.cols; i++ )
    {
        for ( int j = 0; j < flow.rows; j++ )
        {
            line( vectors,
                  Point( int( S_block.width*(i+0.5f) ), int( S_block.height*(j+0.5f) ) ),     // cvRound()
                  Point( int( S_block.width*(flow.at<Point2f>(j,i).x+0.5f) ), int( S_block.height*(flow.at<Point2f>(j,i).y+0.5f) ) ),
                  Scalar(255, 100, 0) );
        }
    }
    imwrite( "Flow.png", vectors );
    imshow( "vec_flow", vectors );
    cout << " --- vec_dif cmplited" << endl;
    
        // Recursive vector median filtering
    Mat flow_new = flow.clone();
    filter_vectors( flow, flow_new, Vsquare);
    
        // Draw new vectors
    vectors = Mat::zeros(img_in1.size(), img_in1.type());
    for ( int i = 0; i < flow_new.cols; i++ )
    {
        for ( int j = 0; j < flow_new.rows; j++ )
        {
            line( vectors,
                  Point( int( S_block.width*(i+0.5f) ), int( S_block.height*(j+0.5f) ) ),     // cvRound()
                  Point( int( S_block.width*(flow_new.at<Point2f>(j,i).x+0.5f) ), int( S_block.height*(flow_new.at<Point2f>(j,i).y+0.5f) ) ),
                  Scalar(255, 100, 0) );
        }
    }
    imwrite( "Flow_new.png", vectors );
    imshow( "vec_flow", vectors );
    cout << " --- New_vec_dif cmplited" << endl;
    
    
    
    waitKey(0);
    MSE.release();
    flow.release();

    return 0; // a.exec();
}
