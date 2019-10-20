#include <QCoreApplication>

#include <vector>
#include <queue>
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/optflow.hpp>

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
    vector< float > mse_i( V.size(), 0.0f );
    for ( size_t k = 0; k < V.size(); k++ )         // num block
    {
        if ( ( delta.x + vS*V[k].x <= mse.cols ) && ( delta.y + vS*V[k].y <= mse.rows ) && 
             ( delta.x + vS*V[k].x >= 0 ) && ( delta.y + vS*V[k].y >= 0 ) )     // border check
        {
            for ( int x = 0; x < MN.width; x++ )        // cols
            {
                for ( int y = 0; y < MN.height; y++ )   // rows
                {
                        mse_i[k] += (img1.at< uchar >(bp.y + y, bp.x + x) - img2.at< uchar >(delta.y + vS*V[k].y + y, delta.x + vS*V[k].x + x)) * 
                                    (img1.at< uchar >(bp.y + y, bp.x + x) - img2.at< uchar >(delta.y + vS*V[k].y + y, delta.x + vS*V[k].x + x));
//                        mse_i[k] += abs(img1.at< uchar >(bp.y + j, bp.x + i) - img2.at< uchar >(delta.y + vS*V[k].y + j, delta.x + vS*V[k].x + i));
                }
            }
            mse.at< float >(delta.y + vS*V[k].y, delta.x + vS*V[k].x) = mse_i[k] / square;
        }
    }
}

void filter_vectors( Mat &src, Mat &dst, const Size MN )
{
    const vector< Point2i > V = { Point(-1,-1), Point(0,-1), Point(1,-1), Point(-1,0), Point(0,0), 
                                        Point(1,0), Point(-1,1), Point(0,1), Point(1,1) };
        // Recursive vector median filtering
    //dst = src.clone();
    for ( int x = 0; x < src.cols; x += MN.width )
    {
        for ( int y = 0; y < src.rows; y += MN.height )
        {
                // проверка нулевости вектора
            if ( !(src.at< Point2f >(y, x) == Point2f(x, y)) )
            {
                    // блок из суммы вычесленных растояний между векторами
                vector< L2_norm > d_block;
                for ( size_t k = 0; k < 9; k++ )         // num block, блоки смещения
                {
                    L2_norm di;
                    di.L2 = 0;
                    di.xy = Point(0,0);
                    di.dxy = Point(0,0);
                    if ( ( x + MN.width*V[k].x <= src.cols ) && ( y + MN.height*V[k].y <= src.rows ) && 
                         ( x + MN.width*V[k].x >= 0 ) && ( y + MN.height*V[k].y >= 0 ) )     // border check
                    {
                            // save value vectors
                        di.xy = Point( x + MN.width*V[k].x, y + MN.height*V[k].y );    // point
                        di.dxy = Point( int(src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).x), 
                                        int(src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).y) );  // offset
                        for ( size_t kk = 0; kk < 9; kk++ )     // 2й блок сещения
                        {
                                // пропускаем схожие смещения
                            if ( V[k] != V[kk] ) 
                            {
                                if ( ( x + MN.width*V[kk].x <= src.cols ) && ( y + MN.height*V[kk].y <= src.rows ) && 
                                     ( x + MN.width*V[kk].x >= 0 ) && ( y + MN.height*V[kk].y >= 0 ) )     // border check
                                {
                                        // Проверка нулевости смещенного вектора
                                    if ( !(src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x) == Point2f(x + MN.width*V[k].x, y + MN.height*V[k].y)) && 
                                         !(src.at< Point2f >(y + MN.height*V[kk].y, x + MN.width*V[kk].x) == Point2f(x + MN.width*V[kk].x, y + MN.height*V[kk].y)) )    
                                    {
                                        di.L2 += sqrt( (((src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).x - (x + MN.width*V[k].x)) - (src.at< Point2f >(y + MN.height*V[kk].y, x + MN.width*V[kk].x).x - (x + MN.width*V[kk].x))) *
                                                        ((src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).x - (x + MN.width*V[k].x)) - (src.at< Point2f >(y + MN.height*V[kk].y, x + MN.width*V[kk].x).x - (x + MN.width*V[kk].x)))) + 
                                                       (((src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).y - (y + MN.height*V[k].y)) - (src.at< Point2f >(y + MN.height*V[kk].y, x + MN.width*V[kk].x).y - (y + MN.height*V[kk].y))) *
                                                        ((src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).y - (y + MN.height*V[k].y)) - (src.at< Point2f >(y + MN.height*V[kk].y, x + MN.width*V[kk].x).y - (y + MN.height*V[kk].y)))) );
                                        
                                    }
                                }
                            }
                        }
                    }
                    d_block.push_back(di);
                }
                    // Сортировка по норме L2
                sort( d_block.begin(), d_block.end(), less_then_L2() );
                for ( size_t i = 0; i < 9; i++ )
                {
                        // Отбрасываем нулевые элементы
                    if ( Point(d_block.at(i).dxy.x, d_block.at(i).dxy.y) != Point(0, 0) ) //( (d_block.at(i).dxy.x + d_block.at(i).dxy.y) > 0.0 )
                    {
                            // записываем значение полученного среднего вектора во все не нулевые вектора блока
//                        for ( size_t k = 0; k < 9; k++ )
//                        {
//                            if ( ( x + V[k].x <= src.cols ) && ( y + V[k].y <= src.rows ) && 
//                                 ( x + V[k].x >= 0 ) && ( y + V[k].y >= 0 ) )     // border check
//                            {
//                                float xdx = x + V[k].x + d_block.at(i).dxy.x;
//                                float ydy = y + V[k].y + d_block.at(i).dxy.y;
//                                if ( (xdx >= 0) && (xdx <= dst.cols) && (ydy >= 0) && (ydy <= dst.rows) )
//                                    dst.at< Point2f >(y + V[k].y, x + V[k].x) = Point2f( xdx, ydy );
//                            }
//                        }
                            // Запись только в центрльную ячейку
                        float xdx = x + d_block.at(i).dxy.x;
                        float ydy = y + d_block.at(i).dxy.y;
                        if ( (xdx >= 0) && (xdx <= dst.cols) && (ydy >= 0) && (ydy <= dst.rows) )
                            dst.at< Point2f >(y, x) = Point2f( xdx, ydy );
                        
                        break;
                    }
                }
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
//    GaussianBlur( img_grey1, img_grey1, Size(3,3), 5, 0, BORDER_DEFAULT);
//    GaussianBlur( img_grey2, img_grey2, Size(3,3), 5, 0, BORDER_DEFAULT);
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
    //Mat flow = Mat::zeros( img_in1.rows / S_block.height, img_in1.cols / S_block.width, CV_32FC2 );
    Mat flow = Mat::zeros( img_grey1.size(), CV_32FC2 );
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
    waitKey(70);
    float progress = 0.0;
    int barWidth = 70;
    float step = 1.0f / flow.total() * S_block.width * S_block.height;
    progress += step;
    cout << " --- Start vector calculation" << endl;
    for ( int i = 0; i < flow.cols; i += S_block.width )
    {
        for ( int j = 0; j < flow.rows; j += S_block.height )
        {
//            block_point.x = S_block.width * i;
//            block_point.y = S_block.height * j;
//            offset_point.x = S_block.width * i;
//            offset_point.y = S_block.height * j;
            block_point.x = i;
            block_point.y = j;
            offset_point.x = i;
            offset_point.y = j;
            while ( Size_vector > 1 )
            {
                LogMSE( S_block, Vcross, block_point, offset_point, Size_vector, img_grey2, img_grey1, MSE );
//                cout << "MSE( " << offset_point.x << ", " << offset_point.y << " ): \n" << MSE << endl;
                minMaxLoc( MSE, &min_MSE, &max_MSE, &min_MSE_point, &max_MSE_point );
//                cout << "min: " << min_MSE << endl;
//                cout << "min P: " << min_MSE_point << endl;
                
                Size_vector /= 2;
                if ( offset_point != min_MSE_point ) offset_point = min_MSE_point;
            }
            LogMSE( S_block, Vsquare, block_point, offset_point, Size_vector, img_grey2, img_grey1, MSE );
//            cout << "MSE( " << offset_point.x << ", " << offset_point.y << " ): \n" << MSE << endl;
            minMaxLoc( MSE, &min_MSE, &max_MSE, &min_MSE_point, &max_MSE_point );
//            cout << "min: " << min_MSE << endl;
//            cout << "min P: " << min_MSE_point << endl;
            
            flow.at< Point2f >(j, i) = Point2f( i - min_MSE_point.x, j - min_MSE_point.y );     // ??????
            //flow.at< Point2f >(j, i) = Point2f( j - min_MSE_point.y, i - min_MSE_point.x );
            
            Size_vector = start_num;
            
                // progress
            progress += step;
            cout << "[";
            int pos = int( float(barWidth) * progress );
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) cout << "=";
                else if (i == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << int(progress * 100.0f) << " %\r";
            cout.flush();
        }
    }
    cout << endl;
    
    //optflow::calcOpticalFlowSparseToDense( img_grey2, img_grey1, flow, 4, 128, 0.1f, true, 500.0f, 1.5f);
    
        // Draw vectors
    Mat vectors = Mat::zeros(img_in1.size(), img_in1.type());
    for ( int x = 0; x < flow.cols; x += S_block.width )
    {
        for ( int y = 0; y < flow.rows; y += S_block.height )
        {
            const Point2f flowatxy = flow.at< Point2f >(y, x) * 1;
//            line( vectors,
//                  Point( int( S_block.width*(x+0.5f) ), int( S_block.height*(y+0.5f) ) ),     // cvRound()
//                  Point( int( S_block.width*(flow_new.at<Point2f>(y,x).x+0.5f) ), int( S_block.height*(flow_new.at<Point2f>(y,x).y+0.5f) ) ),
//                  Scalar(255, 100, 0) );
            line( vectors,
                  Point( x, y ),     // cvRound()
                  Point( cvRound( x + flowatxy.x), cvRound(y + flowatxy.y ) ),
                  Scalar(255, 100, 0) );
        }
    }
    imwrite( "Flow.png", vectors );
    imshow( "vec_flow", vectors );
    cout << " --- vec_dif cmplited" << endl;
    
        // Recursive vector median filtering
    Mat flow_new = flow.clone();
    filter_vectors( flow, flow_new, S_block );
//    flow_new.copyTo(flow);
//    filter_vectors( flow, flow_new, S_block );
    
        // Draw new vectors
    vectors = Mat::zeros(img_in1.size(), img_in1.type());
    for ( int x = 0; x < flow_new.cols; x += S_block.width )
    {
        for ( int y = 0; y < flow_new.rows; y += S_block.height )
        {
            const Point2f flowatxy = flow_new.at< Point2f >(y, x) * 1;
//            line( vectors,
//                  Point( int( S_block.width*(x+0.5f) ), int( S_block.height*(y+0.5f) ) ),     // cvRound()
//                  Point( int( S_block.width*(flow_new.at<Point2f>(y,x).x+0.5f) ), int( S_block.height*(flow_new.at<Point2f>(y,x).y+0.5f) ) ),
//                  Scalar(255, 100, 0) );
            line( vectors,
                  Point( x, y ),     // cvRound()
                  Point( cvRound(x + flowatxy.x), cvRound(y + flowatxy.y) ),
                  Scalar(255, 100, 0) );
        }
    }
    imwrite( "Flow_new.png", vectors );
    imshow( "vec_flow", vectors );
    cout << " --- New_vec_dif cmplited" << endl;
    
        // Clustering
    
    
    
    waitKey(0);
    MSE.release();
    flow.release();

    return 0; // a.exec();
}
