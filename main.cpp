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

void PaintVectors( Mat &flow, Mat &vectors, const Size MN )
{
    for ( int x = 0; x < flow.cols; x += MN.width )
    {
        for ( int y = 0; y < flow.rows; y += MN.height )
        {
            const Point2f flowatxy = flow.at< Point2f >(y, x) * 1;
            line( vectors,
                  Point( x + MN.width/2, y + MN.height/2 ),
                  Point( cvRound( x + flowatxy.x + MN.width/2 ), cvRound(y + flowatxy.y + MN.height/2 ) ),
                  Scalar(255, 100, 0) );
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
//                        mse_i[k] += abs(img1.at< uchar >(bp.y + y, bp.x + x) - img2.at< uchar >(delta.y + vS*V[k].y + y, delta.x + vS*V[k].x + x));
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
            if ( !(src.at< Point2f >(y, x) == Point2f(0, 0)) )
            {
                    // блок из суммы вычесленных растояний между векторами
                vector< L2_norm > d_block;
                for ( size_t k = 0; k < 9; k++ )         // num block, блоки смещения
                {
                    L2_norm di;
                    di.L2 = 0;
                    di.xy = Point(0,0);
                    di.dxy = Point(0,0);
                        // Border check
                    if ( ( x + MN.width*V[k].x <= src.cols ) && ( y + MN.height*V[k].y <= src.rows ) && 
                         ( x + MN.width*V[k].x >= 0 ) && ( y + MN.height*V[k].y >= 0 ) )
                    {
                            // save value vectors
                        di.xy = Point( x + MN.width*V[k].x, y + MN.height*V[k].y );    // point
                        di.dxy = Point( int( src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).x ), 
                                        int( src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x).y ) );  // offset
                        for ( size_t kk = 0; kk < 9; kk++ )     // 2й блок сещения
                        {
                                // пропускаем схожие смещения
                            if ( V[k] != V[kk] ) 
                            {
                                    // Border check
                                if ( ( x + MN.width*V[kk].x <= src.cols ) && ( y + MN.height*V[kk].y <= src.rows ) && 
                                     ( x + MN.width*V[kk].x >= 0 ) && ( y + MN.height*V[kk].y >= 0 ) )
                                {
                                        // Проверка нулевости смещенного вектора
                                    if ( !(src.at< Point2f >(y + MN.height*V[k].y, x + MN.width*V[k].x) == Point2f(0, 0)) && 
                                         !(src.at< Point2f >(y + MN.height*V[kk].y, x + MN.width*V[kk].x) == Point2f(0, 0)) )  
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
                    if ( d_block.at(i).dxy.x != 0 && d_block.at(i).dxy.y != 0) //( Point(d_block.at(i).dxy.x, d_block.at(i).dxy.y) != Point(0, 0) ) //( (d_block.at(i).dxy.x + d_block.at(i).dxy.y) > 0.0 )
                    {
                            // Запись только в центрльную ячейку
                        float xdx = d_block.at(i).dxy.x;
                        float ydy = d_block.at(i).dxy.y;
                        if ( (x + xdx >= 0) && (x + xdx <= dst.cols) && (y + ydy >= 0) && (y + ydy <= dst.rows) )
                            dst.at< Point2f >(y, x) = Point2f( xdx, ydy );
                        break;
                    }
                }
            }
        }
    }
}

int main()  // int argc, char *argv[]
{
        // Read image
    //string file_name = "video.avi";
    string file_name = "/home/roman/Reconst_Stereo/zcm_logs/zcmlog-2019-09-23_L_05.avi";
    Mat img_in1, img_in2;
    ReadVideo( file_name, &img_in1, &img_in2 );
    if ( img_in1.empty() || img_in2.empty() ) 
    {
        cout << " ! ! ! - Image is empty" << endl;
        exit(0);
    }
    imwrite( "img_in1.png", img_in1 );
    imwrite( "img_in2.png", img_in2 );
//    imshow( "img1", img_in1 );
//    imshow( "img2", img_in2 );
        // Convert to gray and blur
    Mat img_grey1, img_grey2;
    cvtColor( img_in1, img_grey1, COLOR_BGR2GRAY );
    cvtColor( img_in2, img_grey2, COLOR_BGR2GRAY );
    GaussianBlur( img_grey1, img_grey1, Size(3,3), 7, 0, BORDER_DEFAULT);
    GaussianBlur( img_grey2, img_grey2, Size(3,3), 7, 0, BORDER_DEFAULT);
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
    if ( S_block.width == 0 || S_block.height == 0 )
    {
        cout << " ! ! ! - M & N mast be > 0" << endl;
        exit(0);
    }
    
        // MSE & optical vectors
    Mat flow = Mat::zeros( img_grey1.size(), CV_32FC2 );
    Mat MSE = Mat::zeros( img_grey1.size(), CV_32FC1 );
    
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
    float progress = 0.0;
    int barWidth = 70;
    float step = 1.0f / flow.total() * S_block.width * S_block.height;
    progress += step;
    cout << " --- Start vector calculation" << endl;
    for ( int i = 0; i < flow.cols; i += S_block.width )
    {
        for ( int j = 0; j < flow.rows; j += S_block.height )
        {
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
            
            flow.at< Point2f >(j, i) = Point2f( i - min_MSE_point.x, j - min_MSE_point.y );
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
    PaintVectors( flow, vectors, S_block );
    imwrite( "Flow.png", vectors );
    //imshow( "vec_flow", vectors );
    cout << " --- vec_dif cmplited" << endl;
    
        // Recursive vector median filtering
    Mat flow_new = flow.clone();
    flow_new *= 0;
    for ( int i = 0; i < 2; i++ )
    {
        filter_vectors( flow, flow_new, S_block );
        flow_new.copyTo( flow );
    }
    
        // Draw new vectors
    vectors = Mat::zeros(img_in1.size(), img_in1.type());
    PaintVectors( flow_new, vectors, S_block );
    imwrite( "Flow_new.png", vectors );
    imshow( "vec_flow", vectors );
    cout << " --- New_vec_dif cmplited" << endl;
    
        // Clustering
    Mat cluster = Mat::zeros( flow_new.size(), CV_8UC1 );
    const vector< Point2i > V = { Point(-1,-1), Point(0,-1), Point(1,-1), Point(-1,0), Point(0,0), 
                                        Point(1,0), Point(-1,1), Point(0,1), Point(1,1) };
    uchar marker = 0;
    for ( int x = 0; x < cluster.cols; x += S_block.width )
    {
        for ( int y = 0; y < cluster.rows; y += S_block.height )
        {
            if ( cluster.at< uchar >(y, x) == 0 && flow_new.at< Point2f >(y, x) != Point2f(0, 0) )
            {
                marker++;
                cluster.at< uchar >(y, x) = marker;
            }
            for ( size_t k = 0; k < 9; k++ )         // num block, блоки смещения
            {
                    // Border check
                if ( ( x + S_block.width*V[k].x <= cluster.cols ) && ( y + S_block.height*V[k].y <= cluster.rows ) && 
                     ( x + S_block.width*V[k].x >= 0 ) && ( y + S_block.height*V[k].y >= 0 ) )
                {
                        // not zero vector & zero cluster
                    if ( flow_new.at< Point2f >(y + S_block.height*V[k].y, x + S_block.width*V[k].x) != Point2f(0, 0) && 
                         cluster.at< uchar >(y + S_block.height*V[k].y, x + S_block.width*V[k].x) == 0 )  
                    {
                        double V1V2 = double( ( flow_new.at< Point2f >(y, x).x ) * 
                                              ( flow_new.at< Point2f >(y + S_block.height*V[k].y, x + S_block.width*V[k].x).x ) + 
                                              ( flow_new.at< Point2f >(y, x).y ) * 
                                              ( flow_new.at< Point2f >(y + S_block.height*V[k].y, x + S_block.width*V[k].x).y ) );
                        double VV = sqrt( (pow( flow_new.at< Point2f >(y, x).x, 2 ) + 
                                           pow( flow_new.at< Point2f >(y, x).y, 2 )) * 
                                          (pow( flow_new.at< Point2f >(y + S_block.height*V[k].y, x + S_block.width*V[k].x).x, 2 ) + 
                                           pow( flow_new.at< Point2f >(y + S_block.height*V[k].y, x + S_block.width*V[k].x).y, 2 )) );
                        double sinA = V1V2 / VV;    // Range -1..1
                        
                            // Angle check
                        if ( sinA > 0 )
                            cluster.at< uchar >(y + S_block.height*V[k].y, x + S_block.width*V[k].x) = cluster.at< uchar >(y, x);
                    }
                }
            }
        }
    }
    cout << "Number of cluster: " << marker * 1.0 << endl;
    normalize( cluster, cluster, 255.0, 0.0, NORM_MINMAX );
    for ( int x = 0; x < cluster.cols; x += S_block.width )
        for ( int y = 0; y < cluster.rows; y += S_block.height )
            if ( cluster.at< uchar >(y, x) )
                for ( int i = 0; i < S_block.width; i++ )
                    for ( int j = 0; j < S_block.height; j++ )
                        cluster.at< uchar >(y+j, x+i) = cluster.at< uchar >(y, x);
    imshow( "Cluster", cluster );
    imwrite( "Cluster.png", cluster );
    
    threshold( cluster, cluster, 1, 255, THRESH_BINARY  );
    Mat img_temp = cluster.clone();
    Mat element = getStructuringElement( MORPH_RECT, Point(int(S_block.width*1.3), int(S_block.height*1.3)), Point(-1, -1) );
    morphologyEx( cluster, img_temp, MORPH_OPEN, element );
    morphologyEx( img_temp, cluster, MORPH_CLOSE, element );
    
    imshow( "Morph", cluster );
    imwrite( "Morph.png", cluster );
    
    vector< vector< Point > > contours;
    vector<Vec4i> hierarchy;
    findContours( cluster, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE );
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[unsigned(idx)][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( img_in1, contours, idx, color, 3, LINE_8, hierarchy );
    }
    imshow( "Conturs", img_in1 );
    imwrite( "Conturs.png", img_in1 );
    
    waitKey(0);
    MSE.release();
    flow.release();

    return 0;
}





// записываем значение полученного среднего вектора во все не нулевые вектора блока
//                        for ( size_t k = 0; k < 9; k++ )
//                        {
//                            if ( ( x + V[k].x <= src.cols ) && ( y + V[k].y <= src.rows ) && 
//                                 ( x + V[k].x >= 0 ) && ( y + V[k].y >= 0 ) )     // border check
//                            {
//                                float xdx = V[k].x + d_block.at(i).dxy.x;
//                                float ydy = V[k].y + d_block.at(i).dxy.y;
//                                if ( (x + xdx >= 0) && (x + xdx <= dst.cols) && (y + ydy >= 0) && (y + ydy <= dst.rows) )
//                                    dst.at< Point2f >(y + V[k].y, x + V[k].x) = Point2f( xdx, ydy );
//                            }
//                        }



//    Point2i Pxy;
//    Pxy = Point2i(1,2);
//    cout << "Pxy: " << Pxy << endl;
//    cout << "Pxy.x = " << Pxy.x << endl;
//    cout << "Pxy.y = " << Pxy.y << endl;
//    Pxy *= 2;
//    cout << "Pxy * 2" << endl;
//    cout << "Pxy.x = " << Pxy.x << endl;
//    cout << "Pxy.y = " << Pxy.y << endl;
    
//    Mat MPxy = Mat::zeros( 5, 8, CV_8UC1 );
//    cout << "MPxy = \n" << MPxy << endl;
//    cout << "MPxy.rows = " << MPxy.rows << endl;
//    cout << "MPxy.cols = " << MPxy.cols << endl;
    
//    MPxy.at< uchar >(1, 3) = 13;
//    cout << "MPxy = \n" << MPxy << endl;
//    MPxy.at< uchar >(Pxy.y, Pxy.x) = 5;
//    cout << "MPxy = \n" << MPxy << endl;
    
//    cout << "Pxy == Point2i(2, 4) = " << endl;
//    if ( Pxy == Point2i(2, 4) )
//        cout << "true" << endl;
//    else
//        cout << "false" << endl;
