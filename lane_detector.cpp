#include <opencv2/opencv.hpp>  
#include <vector>  
#include <deque>  
#include <algorithm>  
#include <iostream>  
#include <ros/ros.h>  
#include <std_msgs/Float64.h>  

using namespace cv;  
using namespace std;  

// ----- 参数配置 -----  
static const int SMOOTH_WINDOW   = 5;    // 历史帧窗口大小，可调 3~10  
static const int POINT_SMOOTH    = 7;    // 单帧点滑动平均窗口  
static const Size GAUSSIAN_KSIZE = Size(5, 5);  
static const int BEZIER_DEGREE   = 3;    // 贝塞尔曲线阶数（3为三次贝塞尔）  
static const int BEZIER_STEPS    = 100;  // 曲线采样点数  
  
// 历史缓存：用于保存连续多帧的数据  
static deque<vector<Point>> leftHistory, rightHistory, midHistory;  
  
// --- 颜色分割：提取黄色与黑色边界 ---  
void extractColorMask(const Mat& src, Mat& mask, Mat& maskedColor) {  
    Mat hsv, yellowMask, blackMask;  
    // 转换为HSV色彩空间，便于颜色检测  
    cvtColor(src, hsv, COLOR_BGR2HSV);  
    // 提取黄色部分（HSV范围：[15, 100, 100] 到 [35, 255, 255]）  
    inRange(hsv, Scalar(15,100,100), Scalar(35,255,255), yellowMask);  
    // 提取黑色部分（HSV范围：[0, 0, 0] 到 [180, 255, 50]）  
    inRange(hsv, Scalar(0,0,0), Scalar(180,255,50), blackMask);  
    // 合并黄色与黑色区域  
    bitwise_or(yellowMask, blackMask, mask);  
    // 形态学操作，去除噪声  
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));  
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,-1), 2);  
    // 提取原图中黄色和黑色区域  
    bitwise_and(src, src, maskedColor, mask);  
}  
  
// --- 边缘检测：提取左边和右边的边缘点 ---  
void detectEdgePoints(const Mat& bin, vector<Point>& leftPts, vector<Point>& rightPts) {  
    int w = bin.cols, h = bin.rows;  
    vector<int> whiteLen(w,0);  
    // 计算每一列白色像素的高度  
    for(int x=0;x<w;x++) for(int y=h-1;y>=0;y--){ if(bin.at<uint8_t>(y,x)) whiteLen[x]++; else break; }  
    // 寻找左右两边的初始点  
    int midX = w/2;  
    int l0 = max_element(whiteLen.begin(), whiteLen.begin()+midX) - whiteLen.begin();  
    int r0 = max_element(whiteLen.begin()+midX, whiteLen.end()) - whiteLen.begin();  
    // 从下往上遍历，提取边缘点  
    for(int y=h-1;y>=0;y--) {  
        for(int x=l0; x>0; x--) if(bin.at<uint8_t>(y,x) && !bin.at<uint8_t>(y,x-1)) { leftPts.emplace_back(x,y); break; }  
        for(int x=r0; x<w-1; x++) if(bin.at<uint8_t>(y,x) && !bin.at<uint8_t>(y,x+1)) { rightPts.emplace_back(x,y); break; }  
    }  
}  
  
// --- 单帧点滑动平均抑抖 ---  
vector<Point> smoothPoints(const vector<Point>& pts, int win = POINT_SMOOTH) {  
    int n = pts.size(); if(!n) return {};  
    win = min(win, n);  
    int half = win/2;  
    vector<Point> out; out.reserve(n);  
    // 滑动平均算法：对每个点进行平滑处理  
    for(int i=0; i<n; i++) {  
        int s = max(0, i-half), e = min(n-1, i+half), sumX=0, c=e-s+1;  
        for(int j=s; j<=e; j++) sumX += pts[j].x;  
        out.emplace_back(sumX/c, pts[i].y);  
    }  
    return out;  
}  
  
// --- 历史窗口线性插值补全 ---  
int interpolateX(const vector<Point>& pts, int y) {  
    Point a(0,-1), b(0,-1);  
    // 在历史窗口中查找与目标y值最接近的点  
    for(auto &p : pts) {  
        if(p.y <= y && (a.y<0 || p.y>a.y)) a = p;  
        if(p.y >= y && (b.y<0 || p.y<b.y)) b = p;  
    }  
    // 如果找到两个点，进行线性插值  
    if(a.y>=0 && b.y>=0 && a.y!=b.y) {  
        float t = float(y - a.y) / (b.y - a.y);  
        return cvRound(a.x + t*(b.x-a.x));  
    }  
    return a.y>=0 ? a.x : (b.y>=0 ? b.x : -1);  
}  
// --- 历史缓存点平滑与填充 ---  
vector<Point> smoothAndFill(deque<vector<Point>>& hist, int H) {  
    vector<Point> out;  
    if(hist.empty()) return out;  
    // 从历史窗口中提取并平滑每一行的点  
    for(int y=H-1; y>=0; y--) {  
        vector<int> xs;  
        for(auto &pts : hist) for(auto &p : pts) if(p.y==y) xs.push_back(p.x);  
        int x = -1;  
        // 对每一行进行中值滤波  
        if(!xs.empty()) {  
            sort(xs.begin(), xs.end());  
            x = xs[xs.size()/2];  
        } else x = interpolateX(hist.back(), y);  
        if(x>=0) out.emplace_back(x, y);  
    }  
    return out;  
}  
  
// --- 生成贝塞尔曲线采样点 ---  
vector<Point> generateBezier(const vector<Point>& ctrlPts) {  
    vector<Point> curve;  
    int n = ctrlPts.size(); if(n < BEZIER_DEGREE+1) return curve;  
    // 计算贝塞尔曲线的采样点  
    for(int step = 0; step <= BEZIER_STEPS; ++step) {  
        double t = double(step) / BEZIER_STEPS;  
        vector<Point2f> tmp(ctrlPts.begin(), ctrlPts.end());  
        // de Casteljau算法计算贝塞尔曲线  
        for(int r=1; r<=BEZIER_DEGREE; ++r) {  
            for(int i=0; i<=BEZIER_DEGREE-r; ++i) {  
                tmp[i].x = (1-t)*tmp[i].x + t*tmp[i+1].x;  
                tmp[i].y = (1-t)*tmp[i].y + t*tmp[i+1].y;  
            }  
        }  
        curve.emplace_back(cvRound(tmp[0].x), cvRound(tmp[0].y));  
    }  
    return curve;  
}  
  
// --- 约束点在赛道区域内 ---  
void clampToMask(const Mat& mask, Point& pt) {  
    int w = mask.cols, h = mask.rows;  
    pt.x = min(max(pt.x, 0), w-1);  
    pt.y = min(max(pt.y, 0), h-1);  
    // 检查点是否在有效区域内，如果不在，则通过查找附近有效区域来修正点  
    if(!mask.at<uint8_t>(pt.y, pt.x)) {  
        int radius = 1;  
        Point best = pt;  
        bool found = false;  
        while(!found && radius < max(w,h)) {  
            for(int dy=-radius; dy<=radius; ++dy) for(int dx=-radius; dx<=radius; ++dx) {  
                int nx = pt.x + dx, ny = pt.y + dy;  
                if(nx>=0&&nx<w&&ny>=0&&ny<h && mask.at<uint8_t>(ny,nx)) {  
                    best = Point(nx,ny); found = true; break;  
                }  
            }  
            radius++;  
        }  
        pt = best;  
    }  
}  
  
// 主函数：实现赛道偏移检测与可视化  
int main(int argc, char** argv) {  
    ros::init(argc, argv, "center_offset_publisher");  
    ros::NodeHandle nh;
    ros::Publisher offset_pub = nh.advertise<std_msgs::Float64>("/center_offset", 10);  // 创建发布器，用于发送中心偏移数据

    // 打开视频流，使用摄像头捕获图像
    VideoCapture cap(0);  
    if (!cap.isOpened()) { 
        cerr << "错误: 无法打开摄像头" << endl; 
        return -1; 
    }  
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);  // 设置摄像头分辨率为1280x720
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);  
    namedWindow("Result", WINDOW_AUTOSIZE);  // 创建窗口用于显示结果图像

    while (ros::ok()) {  // 循环读取图像并处理
        Mat frame; 
        cap >> frame;  // 从摄像头获取一帧图像
        if (frame.empty()) continue;  // 如果图像为空，跳过当前帧

        Mat mask, maskedColor;  
        extractColorMask(frame, mask, maskedColor);  // 提取黄色和黑色边界，生成掩膜图像

        vector<Point> l0, r0;  
        detectEdgePoints(mask, l0, r0);  // 通过边缘检测提取左右边缘点
        auto l1 = smoothPoints(l0);  // 对左边点进行平滑
        auto r1 = smoothPoints(r0);  // 对右边点进行平滑

        // 将当前帧的边缘点添加到历史缓存中，进行窗口平滑
        leftHistory.push_back(l1); 
        if (leftHistory.size() > SMOOTH_WINDOW) leftHistory.pop_front();  
        rightHistory.push_back(r1); 
        if (rightHistory.size() > SMOOTH_WINDOW) rightHistory.pop_front();  

        // 平滑历史缓存中的边缘点，并填充缺失的点
        auto leftPts = smoothAndFill(leftHistory, mask.rows);  
        auto rightPts = smoothAndFill(rightHistory, mask.rows);  

        vector<Point> mid;  
        int N = min(leftPts.size(), rightPts.size());  
        // 计算中线点：左边和右边的中间点
        for (int i = 0; i < N; ++i) 
            mid.emplace_back((leftPts[i].x + rightPts[i].x) / 2, leftPts[i].y);  
        
        // 将当前的中线点添加到历史缓存中
        midHistory.push_back(mid);  
        if (midHistory.size() > SMOOTH_WINDOW) midHistory.pop_front();  

        // 平滑历史缓存中的中线点
        auto midPts = smoothAndFill(midHistory, mask.rows);  

        // 计算中线偏移值（以图像中心为参考）
        std_msgs::Float64 offset_msg;  
        if (!midPts.empty()) {  
            int centerX = frame.cols / 2;  // 获取图像的中心位置
            offset_msg.data = midPts[0].x - centerX;  // 计算中线的偏移量
            offset_pub.publish(offset_msg);  // 发布偏移值
            cout << "Center Offset: " << offset_msg.data << endl;  // 输出偏移值
        }  

        // 根据历史点生成贝塞尔曲线的控制点
        vector<Point> ctrlPts;  
        if (N >= BEZIER_DEGREE + 1) {  
            int step = N / (BEZIER_DEGREE + 1);  
            for (int i = 0; i <= BEZIER_DEGREE; ++i) 
                ctrlPts.push_back(midPts[i * step]);  
            for (auto& p : ctrlPts) clampToMask(mask, p);  // 限制控制点在赛道区域内
        }  
        auto bezierCurve = generateBezier(ctrlPts);  // 生成贝塞尔曲线

        // 将最终结果绘制到图像上
        Mat out = maskedColor.clone();  
        // 绘制左边点
        for (auto& p : leftPts) 
            circle(out, p, 2, Scalar(0, 0, 255), -1);  
        // 绘制右边点
        for (auto& p : rightPts) 
            circle(out, p, 2, Scalar(0, 255, 0), -1);  
        // 绘制贝塞尔曲线
        for (int i = 1; i < bezierCurve.size(); ++i) 
            line(out, bezierCurve[i - 1], bezierCurve[i], Scalar(255, 0, 0), 2);  

        // 显示最终结果
        imshow("Result", out);  
        if (waitKey(1) == 27) break;  // 按下Esc键退出

        ros::spinOnce();  // 处理ROS消息队列
    }  
    return 0;  
}