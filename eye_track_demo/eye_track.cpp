/*
 * @Filename: demo_face.cpp
 * @Description: Pupil tracking + gaze estimation on A1 dual sensor
 * @Date: 2026-04-26
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include "include/utils.hpp"

using namespace std;

#define COLOR_GREEN   2
#define COLOR_GRAY    16
#define COLOR_CURSOR  12
#define COLOR_DIR     0

static const std::array<float,2> CALIB_POINTS[9] = {
    {0.1f,0.1f},{0.5f,0.1f},{0.9f,0.1f},
    {0.1f,0.5f},{0.5f,0.5f},{0.9f,0.5f},
    {0.1f,0.9f},{0.5f,0.9f},{0.9f,0.9f}
};

enum GazeDirection {
    GAZE_CENTER=0,GAZE_LEFT,GAZE_RIGHT,
    GAZE_UP,GAZE_DOWN,
    GAZE_LEFT_UP,GAZE_RIGHT_UP,
    GAZE_LEFT_DOWN,GAZE_RIGHT_DOWN
};

const char* GazeDirectionStr[] = {
    "center","left","right","up","down",
    "left-up","right-up","left-down","right-down"
};

GazeDirection ClassifyGaze(float nx, float ny) {
    bool l=nx<0.35f, r=nx>0.65f, u=ny<0.35f, d=ny>0.65f;
    if(l&&u) return GAZE_LEFT_UP;
    if(r&&u) return GAZE_RIGHT_UP;
    if(l&&d) return GAZE_LEFT_DOWN;
    if(r&&d) return GAZE_RIGHT_DOWN;
    if(l) return GAZE_LEFT;
    if(r) return GAZE_RIGHT;
    if(u) return GAZE_UP;
    if(d) return GAZE_DOWN;
    return GAZE_CENTER;
}

bool FindPupilClassic(const uint8_t* gray, int w, int h, float& nx, float& ny) {
    if(!gray||w<=0||h<=0) return false;
    long sum=0;
    for(int i=0;i<w*h;i++) sum+=gray[i];
    float mean=(float)sum/(w*h);
    float thr=mean*0.55f;
    float cx=0,cy=0; int cnt=0;
    for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
            if(gray[y*w+x]<thr){cx+=x;cy+=y;cnt++;}
    if(cnt<4) return false;
    nx=(cx/cnt)/(float)w; ny=(cy/cnt)/(float)h;
    return true;
}

class PupilDetector {
public:
    void Initialize(int w, int h) {
        img_w_=w; img_h_=h;
        linux_tensor_=create_tensor(w,h,SSNE_Y_8,SSNE_BUF_LINUX);
        buf_size_=w*h;
        img_buf_=new uint8_t[buf_size_];
        printf("[PupilDetector] Initialized (classic mode)\n");
    }
    bool Predict(ssne_tensor_t* img, std::array<float,4>& eye_box,
                 float& out_x, float& out_y) {
        float ew=eye_box[2]-eye_box[0], eh=eye_box[3]-eye_box[1];
        if(ew<4||eh<4) return false;
        if(copy_tensor_buffer(*img,linux_tensor_)!=SSNE_ERRCODE_NO_ERROR) return false;
        if(save_tensor_buffer_ptr(linux_tensor_,img_buf_,buf_size_)!=SSNE_ERRCODE_NO_ERROR) return false;
        int ex1=(int)std::max(0.f,eye_box[0]);
        int ey1=(int)std::max(0.f,eye_box[1]);
        int ex2=(int)std::min((float)img_w_,eye_box[2]);
        int ey2=(int)std::min((float)img_h_,eye_box[3]);
        int ew2=ex2-ex1, eh2=ey2-ey1;
        if(ew2<4||eh2<4) return false;
        std::vector<uint8_t> eye_buf(ew2*eh2);
        for(int y=0;y<eh2;y++)
            memcpy(eye_buf.data()+y*ew2, img_buf_+(ey1+y)*img_w_+ex1, ew2);
        float nx,ny;
        if(!FindPupilClassic(eye_buf.data(),ew2,eh2,nx,ny)) return false;
        out_x=eye_box[0]+nx*ew;
        out_y=eye_box[1]+ny*eh;
        return true;
    }
    void Release() {
        release_tensor(linux_tensor_);
        delete[] img_buf_;
    }
private:
public:
    uint8_t* img_buf_=nullptr;
    int img_w_=0, img_h_=0;
private:
    ssne_tensor_t linux_tensor_;
    int buf_size_=0;
};

std::array<float,4> GetEyeBox(const std::array<float,4>& face,
                               float cx, float cy, float iw, float ih) {
    float fw=face[2]-face[0];
    float hw=fw*0.20f, hh=hw*0.6f;
    float x1=cx-hw, y1=cy-hh, x2=cx+hw, y2=cy+hh;
    if(x1<0){x2-=x1;x1=0;}
    if(y1<0){y2-=y1;y1=0;}
    if(x2>iw){x1-=(x2-iw);x2=iw;}
    if(y2>ih){y1-=(y2-ih);y2=ih;}
    return {std::max(0.f,x1),std::max(0.f,y1),x2,y2};
}

struct CalibSample { float ox,oy; };

class GazeCalibrator {
public:
    bool is_calibrating=false, is_calibrated=false;
    int current_point=0, sample_count=0;
    float sample_ox=0, sample_oy=0;
    static const int SAMPLES_PER_POINT=30;
    CalibSample calib_data[9];
    float scale_x=8.f,scale_y=8.f,offset_x=0.f,offset_y=0.f;

    void StartCalibration(){
        is_calibrating=true; is_calibrated=false;
        current_point=0; sample_count=0; sample_ox=sample_oy=0;
        printf("[Calib] Start calibration\n");
    }
    bool AddSample(float ox, float oy){
        if(!is_calibrating) return false;
        sample_ox+=ox; sample_oy+=oy; sample_count++;
        if(sample_count>=SAMPLES_PER_POINT){
            calib_data[current_point]={sample_ox/SAMPLES_PER_POINT,
                                       sample_oy/SAMPLES_PER_POINT};
            printf("[Calib] Point %d done\n",current_point+1);
            current_point++; sample_count=0; sample_ox=sample_oy=0;
            if(current_point>=9){
                FitMapping();
                is_calibrating=false; is_calibrated=true;
                printf("[Calib] Done! scale=(%.2f,%.2f)\n",scale_x,scale_y);
                return true;
            }
        }
        return false;
    }
    void MapGaze(float ox,float oy,float& gnx,float& gny){
        gnx=0.5f+(ox-offset_x)*scale_x;
        gny=0.5f+(oy-offset_y)*scale_y;
        gnx=std::max(0.f,std::min(1.f,gnx));
        gny=std::max(0.f,std::min(1.f,gny));
    }
    std::array<float,2> GetCurrentCalibPoint(float iw,float ih){
        if(!is_calibrating||current_point>=9) return {iw*0.5f,ih*0.5f};
        return {CALIB_POINTS[current_point][0]*iw,
                CALIB_POINTS[current_point][1]*ih};
    }
private:
    void FitMapping(){
        float sox=0,soy=0,stx=0,sty=0,sox2=0,soy2=0,soxtx=0,soyty=0;
        for(int i=0;i<9;i++){
            float ox=calib_data[i].ox, oy=calib_data[i].oy;
            float tx=CALIB_POINTS[i][0]-0.5f, ty=CALIB_POINTS[i][1]-0.5f;
            sox+=ox; soy+=oy; stx+=tx; sty+=ty;
            sox2+=ox*ox; soy2+=oy*oy;
            soxtx+=ox*tx; soyty+=oy*ty;
        }
        float n=9.f;
        float dx=n*sox2-sox*sox, dy=n*soy2-soy*soy;
        if(fabsf(dx)>1e-6f) scale_x=(n*soxtx-sox*stx)/dx;
        if(fabsf(dy)>1e-6f) scale_y=(n*soyty-soy*sty)/dy;
        offset_x=(sox-scale_x*stx)/n;
        offset_y=(soy-scale_y*sty)/n;
    }
};

// 全局变量
VISUALIZER* g_visualizer=nullptr;
GazeCalibrator g_calibrator;
std::mutex g_calib_mtx;
std::mutex mtx_image;
std::condition_variable cv_image_ready;
std::atomic<bool> stop_inference(false);
struct ImagePair{ ssne_tensor_t img1,img2; int frame_id; };
std::queue<ImagePair> image_queue;
const int MAX_QUEUE_SIZE=2;
bool g_exit_flag=false;
std::mutex g_mtx;
std::atomic<bool> g_clear_marks(false);
float g_last_sgx=0.5f, g_last_sgy=0.5f;

// 眨眼标记点列表
std::vector<std::array<float,2>> g_blink_marks;
std::mutex g_blink_mtx;
const int MAX_BLINK_MARKS=20;

void keyboard_listener(){
    std::string input;
    printf("Commands: q=quit c=calibrate n=next r=clear marks\n");
    while(true){
        std::cin>>input;
        printf("[KEY] got input: '%s'\n",input.c_str());
        if(input=="q"||input=="Q"){
            std::lock_guard<std::mutex> lock(g_mtx);
            g_exit_flag=true; break;
        } else if(input=="c"||input=="C"){
            std::lock_guard<std::mutex> lock(g_calib_mtx);
            g_calibrator.StartCalibration();
        } else if(input=="n"||input=="N"){
            std::lock_guard<std::mutex> lock(g_calib_mtx);
            if(g_calibrator.is_calibrating)
                g_calibrator.sample_count=GazeCalibrator::SAMPLES_PER_POINT;
        } else if(input=="r"||input=="R"){
            std::lock_guard<std::mutex> lock(g_blink_mtx);
            g_blink_marks.clear();
            g_clear_marks=true;
            printf("标记点已清除\n");
        }
    }
}

bool check_exit_flag(){
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

void inference_thread_func(SCRFDGRAY* detector, PupilDetector* pupil_det,
                           int img_w, int img_h){
    printf("[Thread] Inference started\n");
    FaceDetectionResult* det_result=new FaceDetectionResult;
    const int SW=5;
    std::vector<float> gnx_hist,gny_hist;

    ssne_tensor_t mirrored=create_tensor(640,480,SSNE_Y_8,SSNE_BUF_AI);
    while(!stop_inference){
        ImagePair img_pair; bool has_image=false;
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            cv_image_ready.wait(lock,[]{
                return !image_queue.empty()||stop_inference.load();});
            if(stop_inference&&image_queue.empty()) break;
            if(!image_queue.empty()){
                img_pair=image_queue.front(); image_queue.pop(); has_image=true;
            }
        }
        if(!has_image) continue;

        mirror_tensor(img_pair.img1,mirrored);
        detector->Predict(&mirrored,det_result,0.4f);
        if(det_result->boxes.empty()){
            // 人脸检测失败时也计入眨眼帧
            static int blink_frames_nf=0;
            blink_frames_nf++;
            if(blink_frames_nf>30) blink_frames_nf=0;
            if(g_visualizer){
                std::vector<std::array<float,4>> test_boxes;
                test_boxes.push_back({100.f,100.f,200.f,200.f});
                test_boxes.push_back({300.f,200.f,400.f,300.f});
                g_visualizer->Draw(test_boxes);
            }
            continue;
        }

        auto& face=det_result->boxes[0];
        float fw=face[2]-face[0], fh=face[3]-face[1];
        float lcx,lcy,rcx,rcy;
        if(det_result->landmarks_per_face>=2&&det_result->landmarks.size()>=2){
            lcx=det_result->landmarks[0][0]; lcy=det_result->landmarks[0][1];
            rcx=det_result->landmarks[1][0]; rcy=det_result->landmarks[1][1];
        } else {
            lcx=face[0]+fw*0.30f; lcy=face[1]+fh*0.35f;
            rcx=face[0]+fw*0.70f; rcy=face[1]+fh*0.35f;
        }

        auto lb=GetEyeBox(face,lcx,lcy,img_w,img_h);
        auto rb=GetEyeBox(face,rcx,rcy,img_w,img_h);
        float lx,ly,rx,ry;
        bool lok=pupil_det->Predict(&mirrored,lb,lx,ly);
        bool rok=pupil_det->Predict(&mirrored,rb,rx,ry);

        // 眨眼检测：暗区像素比例
        {
            static bool in_blink=false;
            static int blink_count=0;
            int ex1b=(int)std::max(0.f,lb[0]);
            int ey1b=(int)std::max(0.f,lb[1]);
            int ex2b=(int)std::min((float)img_w,lb[2]);
            int ey2b=(int)std::min((float)img_h,lb[3]);
            int ew2b=ex2b-ex1b, eh2b=ey2b-ey1b;
            if(ew2b>4 && eh2b>4 && pupil_det->img_buf_!=nullptr){
                long sum=0;
                int cnt=ew2b*eh2b;
                for(int y=0;y<eh2b;y++)
                    for(int x=0;x<ew2b;x++)
                        sum+=pupil_det->img_buf_[(ey1b+y)*img_w+(ex1b+x)];
                float mean=(float)sum/cnt;
                float thr=mean*0.6f;
                int dark=0;
                for(int y=0;y<eh2b;y++)
                    for(int x=0;x<ew2b;x++)
                        if(pupil_det->img_buf_[(ey1b+y)*img_w+(ex1b+x)]<thr) dark++;
                float dark_ratio=(float)dark/cnt;
                // EAR打印已关闭
                if(dark_ratio<0.06f && !in_blink){
                    in_blink=true;
                    blink_count++;
                    printf("[Blink] detected! count=%d\n",blink_count);
                    std::lock_guard<std::mutex> lk(g_blink_mtx);
                    if((int)g_blink_marks.size()>=MAX_BLINK_MARKS)
                        g_blink_marks.erase(g_blink_marks.begin());
                    g_blink_marks.push_back({g_last_sgx,g_last_sgy});
                } else if(dark_ratio>0.075f){
                    in_blink=false;
                }
            }
        }

        float ox=0,oy=0; bool has_gaze=false;
        if(lok&&rok){
            ox=((lx-lcx)+(rx-rcx))*0.5f/fw;
            oy=((ly-lcy)+(ry-rcy))*0.5f/fh;
            has_gaze=true;
        } else if(lok){
            ox=(lx-lcx)/fw; oy=(ly-lcy)/fh; has_gaze=true;
        } else if(rok){
            ox=(rx-rcx)/fw; oy=(ry-rcy)/fh; has_gaze=true;
        }
        if(!has_gaze) continue;

        {
            std::lock_guard<std::mutex> lock(g_calib_mtx);
            if(g_calibrator.is_calibrating) g_calibrator.AddSample(ox,oy);
        }

        float gnx,gny;
        {
            std::lock_guard<std::mutex> lock(g_calib_mtx);
            g_calibrator.MapGaze(ox,oy,gnx,gny);
        }

        gnx_hist.push_back(gnx); gny_hist.push_back(gny);
        if((int)gnx_hist.size()>SW){
            gnx_hist.erase(gnx_hist.begin());
            gny_hist.erase(gny_hist.begin());
        }
        float sgx=0,sgy=0;
        for(float v:gnx_hist) sgx+=v;
        for(float v:gny_hist) sgy+=v;
        sgx/=gnx_hist.size(); sgy/=gny_hist.size();
        sgx=std::max(0.f,std::min(1.f,sgx));
        sgy=std::max(0.f,std::min(1.f,sgy));

        g_last_sgx=sgx; g_last_sgy=sgy;
        GazeDirection dir=ClassifyGaze(sgx,sgy);
        // Gaze打印已关闭

        // OSD绘制：用VISUALIZER的Draw接口，只画矩形框
        if(g_visualizer){
            std::vector<std::array<float,4>> draw_boxes;

            // 人脸框
            draw_boxes.push_back(face);

            // 左瞳孔框
            if(lok){
                float ps=8.f;
                draw_boxes.push_back({lx-ps,ly-ps,lx+ps,ly+ps});
            }
            // 右瞳孔框
            if(rok){
                float ps=8.f;
                draw_boxes.push_back({rx-ps,ry-ps,rx+ps,ry+ps});
            }

            // 灰色画布（铺满下半屏）
            draw_boxes.push_back({0.f, 480.f, (float)img_w, (float)(img_h*2)});

            // 注视点光标（在第二路显示区域，y偏移480）
            float gx=sgx*img_w, gy=480.f+sgy*img_h;
            float cs=12.f;
            draw_boxes.push_back({gx-cs,gy-cs,gx+cs,gy+cs});

            // 方向指示块（左上角9宫格）
            float cell=18.f,bx=10.f,by=10.f;
            int dc=(dir==GAZE_LEFT_UP||dir==GAZE_LEFT||dir==GAZE_LEFT_DOWN)?0:
                   (dir==GAZE_RIGHT_UP||dir==GAZE_RIGHT||dir==GAZE_RIGHT_DOWN)?2:1;
            int dr2=(dir==GAZE_LEFT_UP||dir==GAZE_UP||dir==GAZE_RIGHT_UP)?0:
                   (dir==GAZE_LEFT_DOWN||dir==GAZE_DOWN||dir==GAZE_RIGHT_DOWN)?2:1;
            float ddx=bx+dc*cell, ddy=by+dr2*cell;
            draw_boxes.push_back({ddx,ddy,ddx+cell-2,ddy+cell-2});

            // 眨眼标记点
            {
                std::lock_guard<std::mutex> lk(g_blink_mtx);
                for(auto& m : g_blink_marks){
                    float mx=m[0]*img_w, my=m[1]*img_h+480.f;
                    float ms=5.f;
                    draw_boxes.push_back({mx-ms,my-ms,mx+ms,my+ms});
                }
            }
            // 测试框已去除

            // 校准点显示（28x28命中橙色分类）
            {
                std::lock_guard<std::mutex> lk(g_calib_mtx);
                if(g_calibrator.is_calibrating){
                    auto cp=g_calibrator.GetCurrentCalibPoint(img_w,img_h);
                    float cs=14.f;
                    draw_boxes.push_back({cp[0]-cs,cp[1]-cs,cp[0]+cs,cp[1]+cs});
                }
            }
            // 清除标志：先画空一次，强制刷新OSD
            if(g_clear_marks.exchange(false)){
                std::vector<std::array<float,4>> empty;
                g_visualizer->Draw(empty);
            }
            g_visualizer->Draw(draw_boxes);
        }
    }
    release_tensor(mirrored);
    delete det_result;
    printf("[Thread] Inference stopped\n");
}

int main(){
    uint8_t load_flag=0;
    int img_width=640, img_height=480;
    array<int,2> det_shape={640,480};
    string path_det="/app_demo/app_assets/models/face_640x480.m1model";

    if(ssne_initial()){fprintf(stderr,"SSNE init failed\n");return -1;}

    array<int,2> img_shape={img_width,img_height};
    const int dual_display_offset_y=480;

    // 用原始VISUALIZER，和原demo一样
    VISUALIZER visualizer;
    std::array<int,2> dual_shape={img_width, img_height*2};
    visualizer.Initialize(dual_shape);
    g_visualizer=&visualizer;

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    SCRFDGRAY detector;
    int box_len=det_shape[0]*det_shape[1]/512*21;
    detector.Initialize(path_det,&img_shape,&det_shape,false,box_len);
    printf("[INFO] Face model OK\n");

    PupilDetector pupil_det;
    pupil_det.Initialize(img_width,img_height);
    printf("[INFO] Pupil detector OK\n");

    sleep(0.2);

    // 自动启动九点校准
    g_calibrator.StartCalibration();
    printf("[INFO] 自动开始九点校准，请依次注视9个校准点\n");

    std::thread inference_thread(inference_thread_func,
                                  &detector,&pupil_det,img_width,img_height);
    std::thread listener_thread(keyboard_listener);

    uint16_t num_frames=0;
    ssne_tensor_t img_sensor[2];
    ssne_tensor_t output_sensor[2];
    output_sensor[0]=create_tensor(img_width,img_height*2,SSNE_Y_8,SSNE_BUF_AI);
    output_sensor[1]=create_tensor(img_width,img_height*2,SSNE_Y_8,SSNE_BUF_AI);

    processor.GetDualImage(&img_sensor[0],&img_sensor[1]);
    copy_double_tensor_buffer(img_sensor[0],img_sensor[1],output_sensor[0]);
    copy_double_tensor_buffer(img_sensor[0],img_sensor[1],output_sensor[1]);
    set_isp_debug_config(output_sensor[0],output_sensor[1]);
    while(num_frames<2){start_isp_debug_load();num_frames++;}

    ssne_tensor_t mirror_display=create_tensor(img_width,img_height,SSNE_Y_8,SSNE_BUF_AI);
    ssne_tensor_t mirror_display1=create_tensor(img_width,img_height,SSNE_Y_8,SSNE_BUF_AI);
    while(!check_exit_flag()){
        processor.GetDualImage(&img_sensor[0],&img_sensor[1]);
        get_even_or_odd_flag(load_flag);
        mirror_tensor(img_sensor[0],mirror_display);
        mirror_tensor(img_sensor[1],mirror_display1);
        if(load_flag==0)
            copy_double_tensor_buffer(mirror_display,mirror_display1,output_sensor[0]);
        else
            copy_double_tensor_buffer(mirror_display,mirror_display1,output_sensor[1]);
        start_isp_debug_load();
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            if(image_queue.size()<MAX_QUEUE_SIZE){
                ImagePair p;
                p.img1=img_sensor[0]; p.img2=img_sensor[1];
                p.frame_id=num_frames;
                image_queue.push(p); cv_image_ready.notify_one();
            }
        }
        num_frames++;
    }

    if(listener_thread.joinable()) listener_thread.join();
    {
        std::unique_lock<std::mutex> lock(mtx_image);
        stop_inference=true; cv_image_ready.notify_one();
    }
    if(inference_thread.joinable()) inference_thread.join();

    release_tensor(mirror_display);
    release_tensor(mirror_display1);
    pupil_det.Release();
    detector.Release();
    processor.Release();
    visualizer.Release();

    if(ssne_release()){fprintf(stderr,"SSNE release failed\n");return -1;}
    return 0;
}
