/*
 * @Filename: eye_track_model.cpp
 * @Description: Pupil tracking + optional pupil_gap model fallback
 * @Date: 2026-04-26
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <cstdlib>
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

bool FindPupilClassic(const uint8_t* gray, int stride, int w, int h,
                      float& nx, float& ny) {
    if(!gray||stride<w||w<=0||h<=0) return false;
    uint64_t sum=0;
    for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
            sum+=gray[y*stride+x];
    float mean=(float)sum/(w*h);
    float thr=mean*0.55f;
    float cx=0,cy=0; int cnt=0;
    for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
            if(gray[y*stride+x]<thr){cx+=x;cy+=y;cnt++;}
    if(cnt<4) return false;
    nx=(cx/cnt)/(float)w; ny=(cy/cnt)/(float)h;
    return true;
}

class PupilDetector {
public:
    void Initialize(int w, int h, const std::string& model_path,
                    bool enable_model=true) {
        img_w_=w; img_h_=h;
        linux_tensor_=create_tensor(w,h,SSNE_Y_8,SSNE_BUF_LINUX);
        buf_size_=w*h;
        img_buf_=new uint8_t[buf_size_];
        printf("[PupilDetector] classic mode ready\n");
        if(enable_model) TryInitializeModel(model_path);
    }
    bool PrepareFrame(ssne_tensor_t* img) {
        if(!img) return false;
        if(copy_tensor_buffer(*img,linux_tensor_)!=SSNE_ERRCODE_NO_ERROR) return false;
        if(save_tensor_buffer_ptr(linux_tensor_,img_buf_,buf_size_)!=SSNE_ERRCODE_NO_ERROR) return false;
        return true;
    }
    bool Detect(const std::array<float,4>& eye_box,
                float& out_x, float& out_y) {
        int ex1,ey1,ex2,ey2;
        if(!GetClippedEyeRegion(eye_box,ex1,ey1,ex2,ey2)) return false;
        const int roi_w=ex2-ex1, roi_h=ey2-ey1;
        if(model_ready_&&DetectWithModel(ex1,ey1,roi_w,roi_h,out_x,out_y))
            return true;
        return DetectClassic(ex1,ey1,roi_w,roi_h,out_x,out_y);
    }
    bool ModelEnabled() const { return model_ready_; }
    const char* ModeName() const {
        return model_ready_ ? "model+classic fallback" : "classic";
    }
    bool GetDarkRatio(const std::array<float,4>& eye_box,
                      float& dark_ratio) const {
        int ex1,ey1,ex2,ey2;
        if(!GetClippedEyeRegion(eye_box,ex1,ey1,ex2,ey2)) return false;
        const int roi_w=ex2-ex1, roi_h=ey2-ey1;
        uint64_t sum=0;
        for(int y=ey1;y<ey2;y++)
            for(int x=ex1;x<ex2;x++)
                sum+=img_buf_[y*img_w_+x];
        const int pixels=roi_w*roi_h;
        const float mean=static_cast<float>(sum)/pixels;
        const float threshold=mean*0.6f;
        int dark=0;
        for(int y=ey1;y<ey2;y++)
            for(int x=ex1;x<ex2;x++)
                if(img_buf_[y*img_w_+x]<threshold) dark++;
        dark_ratio=static_cast<float>(dark)/pixels;
        return true;
    }
    void Release() {
        if(output_ready_) release_tensor(outputs_[0]);
        if(model_input_ready_) release_tensor(model_input_);
        release_tensor(linux_tensor_);
        delete[] img_buf_;
        img_buf_=nullptr;
    }
private:
    static const int MODEL_INPUT_SIZE=224;

    bool DetectClassic(int ex1,int ey1,int roi_w,int roi_h,
                       float& out_x,float& out_y) const {
        float nx,ny;
        if(!FindPupilClassic(img_buf_+ey1*img_w_+ex1,img_w_,
                             roi_w,roi_h,nx,ny)) return false;
        out_x=ex1+nx*roi_w;
        out_y=ey1+ny*roi_h;
        return true;
    }

    void TryInitializeModel(const std::string& model_path) {
        if(model_path.empty()){
            printf("[PupilDetector] pupil model path empty, use classic\n");
            return;
        }
        std::ifstream fin(model_path.c_str(),std::ios::binary);
        if(!fin.good()){
            printf("[PupilDetector] optional model not found: %s, use classic\n",
                   model_path.c_str());
            return;
        }
        char* path_char=const_cast<char*>(model_path.c_str());
        model_id_=ssne_loadmodel(path_char,SSNE_STATIC_ALLOC);

        int input_num=ssne_get_model_input_num(model_id_);
        if(input_num!=1){
            printf("[PupilDetector] model load check failed, input_num=%d, use classic\n",
                   input_num);
            return;
        }

        int dtype=SSNE_UINT8;
        if(ssne_get_model_input_dtype(model_id_,&dtype)!=SSNE_ERRCODE_NO_ERROR){
            printf("[PupilDetector] failed to query model dtype, use classic\n");
            return;
        }
        if(dtype!=SSNE_UINT8&&dtype!=SSNE_FLOAT32){
            printf("[PupilDetector] unsupported model dtype=%d, use classic\n",dtype);
            return;
        }

        model_input_dtype_=dtype;
        if(model_input_dtype_==SSNE_FLOAT32){
            model_input_=create_tensor(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE,
                                       SSNE_BYTES,SSNE_BUF_AI);
            set_data_type(model_input_,SSNE_FLOAT32);
            model_input_f32_.resize(MODEL_INPUT_SIZE*MODEL_INPUT_SIZE);
        } else {
            model_input_=create_tensor(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE,
                                       SSNE_Y_8,SSNE_BUF_AI);
            model_input_u8_.resize(MODEL_INPUT_SIZE*MODEL_INPUT_SIZE);
        }
        model_input_ready_=true;
        model_ready_=true;
        printf("[PupilDetector] optional model ready: %s dtype=%d\n",
               model_path.c_str(),model_input_dtype_);
    }

    bool DetectWithModel(int ex1,int ey1,int roi_w,int roi_h,
                         float& out_x,float& out_y) {
        if(!FillModelInput(ex1,ey1,roi_w,roi_h)){
            DisableModelOnce("[PupilDetector] pupil model input failed, switch to classic\n");
            return false;
        }

        if(ssne_inference(model_id_,1,&model_input_)!=SSNE_ERRCODE_NO_ERROR){
            DisableModelOnce("[PupilDetector] pupil model inference failed, switch to classic\n");
            return false;
        }
        if(ssne_getoutput(model_id_,1,outputs_)!=SSNE_ERRCODE_NO_ERROR){
            DisableModelOnce("[PupilDetector] pupil model output failed, switch to classic\n");
            return false;
        }
        output_ready_=true;

        if(get_data_type(outputs_[0])!=SSNE_FLOAT32||
           get_mem_size(outputs_[0])<2*sizeof(float)){
            DisableModelOnce("[PupilDetector] pupil model output type invalid, switch to classic\n");
            return false;
        }
        float* out=static_cast<float*>(get_data(outputs_[0]));
        if(!out){
            DisableModelOnce("[PupilDetector] pupil model output is null, switch to classic\n");
            return false;
        }
        float nx=out[0], ny=out[1];
        if(!std::isfinite(nx)||!std::isfinite(ny)||
           nx<-0.25f||nx>1.25f||ny<-0.25f||ny>1.25f){
            DisableModelOnce("[PupilDetector] pupil model output invalid, switch to classic\n");
            return false;
        }
        nx=std::max(0.f,std::min(1.f,nx));
        ny=std::max(0.f,std::min(1.f,ny));
        out_x=ex1+nx*roi_w;
        out_y=ey1+ny*roi_h;
        return true;
    }

    bool FillModelInput(int ex1,int ey1,int roi_w,int roi_h) {
        const int pixels=MODEL_INPUT_SIZE*MODEL_INPUT_SIZE;
        if(model_input_dtype_==SSNE_FLOAT32){
            if((int)model_input_f32_.size()!=pixels) return false;
        } else {
            if((int)model_input_u8_.size()!=pixels) return false;
        }

        for(int y=0;y<MODEL_INPUT_SIZE;y++){
            float sy=ey1+(y+0.5f)*roi_h/MODEL_INPUT_SIZE-0.5f;
            sy=std::max(static_cast<float>(ey1),
                        std::min(static_cast<float>(ey1+roi_h-1),sy));
            int y0=static_cast<int>(std::floor(sy));
            int y1=std::min(ey1+roi_h-1,y0+1);
            float fy=sy-y0;
            for(int x=0;x<MODEL_INPUT_SIZE;x++){
                float sx=ex1+(x+0.5f)*roi_w/MODEL_INPUT_SIZE-0.5f;
                sx=std::max(static_cast<float>(ex1),
                            std::min(static_cast<float>(ex1+roi_w-1),sx));
                int x0=static_cast<int>(std::floor(sx));
                int x1=std::min(ex1+roi_w-1,x0+1);
                float fx=sx-x0;
                float p00=img_buf_[y0*img_w_+x0];
                float p01=img_buf_[y0*img_w_+x1];
                float p10=img_buf_[y1*img_w_+x0];
                float p11=img_buf_[y1*img_w_+x1];
                float top=p00+(p01-p00)*fx;
                float bottom=p10+(p11-p10)*fx;
                float pix=top+(bottom-top)*fy;
                int idx=y*MODEL_INPUT_SIZE+x;
                if(model_input_dtype_==SSNE_FLOAT32){
                    model_input_f32_[idx]=pix/255.f;
                } else {
                    int v=static_cast<int>(pix+0.5f);
                    model_input_u8_[idx]=static_cast<uint8_t>(
                        std::max(0,std::min(255,v)));
                }
            }
        }

        if(model_input_dtype_==SSNE_FLOAT32){
            return load_tensor_buffer_ptr(
                model_input_,model_input_f32_.data(),
                pixels*static_cast<int>(sizeof(float)))==SSNE_ERRCODE_NO_ERROR;
        }
        return load_tensor_buffer_ptr(
            model_input_,model_input_u8_.data(),pixels)==SSNE_ERRCODE_NO_ERROR;
    }

    void DisableModelOnce(const char* msg) {
        if(model_ready_) printf("%s",msg);
        model_ready_=false;
    }

    bool GetClippedEyeRegion(const std::array<float,4>& eye_box,
                             int& x1,int& y1,int& x2,int& y2) const {
        x1=std::max(0,std::min(img_w_,static_cast<int>(std::floor(eye_box[0]))));
        y1=std::max(0,std::min(img_h_,static_cast<int>(std::floor(eye_box[1]))));
        x2=std::max(0,std::min(img_w_,static_cast<int>(std::ceil(eye_box[2]))));
        y2=std::max(0,std::min(img_h_,static_cast<int>(std::ceil(eye_box[3]))));
        return x2-x1>=4&&y2-y1>=4;
    }
    uint8_t* img_buf_=nullptr;
    int img_w_=0, img_h_=0;
    ssne_tensor_t linux_tensor_;
    int buf_size_=0;
    bool model_ready_=false;
    bool model_input_ready_=false;
    bool output_ready_=false;
    int model_input_dtype_=SSNE_UINT8;
    uint16_t model_id_=0;
    ssne_tensor_t model_input_;
    ssne_tensor_t outputs_[1];
    std::vector<uint8_t> model_input_u8_;
    std::vector<float> model_input_f32_;
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

void AppendSevenSegmentDigit(std::vector<std::array<float,4>>& boxes,
                             int digit,float x,float y,float scale){
    static const uint8_t masks[10]={
        0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f
    };
    digit=std::max(0,std::min(9,digit));
    const uint8_t mask=masks[digit];
    const float w=8.f*scale,h=14.f*scale,t=2.f*scale,mid=y+h*0.5f;
    if(mask&0x01) boxes.push_back({x+t,y,x+w-t,y+t});
    if(mask&0x02) boxes.push_back({x+w-t,y+t,x+w,mid});
    if(mask&0x04) boxes.push_back({x+w-t,mid,x+w,y+h-t});
    if(mask&0x08) boxes.push_back({x+t,y+h-t,x+w-t,y+h});
    if(mask&0x10) boxes.push_back({x,mid,x+t,y+h-t});
    if(mask&0x20) boxes.push_back({x,y+t,x+t,mid});
    if(mask&0x40) boxes.push_back({x+t,mid-t*0.5f,x+w-t,mid+t*0.5f});
}

void AppendCalibrationProgress(std::vector<std::array<float,4>>& boxes,
                               int point,int valid_samples,int img_w){
    point=std::max(1,std::min(9,point));
    valid_samples=std::max(0,std::min(99,valid_samples));
    const float scale=1.4f;
    const float y=448.f;
    const float x=static_cast<float>(img_w)-62.f;
    AppendSevenSegmentDigit(boxes,point,x,y,scale);
    const float colon_x=x+14.f;
    boxes.push_back({colon_x,y+5.f,colon_x+2.f,y+7.f});
    boxes.push_back({colon_x,y+13.f,colon_x+2.f,y+15.f});
    AppendSevenSegmentDigit(boxes,valid_samples/10,x+21.f,y,scale);
    AppendSevenSegmentDigit(boxes,valid_samples%10,x+35.f,y,scale);
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
const int MAX_QUEUE_SIZE=2;
const int FRAME_POOL_SIZE=MAX_QUEUE_SIZE+1;
using PerfClock=std::chrono::steady_clock;
struct InferenceFrame {
    int pool_index;
    uint64_t frame_id;
    PerfClock::time_point captured_at;
};
std::queue<InferenceFrame> image_queue;
std::atomic<int> g_queue_depth(0);
std::atomic<uint64_t> g_perf_captured(0);
std::atomic<uint64_t> g_perf_enqueued(0);
std::atomic<uint64_t> g_perf_dropped(0);
std::atomic<uint64_t> g_perf_inferred(0);
std::atomic<uint64_t> g_perf_latency_us(0);
std::atomic<uint64_t> g_perf_latency_max_us(0);
bool g_exit_flag=false;
std::mutex g_mtx;
std::atomic<bool> g_clear_marks(false);
float g_last_sgx=0.5f, g_last_sgy=0.5f;

// 眨眼标记点列表
std::vector<std::array<float,2>> g_blink_marks;
std::mutex g_blink_mtx;
const int MAX_BLINK_MARKS=20;

int TryAcquireFrameSlot(std::atomic<unsigned int>& free_mask){
    unsigned int mask=free_mask.load(std::memory_order_relaxed);
    while(mask!=0){
        int slot=0;
        while((mask&(1u<<slot))==0u) slot++;
        const unsigned int updated=mask&~(1u<<slot);
        if(free_mask.compare_exchange_weak(mask,updated,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)){
            return slot;
        }
    }
    return -1;
}

void ReturnFrameSlot(std::atomic<unsigned int>& free_mask,int slot){
    free_mask.fetch_or(1u<<slot,std::memory_order_release);
}

void UpdateMaxLatency(uint64_t latency_us){
    uint64_t current=g_perf_latency_max_us.load(std::memory_order_relaxed);
    while(current<latency_us &&
          !g_perf_latency_max_us.compare_exchange_weak(
              current,latency_us,std::memory_order_relaxed,
              std::memory_order_relaxed)){}
}

class InferenceFrameLease {
public:
    InferenceFrameLease(std::atomic<unsigned int>& free_mask,int slot,
                        PerfClock::time_point captured_at)
        :free_mask_(free_mask),slot_(slot),captured_at_(captured_at){}
    ~InferenceFrameLease(){
        const uint64_t latency_us=static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                PerfClock::now()-captured_at_).count());
        g_perf_latency_us.fetch_add(latency_us,std::memory_order_relaxed);
        UpdateMaxLatency(latency_us);
        g_perf_inferred.fetch_add(1,std::memory_order_release);
        ReturnFrameSlot(free_mask_,slot_);
    }
private:
    std::atomic<unsigned int>& free_mask_;
    int slot_;
    PerfClock::time_point captured_at_;
};

bool TryEnqueueInferenceFrame(ssne_tensor_t& source,
                              ssne_tensor_t* frame_pool,
                              std::atomic<unsigned int>& free_mask,
                              uint64_t frame_id,
                              PerfClock::time_point captured_at){
    if(g_queue_depth.load(std::memory_order_acquire)>=MAX_QUEUE_SIZE)
        return false;

    const int slot=TryAcquireFrameSlot(free_mask);
    if(slot<0) return false;

    if(copy_tensor_buffer(source,frame_pool[slot])!=SSNE_ERRCODE_NO_ERROR){
        ReturnFrameSlot(free_mask,slot);
        return false;
    }

    std::unique_lock<std::mutex> lock(mtx_image,std::try_to_lock);
    if(!lock.owns_lock()||image_queue.size()>=MAX_QUEUE_SIZE){
        ReturnFrameSlot(free_mask,slot);
        return false;
    }

    InferenceFrame frame;
    frame.pool_index=slot;
    frame.frame_id=frame_id;
    frame.captured_at=captured_at;
    image_queue.push(frame);
    g_queue_depth.store(static_cast<int>(image_queue.size()),
                        std::memory_order_release);
    lock.unlock();
    cv_image_ready.notify_one();
    return true;
}

void MaybeLogPerformance(){
    const PerfClock::time_point now=PerfClock::now();
    static PerfClock::time_point last_log=now;
    static uint64_t last_captured=0;
    static uint64_t last_enqueued=0;
    static uint64_t last_dropped=0;
    static uint64_t last_inferred=0;
    static uint64_t last_latency_us=0;
    const double elapsed=std::chrono::duration<double>(now-last_log).count();
    if(elapsed<1.0) return;

    const uint64_t captured=g_perf_captured.load(std::memory_order_relaxed);
    const uint64_t enqueued=g_perf_enqueued.load(std::memory_order_relaxed);
    const uint64_t dropped=g_perf_dropped.load(std::memory_order_relaxed);
    const uint64_t inferred=g_perf_inferred.load(std::memory_order_acquire);
    const uint64_t latency_us=g_perf_latency_us.load(std::memory_order_relaxed);
    const uint64_t inferred_delta=inferred-last_inferred;
    const uint64_t latency_delta=latency_us-last_latency_us;
    const uint64_t latency_max_us=
        g_perf_latency_max_us.exchange(0,std::memory_order_relaxed);
    const double latency_avg_ms=inferred_delta==0?0.0:
        static_cast<double>(latency_delta)/inferred_delta/1000.0;

    printf("[PERF] capture=%llu enqueue=%llu drop=%llu "
           "infer_fps=%.1f queue=%d latency_ms_avg=%.1f latency_ms_max=%.1f\n",
           static_cast<unsigned long long>(captured-last_captured),
           static_cast<unsigned long long>(enqueued-last_enqueued),
           static_cast<unsigned long long>(dropped-last_dropped),
           inferred_delta/elapsed,
           g_queue_depth.load(std::memory_order_acquire),
           latency_avg_ms,latency_max_us/1000.0);

    last_captured=captured;
    last_enqueued=enqueued;
    last_dropped=dropped;
    last_inferred=inferred;
    last_latency_us=latency_us;
    last_log=now;
}

void UpdateBlinkDetection(bool valid,float dark_ratio){
    static const int CLOSE_DEBOUNCE_FRAMES=2;
    static const int OPEN_DEBOUNCE_FRAMES=2;
    static const int MIN_BLINK_MS=80;
    static const int MAX_BLINK_MS=800;
    static int closed_streak=0;
    static int open_streak=0;
    static int blink_count=0;
    static int rejected_short=0;
    static int rejected_long=0;
    static bool blink_active=false;
    static bool suppress_until_open=false;
    static PerfClock::time_point blink_start;
    static PerfClock::time_point open_start;

    if(!valid){
        closed_streak=0;
        open_streak=0;
        if(!suppress_until_open) blink_active=false;
        return;
    }

    const bool is_closed=dark_ratio<0.06f;
    const bool is_open=dark_ratio>0.075f;
    const PerfClock::time_point now=PerfClock::now();

    if(suppress_until_open){
        if(is_open){
            if(++open_streak>=OPEN_DEBOUNCE_FRAMES){
                suppress_until_open=false;
                open_streak=0;
            }
        } else {
            open_streak=0;
        }
        return;
    }

    if(!blink_active){
        if(is_closed){
            if(closed_streak==0) blink_start=now;
            if(++closed_streak>=CLOSE_DEBOUNCE_FRAMES){
                blink_active=true;
                open_streak=0;
            }
        } else {
            closed_streak=0;
        }
        return;
    }

    if(is_open){
        if(open_streak==0) open_start=now;
        if(++open_streak<OPEN_DEBOUNCE_FRAMES) return;

        const int duration_ms=static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                open_start-blink_start).count());
        if(duration_ms>=MIN_BLINK_MS&&duration_ms<=MAX_BLINK_MS){
            blink_count++;
            {
                std::lock_guard<std::mutex> lk(g_blink_mtx);
                if((int)g_blink_marks.size()>=MAX_BLINK_MARKS)
                    g_blink_marks.erase(g_blink_marks.begin());
                g_blink_marks.push_back({g_last_sgx,g_last_sgy});
            }
            printf("[Blink] count=%d duration_ms=%d rejected_short=%d "
                   "rejected_long=%d\n",
                   blink_count,duration_ms,rejected_short,rejected_long);
        } else if(duration_ms<MIN_BLINK_MS){
            rejected_short++;
            printf("[Blink] rejected_short=%d duration_ms=%d count=%d\n",
                   rejected_short,duration_ms,blink_count);
        } else {
            rejected_long++;
            printf("[Blink] rejected_long=%d duration_ms=%d count=%d\n",
                   rejected_long,duration_ms,blink_count);
        }
        blink_active=false;
        closed_streak=0;
        open_streak=0;
        return;
    }

    open_streak=0;
    const int duration_ms=static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now-blink_start).count());
    if(duration_ms>MAX_BLINK_MS){
        rejected_long++;
        printf("[Blink] rejected_long=%d duration_ms=%d count=%d\n",
               rejected_long,duration_ms,blink_count);
        blink_active=false;
        closed_streak=0;
        suppress_until_open=true;
    }
}

void keyboard_listener(){
    std::string input;
    printf("Commands: q=quit c=calibrate n=status r=clear marks\n");
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
            if(g_calibrator.is_calibrating){
                printf("[Calib] point=%d/9 valid=%d/%d\n",
                       g_calibrator.current_point+1,
                       g_calibrator.sample_count,
                       GazeCalibrator::SAMPLES_PER_POINT);
            } else {
                printf("[Calib] active=0 calibrated=%d\n",
                       g_calibrator.is_calibrated?1:0);
            }
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
                           ssne_tensor_t* frame_pool,
                           std::atomic<unsigned int>* free_frame_mask,
                           int img_w, int img_h){
    printf("[Thread] Inference started\n");
    FaceDetectionResult* det_result=new FaceDetectionResult;
    const int SW=5;
    std::vector<float> gnx_hist,gny_hist;

    ssne_tensor_t mirrored=create_tensor(640,480,SSNE_Y_8,SSNE_BUF_AI);
    while(true){
        InferenceFrame frame; bool has_image=false;
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            cv_image_ready.wait(lock,[]{
                return !image_queue.empty()||stop_inference.load();});
            if(stop_inference&&image_queue.empty()) break;
            if(!image_queue.empty()){
                frame=image_queue.front(); image_queue.pop(); has_image=true;
                g_queue_depth.store(static_cast<int>(image_queue.size()),
                                    std::memory_order_release);
            }
        }
        if(!has_image) continue;
        InferenceFrameLease frame_lease(*free_frame_mask,frame.pool_index,
                                        frame.captured_at);

        mirror_tensor(frame_pool[frame.pool_index],mirrored);
        detector->Predict(&mirrored,det_result,0.4f);
        if(det_result->boxes.empty()){
            UpdateBlinkDetection(false,0.f);
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
        const bool frame_ready=pupil_det->PrepareFrame(&mirrored);
        const bool lok=frame_ready&&pupil_det->Detect(lb,lx,ly);
        const bool rok=frame_ready&&pupil_det->Detect(rb,rx,ry);

        // 眨眼检测：暗区像素比例
        float dark_ratio=0.f;
        const bool dark_ratio_valid=
            frame_ready&&pupil_det->GetDarkRatio(lb,dark_ratio);
        UpdateBlinkDetection(dark_ratio_valid,dark_ratio);

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
            if(g_calibrator.is_calibrating&&lok&&rok)
                g_calibrator.AddSample(ox,oy);
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
                    AppendCalibrationProgress(
                        draw_boxes,g_calibrator.current_point+1,
                        g_calibrator.sample_count,img_w);
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
    string path_pupil="/app_demo/app_assets/models/pupil_gap.m1model";
    const char* pupil_model_env=std::getenv("PUPIL_GAP_MODEL");
    if(pupil_model_env&&pupil_model_env[0]) path_pupil=pupil_model_env;
    bool enable_pupil_model=true;
    const char* pupil_mode_env=std::getenv("PUPIL_DETECT_MODE");
    if(pupil_mode_env&&std::string(pupil_mode_env)=="classic")
        enable_pupil_model=false;

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
    pupil_det.Initialize(img_width,img_height,path_pupil,enable_pupil_model);
    printf("[INFO] Pupil detector OK (%s)\n",pupil_det.ModeName());

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // 自动启动九点校准
    g_calibrator.StartCalibration();
    printf("[INFO] 自动开始九点校准，请依次注视9个校准点\n");

    std::array<ssne_tensor_t,FRAME_POOL_SIZE> inference_frame_pool;
    for(int i=0;i<FRAME_POOL_SIZE;i++){
        inference_frame_pool[i]=create_tensor(
            img_width,img_height,SSNE_Y_8,SSNE_BUF_AI);
    }
    std::atomic<unsigned int> free_frame_mask(
        (1u<<FRAME_POOL_SIZE)-1u);

    std::thread inference_thread(inference_thread_func,
                                  &detector,&pupil_det,
                                  inference_frame_pool.data(),&free_frame_mask,
                                  img_width,img_height);
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
    uint64_t inference_frame_id=0;
    while(!check_exit_flag()){
        processor.GetDualImage(&img_sensor[0],&img_sensor[1]);
        const PerfClock::time_point captured_at=PerfClock::now();
        g_perf_captured.fetch_add(1,std::memory_order_relaxed);
        get_even_or_odd_flag(load_flag);
        mirror_tensor(img_sensor[0],mirror_display);
        mirror_tensor(img_sensor[1],mirror_display1);
        if(load_flag==0)
            copy_double_tensor_buffer(mirror_display,mirror_display1,output_sensor[0]);
        else
            copy_double_tensor_buffer(mirror_display,mirror_display1,output_sensor[1]);
        start_isp_debug_load();
        if(TryEnqueueInferenceFrame(img_sensor[0],inference_frame_pool.data(),
                                    free_frame_mask,inference_frame_id,
                                    captured_at))
            g_perf_enqueued.fetch_add(1,std::memory_order_relaxed);
        else
            g_perf_dropped.fetch_add(1,std::memory_order_relaxed);
        inference_frame_id++;
        MaybeLogPerformance();
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
    release_tensor(output_sensor[0]);
    release_tensor(output_sensor[1]);
    for(int i=0;i<FRAME_POOL_SIZE;i++)
        release_tensor(inference_frame_pool[i]);
    pupil_det.Release();
    detector.Release();
    processor.Release();
    visualizer.Release();

    if(ssne_release()){fprintf(stderr,"SSNE release failed\n");return -1;}
    return 0;
}
