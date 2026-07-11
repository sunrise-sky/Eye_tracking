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

struct PupilObservation {
    bool valid;
    float x;
    float y;
    float confidence;
    float blink_dark_ratio;
    bool blink_valid;
    bool used_model;

    PupilObservation()
        :valid(false),x(0.f),y(0.f),confidence(0.f),
         blink_dark_ratio(0.f),blink_valid(false),used_model(false){}
};

float Clamp01(float value) {
    return std::max(0.f,std::min(1.f,value));
}

class PupilDetector {
public:
    void Initialize(int w, int h, const std::string& model_path,
                    bool enable_model=true, bool prefer_model=false) {
        img_w_=w; img_h_=h;
        prefer_model_=prefer_model;
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
    PupilObservation Detect(const std::array<float,4>& eye_box,
                            const PupilObservation* previous,
                            bool allow_model_fallback) {
        int ex1,ey1,ex2,ey2;
        if(!GetClippedEyeRegion(eye_box,ex1,ey1,ex2,ey2))
            return PupilObservation();
        const int roi_w=ex2-ex1, roi_h=ey2-ey1;

        if(prefer_model_&&allow_model_fallback&&model_ready_){
            PupilObservation model=DetectWithModel(
                ex1,ey1,roi_w,roi_h,previous);
            if(model.valid){
                model.blink_dark_ratio=ComputeBlinkDarkRatio(
                    ex1,ey1,roi_w,roi_h);
                model.blink_valid=true;
                return model;
            }
        }

        PupilObservation classic=DetectClassic(
            ex1,ey1,roi_w,roi_h,previous);
        if(allow_model_fallback&&model_ready_&&
           (!classic.valid||classic.confidence<MODEL_FALLBACK_CONFIDENCE)){
            PupilObservation model=DetectWithModel(
                ex1,ey1,roi_w,roi_h,previous);
            if(model.valid){
                model.blink_dark_ratio=classic.blink_dark_ratio;
                model.blink_valid=classic.blink_valid;
                return model;
            }
        }
        return classic;
    }
    bool ModelEnabled() const { return model_ready_; }
    bool PreferModel() const { return prefer_model_; }
    const char* ModeName() const {
        if(!model_ready_) return "classic";
        return prefer_model_ ? "model primary + classic fallback" :
                               "classic + model recovery";
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
    static constexpr float MODEL_FALLBACK_CONFIDENCE=0.45f;
    static const int HISTOGRAM_BINS=64;

    PupilObservation DetectClassic(int ex1,int ey1,int roi_w,int roi_h,
                                   const PupilObservation* previous) const {
        PupilObservation result;
        int histogram[HISTOGRAM_BINS]={0};
        uint64_t intensity_sum=0;
        const int roi_pixels=roi_w*roi_h;

        for(int y=ey1;y<ey1+roi_h;y++){
            const uint8_t* row=img_buf_+y*img_w_;
            for(int x=ex1;x<ex1+roi_w;x++){
                const uint8_t pixel=row[x];
                histogram[pixel>>2]++;
                intensity_sum+=pixel;
            }
        }
        if(roi_pixels<=0) return result;

        const float mean=static_cast<float>(intensity_sum)/roi_pixels;
        const int blink_bin=std::max(0,std::min(
            HISTOGRAM_BINS-1,static_cast<int>(mean*0.6f)>>2));
        int blink_dark=0;
        for(int i=0;i<=blink_bin;i++) blink_dark+=histogram[i];
        result.blink_dark_ratio=static_cast<float>(blink_dark)/roi_pixels;
        result.blink_valid=true;

        const int percentile_target=std::max(4,roi_pixels*18/100);
        int cumulative=0;
        int percentile_bin=0;
        for(;percentile_bin<HISTOGRAM_BINS;percentile_bin++){
            cumulative+=histogram[percentile_bin];
            if(cumulative>=percentile_target) break;
        }
        const int percentile_threshold=std::min(
            255,percentile_bin*4+3);
        const int pupil_threshold=std::max(2,std::min(
            percentile_threshold,static_cast<int>(mean*0.78f)));

        int sx1=ex1,sy1=ey1,sx2=ex1+roi_w,sy2=ey1+roi_h;
        if(previous&&previous->valid&&previous->confidence>=0.40f&&
           previous->x>=ex1&&previous->x<ex1+roi_w&&
           previous->y>=ey1&&previous->y<ey1+roi_h){
            const int half_w=std::max(6,static_cast<int>(roi_w*0.38f));
            const int half_h=std::max(4,static_cast<int>(roi_h*0.42f));
            const int px=static_cast<int>(previous->x+0.5f);
            const int py=static_cast<int>(previous->y+0.5f);
            sx1=std::max(ex1,px-half_w);
            sy1=std::max(ey1,py-half_h);
            sx2=std::min(ex1+roi_w,px+half_w+1);
            sy2=std::min(ey1+roi_h,py+half_h+1);
        }

        double weight_sum=0.0;
        double weighted_x=0.0,weighted_y=0.0;
        double weighted_x2=0.0,weighted_y2=0.0;
        uint64_t dark_intensity_sum=0;
        int dark_count=0;
        for(int y=sy1;y<sy2;y++){
            const uint8_t* row=img_buf_+y*img_w_;
            for(int x=sx1;x<sx2;x++){
                const int pixel=row[x];
                if(pixel>pupil_threshold) continue;
                const double weight=pupil_threshold-pixel+1;
                weight_sum+=weight;
                weighted_x+=weight*x;
                weighted_y+=weight*y;
                weighted_x2+=weight*x*x;
                weighted_y2+=weight*y*y;
                dark_intensity_sum+=pixel;
                dark_count++;
            }
        }
        const int search_pixels=std::max(1,(sx2-sx1)*(sy2-sy1));
        if(dark_count<4||weight_sum<=0.0) return result;

        const float cx=static_cast<float>(weighted_x/weight_sum);
        const float cy=static_cast<float>(weighted_y/weight_sum);
        const float variance_x=std::max(0.f,static_cast<float>(
            weighted_x2/weight_sum-cx*cx));
        const float variance_y=std::max(0.f,static_cast<float>(
            weighted_y2/weight_sum-cy*cy));
        const float spread=std::sqrt(variance_x)/std::max(1,roi_w)+
                           std::sqrt(variance_y)/std::max(1,roi_h);
        const float compactness=Clamp01((0.34f-spread)/0.25f);
        const float dark_mean=static_cast<float>(dark_intensity_sum)/dark_count;
        const float contrast=Clamp01((mean-dark_mean)/std::max(1.f,mean)*2.5f);
        const float dark_ratio=static_cast<float>(dark_count)/search_pixels;
        const float ratio_score=dark_ratio<=0.12f ?
            Clamp01(dark_ratio/0.12f) : Clamp01((0.48f-dark_ratio)/0.36f);

        float temporal_score=0.65f;
        if(previous&&previous->valid){
            const float dx=cx-previous->x,dy=cy-previous->y;
            const float distance=std::sqrt(dx*dx+dy*dy);
            const float radius=std::max(4.f,roi_w*0.28f);
            const float normalized=distance/radius;
            temporal_score=1.f/(1.f+normalized*normalized);
        }
        const float edge_x=std::min(cx-ex1,ex1+roi_w-1-cx);
        const float edge_y=std::min(cy-ey1,ey1+roi_h-1-cy);
        const float edge_score=Clamp01(std::min(
            edge_x/std::max(1.f,roi_w*0.08f),
            edge_y/std::max(1.f,roi_h*0.08f)));

        result.x=cx;
        result.y=cy;
        result.confidence=Clamp01(0.35f*contrast+0.20f*ratio_score+
                                  0.20f*compactness+0.15f*temporal_score+
                                  0.10f*edge_score);
        result.valid=dark_ratio<0.48f&&result.confidence>=0.28f;
        return result;
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

    PupilObservation DetectWithModel(int ex1,int ey1,int roi_w,int roi_h,
                                     const PupilObservation* previous) {
        PupilObservation result;
        if(!FillModelInput(ex1,ey1,roi_w,roi_h)){
            DisableModelOnce("[PupilDetector] pupil model input failed, switch to classic\n");
            return result;
        }

        if(ssne_inference(model_id_,1,&model_input_)!=SSNE_ERRCODE_NO_ERROR){
            DisableModelOnce("[PupilDetector] pupil model inference failed, switch to classic\n");
            return result;
        }
        if(ssne_getoutput(model_id_,1,outputs_)!=SSNE_ERRCODE_NO_ERROR){
            DisableModelOnce("[PupilDetector] pupil model output failed, switch to classic\n");
            return result;
        }
        output_ready_=true;

        if(get_data_type(outputs_[0])!=SSNE_FLOAT32||
           get_mem_size(outputs_[0])<2*sizeof(float)){
            DisableModelOnce("[PupilDetector] pupil model output type invalid, switch to classic\n");
            return result;
        }
        float* out=static_cast<float*>(get_data(outputs_[0]));
        if(!out){
            DisableModelOnce("[PupilDetector] pupil model output is null, switch to classic\n");
            return result;
        }
        float nx=out[0], ny=out[1];
        if(!std::isfinite(nx)||!std::isfinite(ny)||
           nx<-0.25f||nx>1.25f||ny<-0.25f||ny>1.25f){
            DisableModelOnce("[PupilDetector] pupil model output invalid, switch to classic\n");
            return result;
        }
        nx=std::max(0.f,std::min(1.f,nx));
        ny=std::max(0.f,std::min(1.f,ny));
        result.x=ex1+nx*roi_w;
        result.y=ey1+ny*roi_h;
        result.confidence=0.78f;
        if(previous&&previous->valid){
            const float dx=result.x-previous->x;
            const float dy=result.y-previous->y;
            const float normalized=std::sqrt(dx*dx+dy*dy)/
                                   std::max(4.f,roi_w*0.35f);
            result.confidence*=1.f/(1.f+normalized*normalized);
        }
        result.valid=result.confidence>=0.35f;
        result.used_model=true;
        return result;
    }

    float ComputeBlinkDarkRatio(int ex1,int ey1,int roi_w,int roi_h) const {
        if(roi_w<=0||roi_h<=0) return 0.f;
        uint64_t sum=0;
        for(int y=ey1;y<ey1+roi_h;y++){
            const uint8_t* row=img_buf_+y*img_w_;
            for(int x=ex1;x<ex1+roi_w;x++) sum+=row[x];
        }
        const int pixels=roi_w*roi_h;
        const float threshold=static_cast<float>(sum)/pixels*0.6f;
        int dark=0;
        for(int y=ey1;y<ey1+roi_h;y++){
            const uint8_t* row=img_buf_+y*img_w_;
            for(int x=ex1;x<ex1+roi_w;x++)
                if(row[x]<threshold) dark++;
        }
        return static_cast<float>(dark)/pixels;
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
    bool prefer_model_=false;
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

constexpr float PupilDetector::MODEL_FALLBACK_CONFIDENCE;

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

using PerfClock=std::chrono::steady_clock;

float ReadEnvFloat(const char* name,float default_value,
                   float min_value,float max_value) {
    const char* raw=std::getenv(name);
    if(!raw||!raw[0]) return default_value;
    char* end=nullptr;
    const float parsed=std::strtof(raw,&end);
    if(end==raw||!std::isfinite(parsed)) return default_value;
    return std::max(min_value,std::min(max_value,parsed));
}

struct AlgorithmConfig {
    float scrfd_tracking_hz;
    float scrfd_degraded_hz;
    float pupil_confidence_min;
    float one_euro_min_cutoff;
    float one_euro_beta;
    float one_euro_derivative_cutoff;

    static AlgorithmConfig Load() {
        AlgorithmConfig config;
        config.scrfd_tracking_hz=ReadEnvFloat(
            "SCRFD_TRACKING_HZ",30.f,5.f,120.f);
        config.scrfd_degraded_hz=ReadEnvFloat(
            "SCRFD_DEGRADED_HZ",60.f,10.f,180.f);
        config.pupil_confidence_min=ReadEnvFloat(
            "PUPIL_CONFIDENCE_MIN",0.45f,0.1f,0.9f);
        config.one_euro_min_cutoff=ReadEnvFloat(
            "ONE_EURO_MIN_CUTOFF",3.f,0.1f,20.f);
        config.one_euro_beta=ReadEnvFloat(
            "ONE_EURO_BETA",0.6f,0.f,5.f);
        config.one_euro_derivative_cutoff=ReadEnvFloat(
            "ONE_EURO_D_CUTOFF",1.f,0.1f,20.f);
        return config;
    }

    void Print() const {
        printf("[ALGO] scrfd_tracking_hz=%.1f scrfd_degraded_hz=%.1f "
               "pupil_conf_min=%.2f one_euro=(%.2f,%.2f,%.2f)\n",
               scrfd_tracking_hz,scrfd_degraded_hz,
               pupil_confidence_min,one_euro_min_cutoff,
               one_euro_beta,one_euro_derivative_cutoff);
    }
};

class LowPassFilter {
public:
    LowPassFilter():initialized_(false),value_(0.f){}

    float Filter(float value,float alpha) {
        alpha=Clamp01(alpha);
        if(!initialized_){
            value_=value;
            initialized_=true;
        } else {
            value_=alpha*value+(1.f-alpha)*value_;
        }
        return value_;
    }

    void Reset() { initialized_=false; value_=0.f; }

private:
    bool initialized_;
    float value_;
};

class OneEuroFilter {
public:
    OneEuroFilter(float min_cutoff,float beta,float derivative_cutoff)
        :min_cutoff_(min_cutoff),beta_(beta),
         derivative_cutoff_(derivative_cutoff),initialized_(false),
         last_raw_(0.f){}

    float Filter(float value,PerfClock::time_point now) {
        if(!initialized_){
            initialized_=true;
            last_time_=now;
            last_raw_=value;
            derivative_filter_.Reset();
            value_filter_.Reset();
            derivative_filter_.Filter(0.f,1.f);
            return value_filter_.Filter(value,1.f);
        }

        float dt=std::chrono::duration<float>(now-last_time_).count();
        dt=std::max(1.f/500.f,std::min(0.1f,dt));
        const float derivative=(value-last_raw_)/dt;
        const float filtered_derivative=derivative_filter_.Filter(
            derivative,Alpha(dt,derivative_cutoff_));
        const float cutoff=min_cutoff_+beta_*std::fabs(filtered_derivative);
        const float filtered=value_filter_.Filter(value,Alpha(dt,cutoff));
        last_time_=now;
        last_raw_=value;
        return filtered;
    }

    void Reset() {
        initialized_=false;
        derivative_filter_.Reset();
        value_filter_.Reset();
    }

private:
    static float Alpha(float dt,float cutoff) {
        const float tau=1.f/(2.f*3.14159265358979323846f*
                             std::max(0.01f,cutoff));
        return 1.f/(1.f+tau/dt);
    }

    float min_cutoff_;
    float beta_;
    float derivative_cutoff_;
    bool initialized_;
    float last_raw_;
    PerfClock::time_point last_time_;
    LowPassFilter derivative_filter_;
    LowPassFilter value_filter_;
};

enum TrackingMode {
    TRACKING_REACQUIRE=0,
    TRACKING_DEGRADED=1,
    TRACKING_STABLE=2
};

const char* TrackingModeName(TrackingMode mode) {
    switch(mode){
        case TRACKING_STABLE: return "TRACKING";
        case TRACKING_DEGRADED: return "DEGRADED";
        default: return "REACQUIRE";
    }
}

class AdaptiveFaceTracker {
public:
    AdaptiveFaceTracker(int img_w,int img_h,const AlgorithmConfig& config)
        :img_w_(img_w),img_h_(img_h),config_(config),valid_(false),
         mode_(TRACKING_REACQUIRE),missed_detections_(0),
         low_confidence_streak_(0),stable_pupil_streak_(0),
         detector_attempted_(false),detector_succeeded_(false),
         state_time_initialized_(false) {
        box_.fill(0.f);
        velocity_.fill(0.f);
    }

    bool ShouldRunDetector(PerfClock::time_point now) const {
        if(!valid_||mode_==TRACKING_REACQUIRE||!detector_attempted_)
            return true;
        const float hz=mode_==TRACKING_STABLE ?
            config_.scrfd_tracking_hz : config_.scrfd_degraded_hz;
        const float elapsed=std::chrono::duration<float>(
            now-last_detector_attempt_).count();
        return elapsed>=1.f/std::max(1.f,hz);
    }

    void Predict(PerfClock::time_point now) {
        if(!valid_){
            state_time_=now;
            state_time_initialized_=true;
            return;
        }
        if(!state_time_initialized_){
            state_time_=now;
            state_time_initialized_=true;
            return;
        }
        float dt=std::chrono::duration<float>(now-state_time_).count();
        dt=std::max(0.f,std::min(0.1f,dt));
        for(size_t i=0;i<box_.size();i++) box_[i]+=velocity_[i]*dt;
        ClampBox(box_);
        state_time_=now;
    }

    void UpdateDetection(const std::array<float,4>& measured,
                         PerfClock::time_point now) {
        last_detector_attempt_=now;
        detector_attempted_=true;
        std::array<float,4> clipped=measured;
        ClampBox(clipped);
        if(!valid_){
            box_=clipped;
            velocity_.fill(0.f);
        } else {
            float dt=detector_succeeded_ ?
                std::chrono::duration<float>(now-last_detector_success_).count() :
                1.f/config_.scrfd_tracking_hz;
            dt=std::max(1.f/180.f,std::min(0.25f,dt));
            for(size_t i=0;i<box_.size();i++){
                const float residual=clipped[i]-box_[i];
                box_[i]+=0.70f*residual;
                velocity_[i]+=0.05f*residual/dt;
            }
            ClampBox(box_);
        }
        valid_=true;
        detector_succeeded_=true;
        last_detector_success_=now;
        missed_detections_=0;
        if(mode_==TRACKING_REACQUIRE) mode_=TRACKING_DEGRADED;
        state_time_=now;
        state_time_initialized_=true;
    }

    void UpdateDetectionMiss(PerfClock::time_point now) {
        last_detector_attempt_=now;
        detector_attempted_=true;
        missed_detections_++;
        const float age=detector_succeeded_ ?
            std::chrono::duration<float>(now-last_detector_success_).count() :
            1.f;
        if(!valid_||missed_detections_>=3||age>0.15f){
            valid_=false;
            mode_=TRACKING_REACQUIRE;
            velocity_.fill(0.f);
        } else {
            mode_=TRACKING_DEGRADED;
        }
    }

    void UpdatePupilQuality(const PupilObservation& left,
                            const PupilObservation& right) {
        const bool any_valid=left.valid||right.valid;
        const bool both_valid=left.valid&&right.valid;
        const float confidence=both_valid ?
            0.5f*(left.confidence+right.confidence) :
            (left.valid?left.confidence:right.confidence)*0.75f;
        if(!any_valid||confidence<config_.pupil_confidence_min){
            low_confidence_streak_++;
            stable_pupil_streak_=0;
            mode_=low_confidence_streak_>=3 ?
                TRACKING_REACQUIRE : TRACKING_DEGRADED;
            if(low_confidence_streak_>=3) valid_=false;
            return;
        }
        low_confidence_streak_=0;
        stable_pupil_streak_++;
        if(stable_pupil_streak_>=6&&missed_detections_==0)
            mode_=TRACKING_STABLE;
        else if(mode_==TRACKING_REACQUIRE)
            mode_=TRACKING_DEGRADED;
    }

    bool Valid() const { return valid_; }
    TrackingMode Mode() const { return mode_; }
    const std::array<float,4>& Box() const { return box_; }

private:
    void ClampBox(std::array<float,4>& box) const {
        box[0]=std::max(0.f,std::min(static_cast<float>(img_w_-2),box[0]));
        box[1]=std::max(0.f,std::min(static_cast<float>(img_h_-2),box[1]));
        box[2]=std::max(box[0]+2.f,
                        std::min(static_cast<float>(img_w_),box[2]));
        box[3]=std::max(box[1]+2.f,
                        std::min(static_cast<float>(img_h_),box[3]));
    }

    int img_w_;
    int img_h_;
    AlgorithmConfig config_;
    bool valid_;
    TrackingMode mode_;
    std::array<float,4> box_;
    std::array<float,4> velocity_;
    int missed_detections_;
    int low_confidence_streak_;
    int stable_pupil_streak_;
    bool detector_attempted_;
    bool detector_succeeded_;
    bool state_time_initialized_;
    PerfClock::time_point state_time_;
    PerfClock::time_point last_detector_attempt_;
    PerfClock::time_point last_detector_success_;
};

// 全局变量
VISUALIZER* g_visualizer=nullptr;
GazeCalibrator g_calibrator;
std::mutex g_calib_mtx;
std::mutex mtx_image;
std::condition_variable cv_image_ready;
std::atomic<bool> stop_inference(false);
const int MAX_QUEUE_SIZE=1;
const int FRAME_POOL_SIZE=MAX_QUEUE_SIZE+1;
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
std::atomic<uint64_t> g_perf_detector_runs(0);
std::atomic<uint64_t> g_perf_gaze_valid(0);
std::atomic<uint64_t> g_perf_model_recovery(0);
std::atomic<int> g_tracking_mode(TRACKING_REACQUIRE);
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
    int slot=TryAcquireFrameSlot(free_mask);
    if(slot<0){
        std::unique_lock<std::mutex> recycle_lock(mtx_image,std::try_to_lock);
        if(!recycle_lock.owns_lock()||image_queue.empty()) return false;
        slot=image_queue.front().pool_index;
        image_queue.pop();
        g_queue_depth.store(0,std::memory_order_release);
        g_perf_dropped.fetch_add(1,std::memory_order_relaxed);
    }

    if(copy_tensor_buffer(source,frame_pool[slot])!=SSNE_ERRCODE_NO_ERROR){
        ReturnFrameSlot(free_mask,slot);
        return false;
    }

    std::unique_lock<std::mutex> lock(mtx_image,std::try_to_lock);
    if(!lock.owns_lock()){
        ReturnFrameSlot(free_mask,slot);
        return false;
    }

    if(!image_queue.empty()){
        const InferenceFrame stale=image_queue.front();
        image_queue.pop();
        ReturnFrameSlot(free_mask,stale.pool_index);
        g_perf_dropped.fetch_add(1,std::memory_order_relaxed);
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
    static uint64_t last_detector_runs=0;
    static uint64_t last_gaze_valid=0;
    static uint64_t last_model_recovery=0;
    const double elapsed=std::chrono::duration<double>(now-last_log).count();
    if(elapsed<1.0) return;

    const uint64_t captured=g_perf_captured.load(std::memory_order_relaxed);
    const uint64_t enqueued=g_perf_enqueued.load(std::memory_order_relaxed);
    const uint64_t dropped=g_perf_dropped.load(std::memory_order_relaxed);
    const uint64_t inferred=g_perf_inferred.load(std::memory_order_acquire);
    const uint64_t latency_us=g_perf_latency_us.load(std::memory_order_relaxed);
    const uint64_t detector_runs=
        g_perf_detector_runs.load(std::memory_order_relaxed);
    const uint64_t gaze_valid=
        g_perf_gaze_valid.load(std::memory_order_relaxed);
    const uint64_t model_recovery=
        g_perf_model_recovery.load(std::memory_order_relaxed);
    const uint64_t inferred_delta=inferred-last_inferred;
    const uint64_t latency_delta=latency_us-last_latency_us;
    const uint64_t latency_max_us=
        g_perf_latency_max_us.exchange(0,std::memory_order_relaxed);
    const double latency_avg_ms=inferred_delta==0?0.0:
        static_cast<double>(latency_delta)/inferred_delta/1000.0;

    printf("[PERF] capture=%llu enqueue=%llu drop=%llu "
           "epp_fps=%.1f gaze=%llu scrfd=%llu model_recovery=%llu "
           "queue=%d latency_ms_avg=%.1f latency_ms_max=%.1f mode=%s\n",
           static_cast<unsigned long long>(captured-last_captured),
           static_cast<unsigned long long>(enqueued-last_enqueued),
           static_cast<unsigned long long>(dropped-last_dropped),
           inferred_delta/elapsed,
           static_cast<unsigned long long>(gaze_valid-last_gaze_valid),
           static_cast<unsigned long long>(detector_runs-last_detector_runs),
           static_cast<unsigned long long>(model_recovery-last_model_recovery),
           g_queue_depth.load(std::memory_order_acquire),
           latency_avg_ms,latency_max_us/1000.0,
           TrackingModeName(static_cast<TrackingMode>(
               g_tracking_mode.load(std::memory_order_acquire))));

    last_captured=captured;
    last_enqueued=enqueued;
    last_dropped=dropped;
    last_inferred=inferred;
    last_latency_us=latency_us;
    last_detector_runs=detector_runs;
    last_gaze_valid=gaze_valid;
    last_model_recovery=model_recovery;
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
    static bool baseline_ready=false;
    static float open_baseline=0.f;
    static PerfClock::time_point blink_start;
    static PerfClock::time_point open_start;

    if(!valid){
        closed_streak=0;
        open_streak=0;
        if(!suppress_until_open) blink_active=false;
        return;
    }

    if(!baseline_ready){
        open_baseline=std::max(0.02f,dark_ratio);
        baseline_ready=true;
    }
    const float close_threshold=std::max(0.015f,open_baseline*0.60f);
    const float open_threshold=std::max(close_threshold+0.008f,
                                        open_baseline*0.78f);
    const bool is_closed=dark_ratio<close_threshold;
    const bool is_open=dark_ratio>open_threshold;
    const PerfClock::time_point now=PerfClock::now();

    if(!blink_active&&!suppress_until_open&&!is_closed){
        open_baseline=0.99f*open_baseline+0.01f*dark_ratio;
        open_baseline=std::max(0.02f,std::min(0.40f,open_baseline));
    }

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
                           AlgorithmConfig config,
                           int img_w, int img_h){
    printf("[Thread] Inference started\n");
    FaceDetectionResult det_result;
    AdaptiveFaceTracker face_tracker(img_w,img_h,config);
    PupilObservation left_history,right_history;
    OneEuroFilter gaze_filter_x(config.one_euro_min_cutoff,
                                config.one_euro_beta,
                                config.one_euro_derivative_cutoff);
    OneEuroFilter gaze_filter_y(config.one_euro_min_cutoff,
                                config.one_euro_beta,
                                config.one_euro_derivative_cutoff);
    bool gaze_filter_initialized=false;
    int gaze_hold_frames=0;

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
        const PerfClock::time_point now=PerfClock::now();
        face_tracker.Predict(now);

        bool ran_detector=false;
        if(face_tracker.ShouldRunDetector(now)){
            ran_detector=true;
            det_result.Clear();
            detector->Predict(&mirrored,&det_result,0.4f);
            g_perf_detector_runs.fetch_add(1,std::memory_order_relaxed);
            if(!det_result.boxes.empty())
                face_tracker.UpdateDetection(det_result.boxes[0],now);
            else
                face_tracker.UpdateDetectionMiss(now);
        }
        g_tracking_mode.store(face_tracker.Mode(),std::memory_order_release);

        if(!face_tracker.Valid()){
            UpdateBlinkDetection(false,0.f);
            left_history=PupilObservation();
            right_history=PupilObservation();
            gaze_filter_x.Reset();
            gaze_filter_y.Reset();
            gaze_filter_initialized=false;
            gaze_hold_frames=0;
            if(g_visualizer){
                std::vector<std::array<float,4>> no_face_boxes;
                no_face_boxes.push_back(
                    {0.f,480.f,static_cast<float>(img_w),
                     static_cast<float>(img_h*2)});
                g_visualizer->Draw(no_face_boxes);
            }
            continue;
        }

        const std::array<float,4> face=face_tracker.Box();
        float fw=face[2]-face[0], fh=face[3]-face[1];
        float lcx=face[0]+fw*0.30f,lcy=face[1]+fh*0.35f;
        float rcx=face[0]+fw*0.70f,rcy=face[1]+fh*0.35f;
        if(ran_detector&&det_result.landmarks_per_face>=2&&
           det_result.landmarks.size()>=2){
            lcx=det_result.landmarks[0][0];
            lcy=det_result.landmarks[0][1];
            rcx=det_result.landmarks[1][0];
            rcy=det_result.landmarks[1][1];
        }

        auto lb=GetEyeBox(face,lcx,lcy,img_w,img_h);
        auto rb=GetEyeBox(face,rcx,rcy,img_w,img_h);
        const bool frame_ready=pupil_det->PrepareFrame(&mirrored);
        const bool allow_model_recovery=pupil_det->PreferModel()||
            face_tracker.Mode()!=TRACKING_STABLE;
        PupilObservation left,right;
        if(frame_ready){
            left=pupil_det->Detect(lb,left_history.valid?&left_history:nullptr,
                                   allow_model_recovery);
            right=pupil_det->Detect(rb,right_history.valid?&right_history:nullptr,
                                    allow_model_recovery);
        }
        if(left.used_model)
            g_perf_model_recovery.fetch_add(1,std::memory_order_relaxed);
        if(right.used_model)
            g_perf_model_recovery.fetch_add(1,std::memory_order_relaxed);

        if(left.valid) left_history=left;
        else {
            left_history.confidence*=0.70f;
            if(left_history.confidence<0.20f) left_history.valid=false;
        }
        if(right.valid) right_history=right;
        else {
            right_history.confidence*=0.70f;
            if(right_history.confidence<0.20f) right_history.valid=false;
        }

        face_tracker.UpdatePupilQuality(left,right);
        g_tracking_mode.store(face_tracker.Mode(),std::memory_order_release);

        float blink_dark_ratio=0.f;
        bool blink_valid=false;
        if(left.blink_valid&&right.blink_valid){
            blink_dark_ratio=0.5f*(left.blink_dark_ratio+
                                   right.blink_dark_ratio);
            blink_valid=true;
        } else if(left.blink_valid){
            blink_dark_ratio=left.blink_dark_ratio;
            blink_valid=true;
        } else if(right.blink_valid){
            blink_dark_ratio=right.blink_dark_ratio;
            blink_valid=true;
        }
        UpdateBlinkDetection(blink_valid,blink_dark_ratio);

        float ox=0.f,oy=0.f,gaze_confidence=0.f;
        bool has_gaze=false;
        const float left_ox=left.valid?(left.x-lcx)/fw:0.f;
        const float left_oy=left.valid?(left.y-lcy)/fh:0.f;
        const float right_ox=right.valid?(right.x-rcx)/fw:0.f;
        const float right_oy=right.valid?(right.y-rcy)/fh:0.f;
        if(left.valid&&right.valid){
            const float weight_sum=left.confidence+right.confidence;
            const float divergence=std::sqrt(
                (left_ox-right_ox)*(left_ox-right_ox)+
                (left_oy-right_oy)*(left_oy-right_oy));
            if(divergence>0.12f&&
               left.confidence>right.confidence*1.20f){
                ox=left_ox; oy=left_oy;
                gaze_confidence=left.confidence*0.80f;
            } else if(divergence>0.12f&&
                      right.confidence>left.confidence*1.20f){
                ox=right_ox; oy=right_oy;
                gaze_confidence=right.confidence*0.80f;
            } else {
                ox=(left.confidence*left_ox+right.confidence*right_ox)/
                   std::max(0.001f,weight_sum);
                oy=(left.confidence*left_oy+right.confidence*right_oy)/
                   std::max(0.001f,weight_sum);
                gaze_confidence=0.5f*weight_sum*(divergence>0.12f?0.55f:1.f);
            }
            has_gaze=true;
        } else if(left.valid){
            ox=left_ox; oy=left_oy;
            gaze_confidence=left.confidence*0.75f;
            has_gaze=true;
        } else if(right.valid){
            ox=right_ox; oy=right_oy;
            gaze_confidence=right.confidence*0.75f;
            has_gaze=true;
        }

        if(has_gaze&&gaze_confidence>=0.30f){
            std::lock_guard<std::mutex> lock(g_calib_mtx);
            if(g_calibrator.is_calibrating&&left.valid&&right.valid&&
               gaze_confidence>=config.pupil_confidence_min)
                g_calibrator.AddSample(ox,oy);
        }

        float sgx=g_last_sgx,sgy=g_last_sgy;
        bool genuine_gaze=has_gaze&&gaze_confidence>=0.30f;
        if(genuine_gaze){
            float gnx,gny;
            {
                std::lock_guard<std::mutex> lock(g_calib_mtx);
                g_calibrator.MapGaze(ox,oy,gnx,gny);
            }
            sgx=Clamp01(gaze_filter_x.Filter(gnx,now));
            sgy=Clamp01(gaze_filter_y.Filter(gny,now));
            gaze_filter_initialized=true;
            gaze_hold_frames=0;
            g_perf_gaze_valid.fetch_add(1,std::memory_order_relaxed);
        } else if(gaze_filter_initialized&&gaze_hold_frames<2){
            gaze_hold_frames++;
        } else {
            continue;
        }

        g_last_sgx=sgx; g_last_sgy=sgy;
        GazeDirection dir=ClassifyGaze(sgx,sgy);
        // Gaze打印已关闭

        // OSD绘制：用VISUALIZER的Draw接口，只画矩形框
        if(g_visualizer){
            std::vector<std::array<float,4>> draw_boxes;
            draw_boxes.reserve(32);

            // 人脸框
            draw_boxes.push_back(face);

            // 左瞳孔框
            if(left.valid){
                float ps=8.f;
                draw_boxes.push_back(
                    {left.x-ps,left.y-ps,left.x+ps,left.y+ps});
            }
            // 右瞳孔框
            if(right.valid){
                float ps=8.f;
                draw_boxes.push_back(
                    {right.x-ps,right.y-ps,right.x+ps,right.y+ps});
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
    printf("[Thread] Inference stopped\n");
}

int main(){
    uint8_t load_flag=0;
    int img_width=640, img_height=480;
    array<int,2> det_shape={640,480};
    const AlgorithmConfig algorithm_config=AlgorithmConfig::Load();
    string path_det="/app_demo/app_assets/models/face_640x480.m1model";
    string path_pupil="/app_demo/app_assets/models/pupil_gap.m1model";
    const char* pupil_model_env=std::getenv("PUPIL_GAP_MODEL");
    if(pupil_model_env&&pupil_model_env[0]) path_pupil=pupil_model_env;
    bool enable_pupil_model=true;
    bool prefer_pupil_model=false;
    const char* pupil_mode_env=std::getenv("PUPIL_DETECT_MODE");
    if(pupil_mode_env){
        const std::string pupil_mode(pupil_mode_env);
        if(pupil_mode=="classic") enable_pupil_model=false;
        else if(pupil_mode=="model") prefer_pupil_model=true;
    }

    if(ssne_initial()){fprintf(stderr,"SSNE init failed\n");return -1;}
    algorithm_config.Print();

    array<int,2> img_shape={img_width,img_height};
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
    pupil_det.Initialize(img_width,img_height,path_pupil,
                         enable_pupil_model,prefer_pupil_model);
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
                                  algorithm_config,
                                  img_width,img_height);
    std::thread listener_thread(keyboard_listener);

    uint64_t num_frames=0;
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
