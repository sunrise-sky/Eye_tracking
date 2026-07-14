/*
 * @Filename: pipeline_image.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#include "../include/common.hpp"
#include <iostream>
#include <unistd.h>

/**
 * @brief 图像处理器初始化函数
 * @param in_img_shape 输入图像尺寸 [宽度, 高度]
 * @param in_scale Binning降采样倍数（保留参数以兼容接口，但不使用）
 */
// void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape, 
//   BinningRatioType in_scale) {
bool IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape)
{
    if (initialized_) return true;
    if (!in_img_shape || (*in_img_shape)[0] <= 0 || (*in_img_shape)[1] <= 0) {
        last_error_code_ = -1;
        fprintf(stderr, "[CAMERA] invalid image shape\n");
        return false;
    }
    img_shape = *in_img_shape;      // 保存原始图像尺寸
    
    // 在线图像配置参数
    format_online = SSNE_Y_8;                    // 图像格式：8位灰度图
    
    // pipe0设置
    OnlineSetOutputImage(kPipeline0, format_online, 640, 480);  // 输出裁剪后的图像尺寸
    
    // 打开pipe0（裁剪图像通道）
    // int res0 = OpenOnlinePipeline(kPipeline0);
    int res0 = OpenDualSnrOnline(kPipeline0);
    if (res0 != 0) {
        last_error_code_ = res0;
        fprintf(stderr, "[CAMERA] open dual-sensor pipeline failed, ret=%d\n",
                res0);
        return false;
    }
    initialized_ = true;
    last_error_code_ = 0;
    printf("[CAMERA] dual-sensor pipeline ready: %dx%d format=Y8\n",
           img_shape[0], img_shape[1]);
    return true;
}

/**
 * @brief 从pipeline获取图像数据（裁剪图）
 * @param img_sensor 输出参数：存储从pipe0获取的裁剪图像（720×540）
 */
bool IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor) {
    if (!initialized_ || !img_sensor) {
        last_error_code_ = -1;
        return false;
    }
    int capture_code = -1;  // pipe0采集返回码
    
    // 从pipe0获取裁剪后的图像数据
    capture_code = GetImageData(img_sensor, kPipeline0, kSensor0, 0);
    
    // 检查pipe0采集是否成功
    last_error_code_ = capture_code;
    return capture_code == 0;
}

bool IMAGEPROCESSOR::GetDualImage(ssne_tensor_t* img_out0,
                                  ssne_tensor_t* img_out1) {
    if (!initialized_ || !img_out0 || !img_out1) {
        last_error_code_ = -1;
        return false;
    }
    int capture_code = GetDualImageData(img_out0, img_out1, kPipeline0, 0);
    last_error_code_ = capture_code;
    return capture_code == 0;
}

bool IMAGEPROCESSOR::Restart() {
    if (img_shape[0] <= 0 || img_shape[1] <= 0) {
        last_error_code_ = -1;
        return false;
    }
    std::array<int, 2> shape = img_shape;
    Release();
    return Initialize(&shape);
}


/**
 * @brief 释放图像处理器资源，关闭pipeline
 */
void IMAGEPROCESSOR::Release()
{
    if (!initialized_) return;
    CloseOnlinePipeline(kPipeline0);  // 关闭pipe0（裁剪图像通道）
    initialized_ = false;
    printf("[CAMERA] online pipeline closed\n");
}
