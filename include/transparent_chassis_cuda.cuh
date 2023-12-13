/*
 * @Author: zjj
 * @Date: 2023-12-12 15:16:50
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-13 13:48:02
 * @FilePath: /TransparentChassisCuda/include/transparent_chassis_cuda.cuh
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "common.cuh"

namespace ParkingPerception
{
  namespace TransparentChassisCuda
  {
    struct LocData
    {
      LocData() : time(0.0), x(0.0), y(0.0), z(0.0), roll(0.0), pitch(0.0), yaw(0.0){};
      LocData(double t, double xx, double yy, double zz, double r, double p, double y)
          : time(t), x(xx), y(yy), z(zz), roll(r), pitch(p), yaw(y){};
      double time;
      double x;
      double y;
      double z;
      double roll;
      double pitch;
      double yaw;
    };

    static __device__ float bicubic(float x);
    static __device__ void getImpactFactors(float rowU, float colV, float *rowImFac, float *colImFac, int starti,
                                            int startj);
    static __device__ uchar3 bicubic_interpolation(float src_x, float src_y, int width, int height, uint8_t *src,
                                                   float *rowImFac, float *colImFac);
    static __device__ uchar3 bilinear_interpolation(float src_x, float src_y, int width, int height, int line_size,
                                                    uint8_t fill_value, uint8_t *src);
    static __global__ void process_img_kernel(uchar *img_pre, uchar *img_fusion, uchar *img_now, uchar *weight, int w,
                                              int h, float *affine_now2pre, float4 *rowImFac, float4 *colImFac);

    class TransparentChassis
    {
    public:
      TransparentChassis(std::string config_path);
      ~TransparentChassis();
      int init();
      int process(const cv::Mat &img_now, const LocData &loc_now);
      void get_result(cv::Mat &out);

    private:
      int load_config();
      void get_warpaffine(const LocData &loc_now);
      void destroy();

    private:
      //配置文件
      std::string config_path_;
      // cuda
      cudaStream_t stream;
      //拼接图参数
      int w = 0;
      int h = 0;
      float bev_ratio;
      float bev_center_x;
      float bev_center_y;
      int shift_lr = 0;
      int shift_tb = 0;
      //平滑卷积
      int filter_kernel_size = 0;
      uchar3 *weight_device = nullptr; // gpu上存储的融合时权重
      //图像
      cv::Mat output;                      //输出图像
      uchar3 *img_pre_device = nullptr;    // gpu上存储的上一帧图像
      uchar3 *img_fusion_device = nullptr; // gpu上存储的上一帧变换到当前时刻的图像
      uchar3 *img_now_device = nullptr;    // gpu上存储的当前时刻图像
      //仿射变换
      float *affine_now2pre_host = nullptr;   // cpu上存储的img_pre -> img_now仿射变换矩阵
      float *affine_now2pre_device = nullptr; // gpu上存储的img_pre -> img_now仿射变换矩阵
      //定位
      LocData loc_data_pre; //上一帧位置
      //双立方插值
      float4 *rowImFac_device = nullptr; // gpu上存储的行影响因子
      float4 *colImFac_device = nullptr; // gpu上存储的列影响因子
      //是否为首帧
      bool is_first;
    };
  } // namespace TransparentChassisCuda
} // namespace ParkingPerception