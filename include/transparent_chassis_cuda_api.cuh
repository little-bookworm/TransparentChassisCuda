/*
 * @Author: zjj
 * @Date: 2023-12-12 15:16:50
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-12 16:38:57
 * @FilePath: /TransparentChassisCuda/include/transparent_chassis_cuda_api.cuh
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#pragma once

#include "transparent_chassis_cuda.cuh"

namespace ParkingPerception
{
namespace TransparentChassisCuda
{
    TransparentChassis *CreateTransparentChassis(std::string config_file);
} // namespace TransparentChassisCuda
} // namespace ParkingPerception