#include <opencv2/opencv.hpp>

#include "transparent_chassis_cuda_api.cuh"

using namespace ParkingPerception::TransparentChassisCuda;

int main()
{
  //构造实例
  std::string config_path =
      "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/config/TransparentChassisCuda.yaml";
  std::shared_ptr<TransparentChassis> transparent_chassis = CreateTransparentChassis(config_path);

  //初始化
  if (0 != transparent_chassis->init())
  {
    return 0;
  }

  //准备图像
  std::string img0_path = "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/test/0.png";
  std::string img1_path = "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/test/1.png";
  std::string img2_path = "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/test/2.png";
  cv::Mat img0 = cv::imread(img0_path);
  cv::Mat img1 = cv::imread(img1_path);
  cv::Mat img2 = cv::imread(img2_path);

  //虚拟定位数据
  LocData loc0 = LocData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  LocData loc1 = LocData(0.0, 3.3, 0.0, 0.0, 0.0, 0.0, 0.02);
  LocData loc2 = LocData(0.0, 5.43, 0.0, 0.0, 0.0, 0.0, 0.02);

  //运行
  transparent_chassis->process(img0, loc0);
  cv::Mat output0;
  transparent_chassis->get_result(output0);

  //运行
  transparent_chassis->process(img1, loc1);
  cv::Mat output1;
  transparent_chassis->get_result(output1);

  //运行
  transparent_chassis->process(img2, loc2);
  cv::Mat output2;
  transparent_chassis->get_result(output2);

  std::string save_path0 = "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/test/output0.png";
  cv::imwrite(save_path0, output0);
  std::cout << "Save output0 img: " << save_path0 << std::endl;

  std::string save_path1 = "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/test/output1.png";
  cv::imwrite(save_path1, output1);
  std::cout << "Save output1 img: " << save_path1 << std::endl;

  std::string save_path2 = "/hostdata/projects/parking_perception/modules/TransparentChassisCuda/test/output2.png";
  cv::imwrite(save_path2, output2);
  std::cout << "Save output2 img: " << save_path2 << std::endl;

  return 0;
}