#include "transparent_chassis_cuda.cuh"

namespace ParkingPerception
{
    namespace TransparentChassisCuda
    {
        static __device__ float bicubic(float x)
        {
            float a = -0.75; // opencv取值，默认为-0.5
            float res = 0.0;
            x = abs(x);
            if (x <= 1.0)
                res = (a + 2) * x * x * x - (a + 3) * x * x + 1;
            else if (x < 2.0)
                res = a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
            return res;
        }

        static __device__ void getImpactFactors(float rowU, float colV, float *rowImFac, float *colImFac, int starti, int startj)
        {
            //取整
            int row = (int)rowU;
            int col = (int)colV;
            float temp;
            //计算行系数因子
            for (int i = 0; i < 4; i++)
            {
                if (starti + i <= 0)
                    temp = rowU - row - (starti + i);
                else
                    temp = (starti + i) - (rowU - row);
                rowImFac[i] = bicubic(temp);
                // printf("i:%d,temp:%.1f,rowImFac:%.1f\n", i, temp, rowImFac[i]);
            }
            //计算列系数因子
            for (int j = 0; j < 4; j++)
            {
                if (startj + j <= 0)
                    temp = colV - col - (startj + j);
                else
                    temp = (startj + j) - (colV - col);
                colImFac[j] = bicubic(temp);
            }
        }

        static __device__ uchar3 bicubic_interpolation(float src_x, float src_y, int width, int height, uint8_t *src, float *rowImFac, float *colImFac)
        {
            //计算outputMat(row,col)在原图中的位置坐标
            float inputrowf = src_y;
            float inputcolf = src_x;
            // printf("inputrowf:%.1f,inputcolf:%.1f\n", inputrowf, inputcolf);
            //取整
            int interow = (int)inputrowf;
            int intecol = (int)inputcolf;
            float row_dy = inputrowf - interow;
            float col_dx = inputcolf - intecol;
            //因为扩展了边界，所以+2
            // interow += 2;
            // intecol += 2;
            int starti = -1, startj = -1;
            //计算行影响因子，列影响因子
            getImpactFactors(inputrowf, inputcolf, rowImFac, colImFac, starti, startj);
            // printf("rowImFac:%.2f,%.2f,%.2f,%.2f\n", rowImFac[0], rowImFac[1], rowImFac[2], rowImFac[3]);
            // printf("colImFac:%.1f,%.1f,%.1f,%.1f\n", colImFac[0], colImFac[1], colImFac[2], colImFac[3]);
            //计算输出图像(row,col)的值
            // Vec3f tempvec(0, 0, 0);
            float3 c_3f = {0, 0, 0};
            for (int i = starti; i < starti + 4; i++)
            {
                for (int j = startj; j < startj + 4; j++)
                {
                    uint8_t *src_ptr = src + (interow + i) * 3 * width + (intecol + j) * 3;
                    float weight = rowImFac[i - starti] * colImFac[j - startj];
                    // if (i == 0 && j == 0)
                    // {
                    //     printf("i:%d,j:%d,weight:%.1f\n", i, j, weight);
                    // }
                    // printf("i:%d,j:%d,weight:%.1f\n", i, j, weight);
                    c_3f.x += src_ptr[0] * weight;
                    c_3f.y += src_ptr[1] * weight;
                    c_3f.z += src_ptr[2] * weight;
                }
            }
            c_3f.x = floorf(c_3f.x + 0.5f); //四舍五入
            c_3f.y = floorf(c_3f.y + 0.5f); //四舍五入
            c_3f.z = floorf(c_3f.z + 0.5f); //四舍五入
            uchar3 c = {uchar(c_3f.x), uchar(c_3f.y), uchar(c_3f.z)};

            return c;
        }

        static __device__ uchar3 bilinear_interpolation(float src_x, float src_y, int width, int height, int line_size, uint8_t fill_value, uint8_t *src)
        {
            float c0 = fill_value, c1 = fill_value, c2 = fill_value;
            //双线性插值
            if (src_x < -1 || src_x >= width || src_y < -1 || src_y >= height)
            {
                // out of range
                // src_x < -1时，其高位high_x < 0，超出范围
                // src_x >= -1时，其高位high_x >= 0，存在取值
            }
            else
            {
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_values[] = {fill_value, fill_value, fill_value};
                float ly = src_y - y_low;
                float lx = src_x - x_low;
                float hy = 1 - ly;
                float hx = 1 - lx;
                float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t *v1 = const_values;
                uint8_t *v2 = const_values;
                uint8_t *v3 = const_values;
                uint8_t *v4 = const_values;
                if (y_low >= 0)
                {
                    if (x_low >= 0)
                        v1 = src + y_low * line_size + x_low * 3;

                    if (x_high < width)
                        v2 = src + y_low * line_size + x_high * 3;
                }

                if (y_high < height)
                {
                    if (x_low >= 0)
                        v3 = src + y_high * line_size + x_low * 3;

                    if (x_high < width)
                        v4 = src + y_high * line_size + x_high * 3;
                }

                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f); //四舍五入
                c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f); //四舍五入
                c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f); //四舍五入
            }
            return make_uchar3(c0, c1, c2);
        }

        static __global__ void process_img_kernel(uchar *img_pre, uchar *img_fusion, uchar *img_now, uchar *weight, int w, int h, float *affine_now2pre, float4 *rowImFac, float4 *colImFac)
        {

            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int j = blockDim.y * blockIdx.y + threadIdx.y;
            if (i >= w || j >= h)
                return;

            float img_pre_x = affine_now2pre[0] * i + affine_now2pre[1] * j + affine_now2pre[2]; //当前ij对应的上一帧图像坐标
            float img_pre_y = affine_now2pre[3] * i + affine_now2pre[4] * j + affine_now2pre[5]; //当前ij对应的上一帧图像坐标

            //双线性插值
            // uchar3 c = bilinear_interpolation(img_pre_x, img_pre_y, w, h, 3 * w, 0, img_pre);
            //双立方插值
            uchar3 c = bicubic_interpolation(img_pre_x, img_pre_y, w, h, img_pre, (float *)(rowImFac + j * w + i), (float *)(colImFac + j * w + i));

            // pre->now
            uint8_t *fusion_ptr = img_fusion + j * w * 3 + i * 3;
            fusion_ptr[0] = c.x;
            fusion_ptr[1] = c.y;
            fusion_ptr[2] = c.z;

            // pre_now + now
            uint8_t *weight_ij = weight + j * w * 3 + i * 3;
            uint8_t *now_ptr = img_now + j * w * 3 + i * 3;
            if (weight_ij[0] == 0) //权重为0即中间黑框部分，仅使用pre2now，不做融合
            {
                now_ptr[0] = fusion_ptr[0];
                now_ptr[1] = fusion_ptr[1];
                now_ptr[2] = fusion_ptr[2];
                return;
            }
            float r = weight_ij[0] / 255.0;

            fusion_ptr[0] = (1 - r) * fusion_ptr[0] + r * now_ptr[0];
            fusion_ptr[1] = (1 - r) * fusion_ptr[1] + r * now_ptr[1];
            fusion_ptr[2] = (1 - r) * fusion_ptr[2] + r * now_ptr[2];
        }

        TransparentChassis::TransparentChassis(std::string config_path) : config_path_(config_path)
        {
        }

        TransparentChassis::~TransparentChassis()
        {
            destroy();
        }

        int TransparentChassis::init()
        {
            if (0 != load_config())
            {
                std::cout << "[TransparentChassis]->[init] Failed to load config file." << std::endl;
                return -1;
            }

            //初始化融合权重
            cv::Mat weight_tmp(cv::Size(w, h), CV_8UC3, cv::Scalar(255, 255, 255));
            cv::rectangle(
                weight_tmp, cv::Point(shift_lr - int(filter_kernel_size / 2), shift_tb - int(filter_kernel_size / 2)),
                cv::Point(w - shift_lr + int(filter_kernel_size / 2), h - shift_tb + int(filter_kernel_size / 2)),
                cv::Scalar{0}, -1);                                                         // weight中间部分置黑，中间只用前一帧变换过来的图,外围只用前一帧变换后的图，中间平滑过渡
            cv::Mat weight;                                                                 //形成平滑的weight
            cv::blur(weight_tmp, weight, cv::Size(filter_kernel_size, filter_kernel_size)); //卷积核大小，根据透明部分是否有波纹修改，越大波纹越不明显，但是会变模糊

            // 设置stream;
            checkRuntime(cudaStreamCreate(&stream));
            //分配内存
            checkRuntime(cudaMalloc(&weight_device, sizeof(uchar3) * w * h));
            checkRuntime(cudaMalloc(&img_pre_device, sizeof(uchar3) * w * h));
            checkRuntime(cudaMalloc(&img_fusion_device, sizeof(uchar3) * w * h));
            checkRuntime(cudaMalloc(&img_now_device, sizeof(uchar3) * w * h));
            affine_now2pre_host = new float[6];
            checkRuntime(cudaMalloc(&affine_now2pre_device, sizeof(float) * 6));
            checkRuntime(cudaMalloc(&rowImFac_device, sizeof(float4) * w * h));
            checkRuntime(cudaMalloc(&colImFac_device, sizeof(float4) * w * h));

            // cpu -> gpu
            checkRuntime(cudaMemcpy(weight_device, weight.data, sizeof(uchar3) * w * h, cudaMemcpyHostToDevice));

            //初始化首帧
            is_first = true;

            //融合结果分配内存
            output.create(h, w, CV_8UC3);

            std::cout << "[TransparentChassis]->[init] Init success." << std::endl;

            return 0;
        }

        int TransparentChassis::load_config()
        {
            //导入yaml文件
            YAML::Node config;
            try
            {
                config = YAML::LoadFile(config_path_);
            }
            catch (const std::exception &e)
            {
                std::cout << "[TransparentChassis]->[load_config] No config file: " << config_path_ << std::endl;
                return -1;
            }

            //导入配置参数
            auto img_params = config["img_params"];
            w = img_params["w"].as<int>();
            h = img_params["h"].as<int>();
            if (w == 0 || h == 0)
            {
                std::cout << "[TransparentChassis]->[load_config] Img_params size error!!!" << std::endl;
                return -1;
            }
            bev_ratio = config["stitch_params"]["bev_ratio"].as<float>();
            bev_center_x = config["stitch_params"]["center_x"].as<float>();
            bev_center_y = config["stitch_params"]["center_y"].as<float>();
            shift_lr = config["stitch_params"]["shift_lr"].as<int>();
            shift_tb = config["stitch_params"]["shift_tb"].as<int>();
            if (shift_lr == 0 || shift_tb == 0)
            {
                std::cout << "[TransparentChassis]->[load_config] shift_lr or shift_tb error!!!" << std::endl;
                return -1;
            }
            filter_kernel_size = config["filter_kernel_size"].as<int>();
            if (filter_kernel_size == 0)
            {
                std::cout << "[TransparentChassis]->[load_config] filter_kernel_size error!!!" << std::endl;
                return -1;
            }

            return 0;
        }

        int TransparentChassis::process(const cv::Mat &img_now, const LocData &loc_now)
        {
            //判断输入图像是否正常
            if (img_now.rows != h || img_now.cols != w)
            {
                std::cout << "[TransparentChassis]->[process] Input img size error!!!" << std::endl;
                return -1;
            }
            if (img_now.empty())
            {
                std::cout << "[TransparentChassis]->[process] Input img is empty!!!" << std::endl;
                return -1;
            }

            //如果为首帧，初始化img_pre_device和loc_data_pre
            if (is_first)
            {
                checkRuntime(cudaMemcpyAsync(img_pre_device, img_now.data, w * h * sizeof(uchar3), cudaMemcpyHostToDevice, stream));
                loc_data_pre = loc_now;
                is_first = false;
                output = img_now.clone();
                return 0;
            }

            //计算从上一帧到当前帧的仿射变换
            get_warpaffine(loc_now);

            // cpu -> gpu
            checkRuntime(cudaMemcpyAsync(affine_now2pre_device, affine_now2pre_host, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
            checkRuntime(cudaMemcpyAsync(img_now_device, img_now.data, w * h * sizeof(uchar3), cudaMemcpyHostToDevice, stream));

            //图像变换融合
            dim3 block_size(32, 32, 1); // blocksize最大是1024
            dim3 grid_size((w + 31) / 32, (h + 31) / 32, 1);
            checkRuntime(cudaMemset(img_fusion_device, 0, w * h * sizeof(uchar3)));
            process_img_kernel<<<grid_size, block_size, 0, stream>>>((uchar *)img_pre_device, (uchar *)img_fusion_device, (uchar *)img_now_device, (uchar *)weight_device, w, h, affine_now2pre_device, rowImFac_device, colImFac_device);

            //输出结果gpu -> cpu
            checkRuntime(cudaMemcpyAsync(output.data, img_now_device, w * h * sizeof(uchar3), cudaMemcpyDeviceToHost, stream));

            //当前帧融合结果用于下一帧融合
            checkRuntime(cudaMemcpyAsync(img_pre_device, img_fusion_device, w * h * sizeof(uchar3), cudaMemcpyDeviceToDevice, stream));
            loc_data_pre = loc_now;

            checkRuntime(cudaStreamSynchronize(stream));

            return 0;
        }

        void TransparentChassis::get_result(cv::Mat &out)
        {
            out = output.clone();
        }

        void TransparentChassis::get_warpaffine(const LocData &loc_now)
        {
            //前一帧的car->global
            double x_pre = loc_data_pre.x;
            double y_pre = loc_data_pre.y;
            double theta_pre = loc_data_pre.yaw;
            cv::Mat RT_c2g_pre = cv::Mat::eye(4, 4, CV_64F);
            RT_c2g_pre.at<double>(0, 0) = cos(theta_pre);
            RT_c2g_pre.at<double>(0, 1) = -sin(theta_pre);
            RT_c2g_pre.at<double>(0, 2) = 0;
            RT_c2g_pre.at<double>(0, 3) = x_pre;
            RT_c2g_pre.at<double>(1, 0) = sin(theta_pre);
            RT_c2g_pre.at<double>(1, 1) = cos(theta_pre);
            RT_c2g_pre.at<double>(1, 2) = 0;
            RT_c2g_pre.at<double>(1, 3) = y_pre;
            //当前帧的car->global
            double x_now = loc_now.x;
            double y_now = loc_now.y;
            double theta_now = loc_now.yaw;
            cv::Mat RT_c2g_now = cv::Mat::eye(4, 4, CV_64F);
            RT_c2g_now.at<double>(0, 0) = cos(theta_now);
            RT_c2g_now.at<double>(0, 1) = -sin(theta_now);
            RT_c2g_now.at<double>(0, 2) = 0;
            RT_c2g_now.at<double>(0, 3) = x_now;
            RT_c2g_now.at<double>(1, 0) = sin(theta_now);
            RT_c2g_now.at<double>(1, 1) = cos(theta_now);
            RT_c2g_now.at<double>(1, 2) = 0;
            RT_c2g_now.at<double>(1, 3) = y_now;
            // img->car
            cv::Mat RT_i2c = cv::Mat::eye(4, 4, CV_64F);
            RT_i2c.at<double>(0, 0) = 0;
            RT_i2c.at<double>(0, 1) = -1 * bev_ratio;
            RT_i2c.at<double>(0, 2) = 0;
            RT_i2c.at<double>(0, 3) = (bev_center_y - 12) * bev_ratio; // Y
            RT_i2c.at<double>(1, 0) = -1 * bev_ratio;
            RT_i2c.at<double>(1, 1) = 0;
            RT_i2c.at<double>(1, 2) = 0;
            RT_i2c.at<double>(1, 3) = bev_center_x * bev_ratio; // X
            RT_i2c.at<double>(2, 0) = 0;
            RT_i2c.at<double>(2, 1) = 0;
            RT_i2c.at<double>(2, 2) = -1 * bev_ratio;
            RT_i2c.at<double>(2, 3) = 0;
            //上一帧图到当前图的变换
            cv::Mat RT_pre2now = RT_i2c.inv() * RT_c2g_now.inv() * RT_c2g_pre * RT_i2c;
            cv::Mat RT_now2pre = RT_pre2now.inv();
            //计算仿射变换矩阵
            affine_now2pre_host[0] = float(RT_now2pre.at<double>(0, 0));
            affine_now2pre_host[1] = float(RT_now2pre.at<double>(0, 1));
            affine_now2pre_host[2] = float(RT_now2pre.at<double>(0, 3));
            affine_now2pre_host[3] = float(RT_now2pre.at<double>(1, 0));
            affine_now2pre_host[4] = float(RT_now2pre.at<double>(1, 1));
            affine_now2pre_host[5] = float(RT_now2pre.at<double>(1, 3));
        }

        void TransparentChassis::destroy()
        {
            checkRuntime(cudaStreamDestroy(stream));
            checkRuntime(cudaFree(weight_device));
            checkRuntime(cudaFree(img_pre_device));
            checkRuntime(cudaFree(img_fusion_device));
            checkRuntime(cudaFree(img_now_device));
            delete[] affine_now2pre_host;
            affine_now2pre_host = nullptr;
            checkRuntime(cudaFree(affine_now2pre_device));
            checkRuntime(cudaFree(rowImFac_device));
            checkRuntime(cudaFree(colImFac_device));
        }
    }
}