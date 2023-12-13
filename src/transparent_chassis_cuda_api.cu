#include "transparent_chassis_cuda_api.cuh"

namespace ParkingPerception
{
    namespace TransparentChassisCuda
    {
        std::shared_ptr<TransparentChassis> CreateTransparentChassis(std::string config_file)
        {
            return std::make_shared<TransparentChassis>(config_file);
        }
    } // namespace TransparentChassisCuda
} // namespace ParkingPerception