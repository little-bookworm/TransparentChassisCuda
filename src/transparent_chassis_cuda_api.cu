#include "transparent_chassis_cuda_api.cuh"

namespace ParkingPerception
{
namespace TransparentChassisCuda
{
    TransparentChassis *CreateTransparentChassis(std::string config_file)
    {
        return new TransparentChassis(config_file);
    }
} // namespace TransparentChassisCuda
} // namespace ParkingPerception