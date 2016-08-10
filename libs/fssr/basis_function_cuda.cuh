/*
 * This file is part of the Floating Scale Surface Reconstruction software.
 * Written by Simon Fuhrmann.
*/

#ifndef FSSR_BASIS_FUNCTION_CUDA_HEADER
#define FSSR_BASIS_FUNCTION_CUDA_HEADER

#define MATH_SQRT_2PI 2.506628274631000502415765284811045253 // sqrt(2*pi)
#define MATH_2_PI 0.636619772367581343075535053490057448 // 2/pi

#include <cuda_runtime.h>

#include "defines.h"
#include "cuda_types.h"
// #include "stdio.h"




FSSR_NAMESPACE_BEGIN


namespace CUBasisFunction {
    // Transformation stuff
//     __host__ __device__ __forceinline__ CUTypes::Mat3f
//     matrix_rotation_from_axis_angle (float3 axis, float angle);

//     __host__ __device__ __forceinline__ CUTypes::Mat3f
//     rotation_from_normal (float3 normal);

    __host__ __device__ __forceinline__ float3
    transform_position (float3 pos, float3 sample_pos, float3 sample_normal);

    // Weighting stuff
    __host__ __device__ __forceinline__ float
    weighting_function (float sample_scale, float3 pos);

    __host__ __device__ __forceinline__ float
    weighting_function_x (float x);

    __host__ __device__ __forceinline__ float
    weighting_function_yz (float y, float z);

    // Gaussian stuff
    __host__ __device__ __forceinline__ float
    gaussian_normalized (float sigma, float3 pos);

    __host__ __device__ __forceinline__ float
    gaussian_derivative (float sigma, float3 pos);

    __host__ __device__ __forceinline__ float
    gaussian (float sigma, float3 pos);






    // implementation

    __host__ __device__ __forceinline__ float3
    transform_position (float3 pos, float3 sample_pos, float3 sample_normal)
    {
        float rot00, rot01, rot02;
        float rot10, rot11, rot12;
        float rot20, rot21, rot22;

        // ref 1 0 0
        if (fabsf(sample_normal.x - 1.0f) < 0.001f
            || fabsf(sample_normal.y) < 0.001f
            || fabsf(sample_normal.z) < 0.001f)
        {
            rot00 = 1.0f; rot01 = 0.0f; rot02 = 0.0f;
            rot10 = 0.0f; rot11 = 1.0f; rot12 = 0.0f;
            rot20 = 0.0f; rot21 = 0.0f; rot22 = 1.0f;
        }
        // mirror -1 0 0
        else if (fabsf(sample_normal.x + 1.0f) < 0.001f
            || fabsf(sample_normal.y) < 0.001f
            || fabsf(sample_normal.z) < 0.001f)
        {
            /* 180 degree rotation around the z-axis. */
            rot00 = -1.0f; rot01 = 0.0f; rot02 = 0.0f;
            rot10 = 0.0f; rot11 = -1.0f; rot12 = 0.0f;
            rot20 = 0.0f; rot21 = 0.0f; rot22 = 1.0f;
        }
        else {
            float3 axis = make_float3(0.0f, sample_normal.z, -sample_normal.y);
            float n = sqrtf(axis.y * axis.y + axis.z * axis.z);
            axis.y /= n;
            axis.z /= n;

            float cos_alpha = fminf(1.0f, fmaxf(sample_normal.x, -1.0f));
            float angle = acosf(cos_alpha);

            /*
            * http://en.wikipedia.org/wiki/Rotation_matrix
            *     #Rotation_matrix_from_axis_and_angle
            */
            float ca = cosf(angle);
            float sa = sinf(angle);
            float omca = 1.0f - ca;

            rot00 = ca + axis.x * axis.x * omca;
            rot01 = axis.x * axis.y * omca - axis.z * sa;
            rot02 = axis.x * axis.z * omca + axis.y * sa;

            rot10 = axis.y * axis.x * omca + axis.z * sa;
            rot11 = ca + axis.y * axis.y * omca;
            rot12 = axis.y * axis.z * omca - axis.x * sa;

            rot20 = axis.z * axis.x * omca - axis.y * sa;
            rot21 = axis.z * axis.y * omca + axis.x * sa;
            rot22 = ca + axis.z * axis.z * omca;
        }

        float3 t = make_float3(pos.x - sample_pos.x, pos.y - sample_pos.y, pos.z - sample_pos.z);

        float3 new_pos;
        new_pos.x = rot00 * t.x + rot01 * t.y + rot02 * t.z;
        new_pos.y = rot10 * t.x + rot11 * t.y + rot12 * t.z;
        new_pos.z = rot20 * t.x + rot21 * t.y + rot22 * t.z;
        return new_pos;
    }



    __host__ __device__ __forceinline__ float
    weighting_function (float sample_scale, float3 pos)
    {
        return weighting_function_x(pos.x / sample_scale)
            * weighting_function_yz(pos.y / sample_scale, pos.z / sample_scale);
    }

    __host__ __device__ __forceinline__ float
    weighting_function_x (float x)
    {
        if (x <= -3.0f || x >= 3.0f)
            return 0.0f;

        if (x > 0.0f)
        {
            float a_o = 2.0f / 27.0f;
            float b_o = -1.0f / 3.0f;
            float d_o = 1.0f;
            float value = a_o * x * x * x + b_o * x * x + d_o;
            return value;
        }

        float a_i = 1.0f / 9.0f;
        float b_i = 2.0f / 3.0f;
        float c_i = 1.0f;
        float value = a_i * x * x + b_i * x + c_i;
        return value;
    }

    __host__ __device__ __forceinline__ float
    weighting_function_yz (float y, float z)
    {
        if (y * y + z * z > 9.0f)
            return 0.0f;

        float a_o = 2.0f / 27.0f;
        float b_o = -1.0f / 3.0f;
        float d_o = 1.0f;
        float value = a_o * powf(y * y + z * z, 1.5f)
            + b_o * (y * y + z * z) + d_o;
        return value;
    }

    __host__ __device__ __forceinline__ float
    gaussian_normalized (float sigma, float3 pos)
    {
        return gaussian(sigma, pos) / (sigma * MATH_SQRT_2PI);
    }

    __host__ __device__ __forceinline__ float
    gaussian_derivative (float sigma, float3 pos)
    {
        return pos.x * gaussian(sigma, pos) / (sigma * sigma * sigma * sigma * MATH_2_PI);
    }

    __host__ __device__ __forceinline__ float
    gaussian (float sigma, float3 pos)
    {
        return expf(-(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z) / (2.0f * sigma * sigma));
    }
}

FSSR_NAMESPACE_END

#endif // FSSR_BASIS_FUNCTION_CUDA_HEADER
