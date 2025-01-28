/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "stdio.h"
#include "config.h"
// #include "helper_cuda.h"
namespace cg = cooperative_groups;

__device__ __constant__ float pi = 3.14159265358979323846f;
__device__ __constant__ float Ray_Divergence_Angle = 0.002; // 
__device__ __constant__ float MAX_DEPTH = 80;
__device__ __constant__ float NEAR_DEPTH = 0;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.  // 截断的值做个标记 虽然不使用该rgb 但在反向传播中还是要传梯度回来
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}
__device__ float3 normalize_f3(float3 v) {
    float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z); // 计算向量的长度
    if (length > 0.0f) {
        v.x /= length;
        v.y /= length;
        v.z /= length;
    }
    return v;
}
// __device__ float compute_dist(const float3 p_orig, const glm::vec3* cam_pos)
// {
// 	glm::vec3 vec_p_orig = {p_orig.x, p_orig.y, p_orig.z};
// 	float dist = sqrt((vec_p_orig[0] - cam_pos[0])*(vec_p_orig[0] - cam_pos[0]) + (vec_p_orig[1] - cam_pos[1])*(vec_p_orig[1] - cam_pos[1]) + (vec_p_orig[2] - cam_pos[2])*(vec_p_orig[2] - cam_pos[2]));
// 	return dist;
// }
__device__ glm::mat3 _proj_2basis(const float3 mean)
{	//选择两个基向量组成投影矩阵
		// gs椭球心到相机原点的方向
	float3 dir = { 
		mean.x,
		mean.y,
		mean.z
	};
	dir = normalize_f3(dir);
	// 构造平面  由2个正交向量基组成
	float3 u1 = {dir.y, -dir.x, 0}; //  0.519737964 -0.853305199
	u1 = normalize_f3(u1); //0.520191427 0.854049694 
	// 叉乘得到第二个基
	float3 u2 = {
		dir.y * u1.z - dir.z * u1.y,
		dir.z * u1.x - dir.x * u1.z,
		dir.x * u1.y - dir.y * u1.x,
	};
	glm::mat3 P = glm::mat3( // 构造投影矩阵
		u1.x, u1.y, u1.z,
		u2.x, u2.y, u2.z,
		0, 0, 0
	);
	return P;
}
__device__ glm::mat3 proj_2basis(const float3 mean, const glm::vec3 campos)
{	//选择两个基向量组成投影矩阵
		// gs椭球心到相机原点的方向
	float3 dir = { 
		mean.x - campos.x, 
		mean.y - campos.y,
		mean.z - campos.z,
	};
	dir = normalize_f3(dir);
	// 构造平面  由2个正交向量基组成
	float3 u1 = {dir.y, -dir.x, 0}; //  0.519737964 -0.853305199
	u1 = normalize_f3(u1); //0.520191427 0.854049694 
	// 叉乘得到第二个基
	float3 u2 = {
		dir.y * u1.z - dir.z * u1.y,
		dir.z * u1.x - dir.x * u1.z,
		dir.x * u1.y - dir.y * u1.x,
	};
	// 没有fx fy 没法直接学球面做雅格比
	glm::mat3 P = glm::mat3( // 构造投影矩阵
		u1.x, u1.y, u1.z,
		u2.x, u2.y, u2.z,
		0, 0, 0
	);
	return P;
}
__device__ float3 computeCov2D_lidar(const glm::mat3 P, const float* cov3D, const float* viewmatrix)
{
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]); 

	glm::mat3 T = W * P;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]
	);
	
	// glm::mat3 cov =  P * Vrk * glm::transpose(P); //glm::transpose(P) * glm::transpose(Vrk) * P
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T; 
	// glm::mat3 cov =  T * Vrk * glm::transpose(T); 

	cov[0][0] += 0.01f;
	cov[1][1] += 0.01f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };  // 舍弃第三行和第三列 得到投影的2d椭圆的协方差矩阵 

}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]); // 正交矩阵

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(  // 对称正定矩阵
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;  //

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };  // 舍弃第三行和第三列 得到投影的2d椭圆的协方差矩阵 
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;
	// S[0][0] = mod * 1.0;
	// S[1][1] = mod * 1.0;
	// S[2][2] = mod * 1.0;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R; // 始终怀疑这里的转置
	// glm::mat3 M = R * S; //
	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;    //
	// glm::mat3 Sigma = M * glm::transpose(M);
	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float3* basis_u1,
	float3* basis_u2,
	float3* sphere_means3D,
	const int far,
	const int near)  // 
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);

	glm::vec3 v_cam_pos = *cam_pos;
	float dist = sqrt((p_view.x)*(p_view.x) + (p_view.y)*(p_view.y) + (p_view.z)*(p_view.z));

	if (dist >= far || dist <= near) 
		return;
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;  
	}

	glm::mat3 Proj = _proj_2basis(p_view);
	float3 cov = computeCov2D_lidar(Proj, cov3D, viewmatrix);  
	cov.x = cov.x / (dist*dist);
	cov.y = cov.y / (dist*dist);
	cov.z = cov.z / (dist*dist);
	float det = (cov.x * cov.z - cov.y * cov.y); 

	if (det == 0.0f) return;
	float det_inv = 1.f / det;  
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max( 1e-9, mid * mid - det));
	float lambda2 = mid - sqrt(max( 1e-9, mid * mid - det));
	float my_radius =  sqrt(max(1e-9,max(lambda1, lambda2))); // ceil(3.f * sqrt(max(lambda1, lambda2)));
 

	float beta = pi - atan2(p_view.y, p_view.x);
	float p_c = beta / (2 * pi / W);

	float alpha = atan2(p_view.z, sqrt(p_view.x*p_view.x + p_view.y*p_view.y)); // [-pi,pi] 弧度
	int p_r_int = find_closest_label(beam_inclinations, alpha, H); // 二分最近邻搜索 beam_inclinations要升序
	float before = 0;
	float after = 0;
	float p_r = 0;
	if (p_r_int>0){
		before = beam_inclinations[p_r_int - 1]; 
		after = beam_inclinations[p_r_int];
		p_r = p_r_int - 1 + (alpha - before)/(after - before); 
		// if((alpha - before)/(after - before)>1.2) return;
		// if((alpha - before)/(after - before)>0.2 && (alpha - before)/(after - before)<0.8) return; 
		if( alpha > (after + Ray_Divergence_Angle*2)) return; 
		// if( alpha > (before + Ray_Divergence_Angle*2) && alpha < (after - Ray_Divergence_Angle*2) ) return;
	}
	else{ // p_r_int==0
		before = beam_inclinations[p_r_int];
		after = beam_inclinations[p_r_int+1];
		p_r = p_r_int + 1 + (alpha - after)/(after - before); 
		// if((alpha - after)/(after - before)<-1.2) return;
		// if((alpha - after)/(after - before)<-0.2 && (alpha - after)/(after - before)>-0.8) return;
		if( alpha < (before - Ray_Divergence_Angle*2) ) return;
		// if( alpha > (before + Ray_Divergence_Angle*2) && alpha < (after - Ray_Divergence_Angle*2) ) return;
	}
	p_r = H-p_r-1;

	int my_radius_y = ceil(3.f * my_radius / tan(abs(after - before))); 
	int my_radius_x = ceil(3.f *my_radius / tan(2*pi/W)); 
	float2 point_image = {p_c, p_r}; // {W,H}

	uint2 rect_min, rect_max;
	getRect_lidar(point_image, my_radius_x, my_radius_y, rect_min, rect_max, grid);  
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx]};

	depths[idx] = dist; 
	radii[idx] = max(my_radius_x, my_radius_y); 
	radii_xy[2*idx+0] = my_radius_x;
	radii_xy[2*idx+1] = my_radius_y;
	points_xy_image[idx] = point_image;

	basis_u1[idx] = {Proj[0][0], Proj[0][1], Proj[0][2]};
	basis_u2[idx] = {Proj[1][0], Proj[1][1], Proj[1][2]};
	sphere_means3D[idx] = {p_view.x/dist, p_view.y/dist, p_view.z/dist};


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); 
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void filter_preprocessCUDA(int P, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float2* points_xy_image,
	float* cov3Ds,
	const dim3 grid,
	bool prefiltered,
	const int far,
	const int near)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	glm::vec3 v_cam_pos = *cam_pos;
	// float dist = sqrt((p_orig.x - v_cam_pos.x)*(p_orig.x - v_cam_pos.x) + (p_orig.y - v_cam_pos.y)*(p_orig.y - v_cam_pos.y) + (p_orig.z - v_cam_pos.z)*(p_orig.z - v_cam_pos.z));
	float dist = sqrt((p_view.x)*(p_view.x) + (p_view.y)*(p_view.y) + (p_view.z)*(p_view.z));

	if (dist >= far || dist <= near)   
		return;
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;  
	}

	glm::mat3 Proj = _proj_2basis(p_view);
	float3 cov = computeCov2D_lidar(Proj, cov3D, viewmatrix);   
	cov.x = cov.x / (dist*dist);
	cov.y = cov.y / (dist*dist);
	cov.z = cov.z / (dist*dist);
	float det = (cov.x * cov.z - cov.y * cov.y); 

	if (det == 0.0f) return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv }; 

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(1e-9, mid * mid - det));
	float lambda2 = mid - sqrt(max(1e-9, mid * mid - det));
	float my_radius = sqrt(max(1e-9,max(lambda1, lambda2))); // ceil(3.f * sqrt(max(lambda1, lambda2))); // 最长轴的三倍作为投影椭圆半径 

	float beta = pi - atan2(p_view.y, p_view.x);
	float p_c = beta / (2 * pi / W); 

	float alpha = atan2(p_view.z, sqrt(max(1e-9,p_view.x*p_view.x + p_view.y*p_view.y))); 
	int p_r_int = find_closest_label(beam_inclinations, alpha, H);
	float before = 0;
	float after = 0;
	float p_r  = 0;
	if (p_r_int>0){
		before = beam_inclinations[p_r_int - 1]; 
		after = beam_inclinations[p_r_int];
		p_r = p_r_int - 1 + (alpha - before)/(after - before); 
		// if((alpha - before)/(after - before)>1.2) return;
		// if((alpha - before)/(after - before)>0.2 && (alpha - before)/(after - before)<0.8) return; 
		if( alpha > (after + Ray_Divergence_Angle*2) ) return; 
		// if( alpha > (before + Ray_Divergence_Angle) && alpha < (after - Ray_Divergence_Angle) ) return;
	}
	else{ 
		before = beam_inclinations[p_r_int];
		after = beam_inclinations[p_r_int+1];
		p_r = p_r_int + 1 + (alpha - after)/(after - before); 
		// if((alpha - after)/(after - before)<-1.2) return;
		// if((alpha - after)/(after - before)<-0.2 && (alpha - after)/(after - before)>-0.8) return;
		if( alpha < (before - Ray_Divergence_Angle*2) ) return;
		// if( alpha > (before + Ray_Divergence_Angle) && alpha < (after - Ray_Divergence_Angle) ) return;
	}
	p_r = H-p_r-1;

	int my_radius_y = ceil(3.f *  my_radius /  tan(abs(after - before)));
	int my_radius_x = ceil(3.f * my_radius /  tan(2*pi/W)); 

	float2 point_image = {p_c, p_r}; // {W,H}
	
	uint2 rect_min, rect_max;
	getRect_lidar(point_image, my_radius_x, my_radius_y, rect_min, rect_max, grid); 
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0){
		return;
	}

	radii[idx] = max(my_radius_x, my_radius_y);  // 高斯半径
	radii_xy[2*idx+0] = my_radius_x;
	radii_xy[2*idx+1] = my_radius_y;
	points_xy_image[idx] = point_image;
	
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	const float3* __restrict__ basis_u1,
	const float3* __restrict__ basis_u2,
	const float3* __restrict__ sphere_means3D,
	const float* __restrict__ beam_inclinations,
	float* __restrict__ out_depth,
	float* __restrict__ out_occ)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();  
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; 
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; 
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };  
	uint32_t pix_id = W * pix.y + pix.x;   
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W&& pix.y < H;   
	bool done = !inside;  

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; 
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); 
	int toDo = range.y - range.x; 

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];  
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float3 collected_basis_u1[BLOCK_SIZE];
	__shared__ float3 collected_basis_u2[BLOCK_SIZE];
	__shared__ float3 collected_sphere_means3D[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;  
	uint32_t contributor = 0;  
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 }; 
	float D = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) 
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;  

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) // 
		{  
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
			collected_basis_u1[block.thread_rank()] = basis_u1[coll_id];
			collected_basis_u2[block.thread_rank()] = basis_u2[coll_id];
			collected_sphere_means3D[block.thread_rank()] = sphere_means3D[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			float2 d;
			{
				float3 unit_sphere_xyz = collected_sphere_means3D[j];
				float3 u1 = collected_basis_u1[j]; 
				float3 u2 = collected_basis_u2[j];
				float alp = beam_inclinations[H-1-pix.y];
				float beta = -(pixf.x - (float)W / 2.0) / (float)W * 2.0 * pi;
				float3 uint_sphere_pixf = {cos(alp) * cos(beta), cos(alp) * sin(beta), sin(alp)};
				float3 uint_sphere_d = {unit_sphere_xyz.x - uint_sphere_pixf.x, unit_sphere_xyz.y - uint_sphere_pixf.y, unit_sphere_xyz.z - uint_sphere_pixf.z}; // 规范空间下的相对位置
				float u1_u1 = u1.x*u1.x + u1.y*u1.y + u1.z*u1.z;
				float u2_u2 = u2.x*u2.x + u2.y*u2.y + u2.z*u2.z;
				float _d_u1 = uint_sphere_d.x * u1.x + uint_sphere_d.y * u1.y + uint_sphere_d.z * u1.z;
				float _d_u2 = uint_sphere_d.x * u2.x + uint_sphere_d.y * u2.y + uint_sphere_d.z * u2.z;
				d = {_d_u1/u1_u1, _d_u2/u2_u2}; 
			}

			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; 
			if (power > 0.0f)
				continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) 
				continue;
			float test_T = T * (1 - alpha);  
			if (test_T < 0.0001f) 
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++){
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T; // 渲染公式 对应gs论文公式(3) 
				// C[ch] += 0.5 * alpha * T;
			}
			D += collected_depth[j] * alpha * T; // lidar深度
			
			T = test_T; // 透光率更新

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)  
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
		out_occ[pix_id] = 1 - T;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depths,
	const float3* basis_u1,
	const float3* basis_u2,
	const float3* sphere_means3D,
	const float* beam_inclinations,
	float* out_depth,
	float* out_occ)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths,
		basis_u1,
		basis_u2,
		sphere_means3D,
		beam_inclinations,
		out_depth,
		out_occ);

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("render Error %s \n",cudaGetErrorString(err));
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float3* basis_u1,
	float3* basis_u2,
	float3* sphere_means3D,
	const int far,
	const int near)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		beam_inclinations,
		radii,
		radii_xy,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		basis_u1,
		basis_u2,
		sphere_means3D,
		far,
		near
		);

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("preprocess Error %s \n",cudaGetErrorString(err));
	}
	
}


void FORWARD::filter_preprocess(int P, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float2* means2D,
	float* cov3Ds,
	const dim3 grid,
	bool prefiltered,
	const int far,
	const int near)
{


	filter_preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		beam_inclinations,
		radii,
		radii_xy,
		means2D,
		cov3Ds,
		grid,
		prefiltered,
		far,
		near);

}