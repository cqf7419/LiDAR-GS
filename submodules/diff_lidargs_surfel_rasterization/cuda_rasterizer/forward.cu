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
namespace cg = cooperative_groups;
__device__ __constant__ float pi = 3.14159265358979323846f;
__device__ __constant__ float Ray_Divergence_Angle = 0.006; 
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
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

__device__ float2 cpmpute_pix_f( 
	float3 point,
	int W,int H, 
	const float* beam_inclinations
){
	float beta = pi - atan2(point.y, point.x);
	float p_c = beta / (2 * pi / float(W)); 
	float alpha = atan2(point.z, sqrt(point.x*point.x + point.y*point.y)); 
	int p_r_int = find_closest_label(beam_inclinations, alpha, H);
	if(p_r_int>=H) printf("[error] cpmpute_pix_f p_r_int > H \n");
	float before = 0;
	float after = 0;
	float p_r = 0;
	if (p_r_int>0){
		before = beam_inclinations[p_r_int - 1]; // 对于升序的beam rad来说 通常只是较小的值
		after = beam_inclinations[p_r_int];
		p_r = p_r_int - 1 + (alpha - before)/(after - before); // 保留小数位  / Ray_Divergence_Angle
	}
	else{ // p_r_int==0
		before = beam_inclinations[p_r_int];
		after = beam_inclinations[p_r_int+1];
		p_r = p_r_int + 1 + (alpha - after)/(after - before); // 保留小数位
	}
	p_r = float(H)-p_r-1;
	float2 pix = {p_c,p_r};
	return pix;
}
__device__ bool cpmpute_pix( 
	float3 point,
	int W,int H, 
	float2& pix,
	const float* beam_inclinations
){
	float beta = pi - atan2(point.y, point.x);
	float p_c = beta / (2 * pi / float(W)); 
	float alpha = atan2(point.z, sqrt(point.x*point.x + point.y*point.y)); 
	int p_r_int = find_closest_label(beam_inclinations, alpha, H);
	if(p_r_int>=H) printf("[error] cpmpute_pix_f p_r_int > H \n");
	float before = 0;
	float after = 0;
	float p_r = 0;
	if (p_r_int>0){
		before = beam_inclinations[p_r_int - 1]; // 对于升序的beam rad来说 通常只是较小的值
		after = beam_inclinations[p_r_int];
		p_r = p_r_int - 1 + (alpha - before)/(after - before); // 保留小数位  / Ray_Divergence_Angle
		if( alpha > (after + Ray_Divergence_Angle)) return false; 
	}
	else{ // p_r_int==0
		before = beam_inclinations[p_r_int];
		after = beam_inclinations[p_r_int+1];
		p_r = p_r_int + 1 + (alpha - after)/(after - before); // 保留小数位
		if( alpha < (before - Ray_Divergence_Angle) ) return false;
	}
	p_r = float(H)-p_r-1;
	pix = {p_c,p_r};
	return true;
}
// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb_cylinder( // 用最长轴近似半径
	glm::mat3 T, 
	float cutoff, 
	// float2& point_image,
	float2 & extent,
	int W,int H, 
	float cylindrical_pix_x, float cylindrical_pix_y,
	const float* beam_inclinations
) {
	float3 T0 = {T[0][0]*cutoff, T[1][0]*cutoff, T[2][0]*cutoff}; 
	float3 T1 = {T[0][1]*cutoff, T[1][1]*cutoff, T[2][1]*cutoff}; // view空间下gs的两条轴（基向量）
	float3 T3 = {T[0][2], T[1][2], T[2][2]}; // view空间下高斯中心 

	float3 La = {T0.x+T3.x, T0.y+T3.y, T0.z+T3.z}; // L0+p_view
	float3 Lb = {T1.x+T3.x, T1.y+T3.y, T1.z+T3.z}; // L1+p_view
	float3 La2 = {-T0.x+T3.x, -T0.y+T3.y, -T0.z+T3.z};
	float3 Lb2 = {-T1.x+T3.x, -T1.y+T3.y, -T1.z+T3.z};

	float2 La_pix = cpmpute_pix_f(La,W,H,beam_inclinations);
	float2 La_pix2 = cpmpute_pix_f(La2,W,H,beam_inclinations);
	float2 La_pix_final = {
		max(fabs(La_pix.x - cylindrical_pix_x), fabs(La_pix2.x - cylindrical_pix_x)),
		max(fabs(La_pix.y - cylindrical_pix_y), fabs(La_pix2.y - cylindrical_pix_y))
	};

	float2 Lb_pix = cpmpute_pix_f(Lb,W,H,beam_inclinations);
	float2 Lb_pix2 = cpmpute_pix_f(Lb2,W,H,beam_inclinations);
	float2 Lb_pix_final = {
		max(fabs(Lb_pix.x - cylindrical_pix_x), fabs(Lb_pix2.x - cylindrical_pix_x)),
		max(fabs(Lb_pix.y - cylindrical_pix_y), fabs(Lb_pix2.y - cylindrical_pix_y))
	};

	extent = {
		ceil(max(max(La_pix_final.x, Lb_pix_final.x),1.0f)),
		ceil(max(max(La_pix_final.y, Lb_pix_final.y),1.0f)),
	}; // 低通滤波 防止退化成线
	return true;

}

template<int C>
__global__ void preprocessCUDA_cylinder(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
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
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
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
	radii_xy[2*idx] = 0;
	radii_xy[2*idx+1] = 0;
	tiles_touched[idx] = 0;
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Perform near culling, quit if outside.
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float dist = sqrt((p_view.x)*(p_view.x) + (p_view.y)*(p_view.y) + (p_view.z)*(p_view.z));
	if (dist >= far || dist <= near) // 大于雷达可见范围   
		return;

	// Compute transformation matrix
	float2 point_image;
	bool ok_pix = cpmpute_pix(p_view, W, H, point_image, beam_inclinations);
	if(!ok_pix) return;

	glm::mat3 T;
	float3 normal;
	glm::mat3 R = quat_to_rotmat(rotations[idx]); // note. 这个R在里面转过置 
	glm::mat3 S = scale_to_mat(scales[idx], scale_modifier);
	glm::mat3 L = R * S;

	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix); // 2dgs 法线量为s为0的轴方向

	glm::mat3x4 world2view = glm::mat3x4(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
		viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
		viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14]);
	
	// L = world2view*L; 
	// // 3列4行 列优先存储
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);
	
	T = glm::transpose(splat2world)*world2view;

	float3 *T_ptr = (float3*)transMats;
	T_ptr[idx * 3 + 0] = {T[0][0], T[1][0], T[2][0]};
	T_ptr[idx * 3 + 1] = {T[0][1], T[1][1], T[2][1]};
	T_ptr[idx * 3 + 2] = {T[0][2], T[1][2], T[2][2]};

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal); 
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal; 
#endif

	float cutoff = 3.0f; 
	float2 extent;
	bool ok = compute_aabb_cylinder(T, cutoff, extent, W, H, point_image.x, point_image.y, beam_inclinations);
	if (!ok) return;

	uint2 rect_min, rect_max;
	getRect_lidar(point_image,  int(extent.x), int(extent.y), rect_min, rect_max, grid);
	if(rect_max.y==0) printf("py is %f, max_radius_y is %d, grid.y is %d",point_image.y, int(extent.y), grid.y);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0){
		printf("rec zero: %f, %f when point_image is %f, %f and rect_y is %d, %d\n", extent.x, extent.y, point_image.x, point_image.y, rect_max.y, rect_min.y);
		return;
	}

	depths[idx] = dist;
	radii[idx] = int(max(extent.x, extent.y));
	radii_xy[2*idx] = int(extent.x);
	radii_xy[2*idx+1] = int(extent.y);
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ beam_inclinations,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	float* __restrict__ pixels)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j]; // 
			float real_depth; 
			float rho2d;
			float rho3d;
			float rho_r;

			float beta_pix = -(pixf.x - float(W) / 2.0) / float(W) * 2.0 * pi;
			if( (H-1-pix.y)<0 ||(H-1-pix.y)>31) printf("forward render / H-1-pix.y shoud't ");
			float alpha_pix = beam_inclinations[H-1-pix.y];
			rho_r = sqrtf(Tw.x * Tw.x + Tw.y * Tw.y + Tw.z * Tw.z);
			if(rho_r == 0) printf("forward render / rho_r should't be zero\n");

			// 真值方法
			float3 p = {
				cos(alpha_pix) * cos(beta_pix),
				cos(alpha_pix) * sin(beta_pix),
				sin(alpha_pix),
			};
			float L2_normal = 1.0; //sqrtf(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]); // normal本身是单位向量
			float L2_Tw = rho_r;
			float L2_p = 1.0; //sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
			float cos_phi1 = (Tw.x*normal[0] + Tw.y*normal[1] + Tw.z*normal[2])/(L2_Tw * L2_normal);// view空间下原点指向gs中心的向量与gs法向量的夹角
			// if(cos_phi1<0) printf("");
			float lambda = L2_Tw * cos_phi1; // view空间下与相机原点到gs平面垂直的距离 （点到面的距离）
			float cos_phi2 = (p.x*normal[0] + p.y*normal[1] + p.z*normal[2])/(L2_p*L2_normal);
			// if(cos_phi2<0) cos_phi2 = -1.0 * cos_phi2;
			if(cos_phi2 == 0) continue; // 平行没有交点不计算
			float lambda2 = lambda / cos_phi2; // ray与gs平面的交点到相机原点的距离
			// if(lambda2<=0) continue; // 交点位于z负区域 不计算
			// float alpha_pix = atan2f(p.y,1); // ray与世界坐标系xz平面的夹角
			real_depth = lambda2; //* cos(alpha_pix);
			float3 real_p = {real_depth*p.x, real_depth*p.y, real_depth*p.z}; // ray与gs平面的交点
			float3 dp = real_p - Tw; // view空间下交点与gs中心的相对位置向量
			float Tu_Tu = Tu.x*Tu.x + Tu.y*Tu.y + Tu.z*Tu.z;
			float Tv_Tv = Tv.x*Tv.x + Tv.y*Tv.y + Tv.z*Tv.z; 
			float dp_Tu = dp.x*Tu.x + dp.y*Tu.y + dp.z*Tu.z;
			float dp_Tv = dp.x*Tv.x + dp.y*Tv.y + dp.z*Tv.z;
			float2 s = {dp_Tu/Tu_Tu, dp_Tv/Tv_Tv};
			rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			rho2d = FilterInvSquare * (40*d.x * d.x + 100*d.y * d.y);  // gs投影到像素平面计算的相对位置 // 这一步也是为了防止gs退化成点或者线

			// compute intersection and depth
			float rho = (real_depth>0)? min(rho3d, rho2d) : rho2d;
			float depth = (rho3d <= rho2d && real_depth>0) ? real_depth : rho_r;  // 点(（s.x, s.y, 1）转乘Tw	 	gs坐标系下的点乘T转到img space 但由于只用z轴 所以只乘第三维	
			if (depth < near_n) continue;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			// atomicAdd(&(pixels[collected_id[j]]), 1.0f);
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

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void filter_preprocessCUDA(int P, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float* transMats,
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
	radii_xy[2*idx+0] = 0;
	radii_xy[2*idx+1] = 0;
	// tiles_touched[idx] = 0;
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Perform near culling, quit if outside.
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float dist = sqrt((p_view.x)*(p_view.x) + (p_view.y)*(p_view.y) + (p_view.z)*(p_view.z));
	if (dist >= far || dist <= near) // 大于雷达可见范围   
		return;
	
	// Compute transformation matrix
	float2 point_image;
	bool ok_pix = cpmpute_pix(p_view, W, H, point_image, beam_inclinations);
	if(!ok_pix) return;

	glm::mat3 T;
	glm::mat3 R = quat_to_rotmat(rotations[idx]); // note. 这个R在里面转过置 
	glm::mat3 S = scale_to_mat(scales[idx], scale_modifier);
	glm::mat3 L = R * S;

	// normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix); // 2dgs 法线量为s为0的轴方向

	glm::mat3x4 world2view = glm::mat3x4(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
		viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
		viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14]); 
	
	// // 3列4行 列优先存储
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);
	
	T = glm::transpose(splat2world)*world2view;

	float3 *T_ptr = (float3*)transMats;
	T_ptr[idx * 3 + 0] = {T[0][0], T[1][0], T[2][0]};
	T_ptr[idx * 3 + 1] = {T[0][1], T[1][1], T[2][1]};
	T_ptr[idx * 3 + 2] = {T[0][2], T[1][2], T[2][2]};

	float cutoff = 3.0f;
	float2 extent;
	bool ok = compute_aabb_cylinder(T, cutoff, extent, W, H, point_image.x, point_image.y, beam_inclinations);
	if (!ok) return;

	uint2 rect_min, rect_max;
	getRect_lidar(point_image, int(extent.x), int(extent.y), rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	radii[idx] = int(max(extent.x, extent.y));
	radii_xy[2*idx+0] = int(extent.x);
	radii_xy[2*idx+1] = int(extent.y);
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* beam_inclinations,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	float* pixels)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		beam_inclinations,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others,
		pixels);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("forward render Error %s \n",cudaGetErrorString(err));
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const int far,
	const int near)
{
	// preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
	cudaDeviceSynchronize();
	cudaError_t err1 = cudaGetLastError();
	if(err1 != cudaSuccess){
		printf("before forward preprocess Error %s \n",cudaGetErrorString(err1));
	}
	preprocessCUDA_cylinder<NUM_CHANNELS> << <(P + 31) / 32, 32 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
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
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		far,
		near
		);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("forward preprocess Error %s \n",cudaGetErrorString(err));
	}
}


void FORWARD::filter_preprocess(int P, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float* beam_inclinations,
	int* radii,
	int* radii_xy,
	float* transMats,
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
			transMat_precomp,
			viewmatrix, 
			projmatrix,
			W, H,
			beam_inclinations,
			radii,
			radii_xy,
			transMats,
			grid,
			prefiltered,
			far,
			near
			);
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess){
			printf("filter Error %s \n",cudaGetErrorString(err));
		}
	}
