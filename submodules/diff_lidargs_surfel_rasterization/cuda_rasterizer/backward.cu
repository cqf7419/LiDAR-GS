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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
__device__ __constant__ float pi = 3.14159265358979323846f;
// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ beam_inclinations,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float * __restrict__ dL_dtransMat_2dtemp,
	float4* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float beta_pix = -(pixf.x - float(W) / 2.0) / float(W) * 2.0 * pi;
			float alpha_pix = beam_inclinations[H-1-pix.y];
			float rho_r = sqrtf(Tw.x * Tw.x + Tw.y * Tw.y + Tw.z * Tw.z); //

			// 真值方法
			float3 p = {
				cos(alpha_pix) * cos(beta_pix),
				cos(alpha_pix) * sin(beta_pix),
				sin(alpha_pix),
			};
			float L2_normal = 1.0; //sqrtf(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
			float L2_Tw = rho_r;
			float L2_p = 1.0;//sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
			float cos_phi1 = (Tw.x*normal[0] + Tw.y*normal[1] + Tw.z*normal[2])/(L2_Tw * L2_normal);// view空间下原点指向gs中心的向量与gs法向量的夹角
			float lambda = L2_Tw * cos_phi1; // view空间下与相机原点到gs平面垂直的距离 （点到面的距离）
			float cos_phi2 = (p.x*normal[0] + p.y*normal[1] + p.z*normal[2])/(L2_p*L2_normal);
			if(cos_phi2 == 0) continue; // 平行没有交点不计算
			float lambda2 = lambda / cos_phi2; // ray与gs平面的交点到相机原点的距离
			// if(lambda2<=0) continue; // 交点位于z负区域 不计算 或者直接用rho2d 这样这部分的normal还会继续优化
			// float alpha_pix = atan2f(p.y,1); // [-pi/2, pi/2]
 			float real_depth = lambda2 ;//* cos(alpha_pix);
			float3 real_p = {real_depth*p.x, real_depth*p.y, real_depth*p.z}; // ray与gs平面的交点
			float3 dp = real_p - Tw; // view空间下交点与gs中心的相对位置向量
			float Tu_Tu = Tu.x*Tu.x + Tu.y*Tu.y + Tu.z*Tu.z;
			float Tv_Tv = Tv.x*Tv.x + Tv.y*Tv.y + Tv.z*Tv.z; 
			float dp_Tu = dp.x*Tu.x + dp.y*Tu.y + dp.z*Tu.z;
			float dp_Tv = dp.x*Tv.x + dp.y*Tv.y + dp.z*Tv.z;
			float2 s = {dp_Tu/Tu_Tu, dp_Tv/Tv_Tv};
			float rho3d = (s.x * s.x + s.y * s.y); 

			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (40*d.x * d.x + 100*d.y * d.y);  // gs投影到像素平面计算的相对位置 // 这一步也是为了防止gs退化成点或者线

			// compute intersection and depth
			float rho = (real_depth>0)? min(rho3d, rho2d) : rho2d;
			float c_d = (rho3d <= rho2d && real_depth>0) ? real_depth : rho_r;
			if (c_d < near_n) continue;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				if(ch==0)
					dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth;  // 此处的z表示深度 规范空间下
#endif
			float beta_temp = pi - atan2(Tw.y, Tw.x);
			float alpha_temp = atan2(Tw.z, sqrt(Tw.x*Tw.x + Tw.y*Tw.y)); 
			float grad_alpha = 0;
			grad_alpha = fabs(beam_inclinations[H-1] - beam_inclinations[0]) / (float(H)-1);

			if (rho3d <= rho2d && real_depth>0) {
				// // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				float dL_dD = dL_dz;
				// D = lambda2 * cos(alpha_pix)
				// float dL_dlambda2 = cos(alpha_pix) * dL_dD;
				float dD_dlambda2 = 1.0; //cos(alpha_pix);
				// lambda2 = lambda / cos_phi2 = L2_Tw * cos_phi1 / cos_phi2 
				//         = (Tw.x*normal[0] + Tw.y*normal[1] + Tw.z*normal[2]) / cos_phi2
				//         = (Tw.x*normal[0] + Tw.y*normal[1] + Tw.z*normal[2]) * L2_p / (p.x*normal[0] + p.y*normal[1] + p.z*normal[2])
				//         = (Tw.x*normal[0] + Tw.y*normal[1] + Tw.z*normal[2]) * sqrtf(p.x*p.x + p.y*p.y + p.z*p.z) / (p.x*normal[0] + p.y*normal[1] + p.z*normal[2]);

				float sum_p_normal = p.x*normal[0] + p.y*normal[1] + p.z*normal[2];
				float sum_Tw_normal = Tw.x*normal[0] + Tw.y*normal[1] + Tw.z*normal[2];
				float3 dlambda2_dTw = {
					normal[0] * L2_p / sum_p_normal, 
					normal[1] * L2_p / sum_p_normal,
					normal[2] * L2_p / sum_p_normal 
				};
				float3 dlambda2_dnormal = {
					(Tw.x * L2_p * sum_p_normal - sum_Tw_normal * L2_p * p.x) / (sum_p_normal*sum_p_normal),
					(Tw.y * L2_p * sum_p_normal - sum_Tw_normal * L2_p * p.y) / (sum_p_normal*sum_p_normal),
					(Tw.z * L2_p * sum_p_normal - sum_Tw_normal * L2_p * p.z) / (sum_p_normal*sum_p_normal)
				};
				// -----------------------------------------------------------------------// 
				float2 dL_ds = {dL_dG * -G * s.x, dL_dG * -G * s.y};
				// s = {dp_Tu/Tu_Tu, dp_Tv/Tv_Tv}
				float3 dsx_dTu = {
					(dp.x * Tu_Tu - dp_Tu * 2 * Tu.x) / (Tu_Tu*Tu_Tu),
					(dp.y * Tu_Tu - dp_Tu * 2 * Tu.y) / (Tu_Tu*Tu_Tu),
					(dp.z * Tu_Tu - dp_Tu * 2 * Tu.z) / (Tu_Tu*Tu_Tu),
				};
				float3 dsx_ddp = { 
					Tu.x / Tu_Tu,
					Tu.y / Tu_Tu,
					Tu.z / Tu_Tu,
				};
				float3 dsy_dTv = {
					(dp.x * Tv_Tv - dp_Tv * 2 * Tv.x) / (Tv_Tv*Tv_Tv),
					(dp.y * Tv_Tv - dp_Tv * 2 * Tv.y) / (Tv_Tv*Tv_Tv),
					(dp.z * Tv_Tv - dp_Tv * 2 * Tv.z) / (Tv_Tv*Tv_Tv),
				};
				float3 dsy_ddp = {
					Tv.x / Tv_Tv,
					Tv.y / Tv_Tv,
					Tv.z / Tv_Tv,
				};

				// dp = real_p - Tw = real_depth * p - Tw = {real_depth * p.x - Tw.x, ...}
				// float3 ddpx_dTw_temp = {-1,0,0};
				// float3 ddpy_dTw_temp = {0,-1,0};
				// float3 ddpz_dTw_temp = {0,0,-1};
				// float ddpx_dlambda2 = p.x * dD_dlambda2; // dlambda2_dTw 和 dlabda2_dnormal 已知
				// float ddpy_dlambda2 = p.y * dD_dlambda2;
				// float ddpz_dlambda2 = p.z * dD_dlambda2;
				float3 ddpx_dTw = {
					p.x * dD_dlambda2 * dlambda2_dTw.x - 1.0,
					p.x * dD_dlambda2 * dlambda2_dTw.y,
					p.x * dD_dlambda2 * dlambda2_dTw.z
				};//ddpx_dlambda2 * dlambda2_dTw -1;
				float3 ddpy_dTw = {
					p.y * dD_dlambda2 * dlambda2_dTw.x,
					p.y * dD_dlambda2 * dlambda2_dTw.y - 1.0,
					p.y * dD_dlambda2 * dlambda2_dTw.z
				};//ddpy_dlambda2 * dlambda2_dTw -1;
				float3 ddpz_dTw = {
					p.z * dD_dlambda2 * dlambda2_dTw.x,
					p.z * dD_dlambda2 * dlambda2_dTw.y,
					p.z * dD_dlambda2 * dlambda2_dTw.z - 1.0
				};//ddpz_dlambda2 * dlambda2_dTw -1;
				float3 ddpx_dnormal = p.x * dD_dlambda2 * dlambda2_dnormal; // ddpx_dlambda2 * dlambda2_dnormal;
				float3 ddpy_dnormal = p.y * dD_dlambda2 * dlambda2_dnormal;
				float3 ddpz_dnormal = p.z * dD_dlambda2 * dlambda2_dnormal;
				
				float3 dsx_dTw = {
					dsx_ddp.x * ddpx_dTw.x + dsx_ddp.y * ddpy_dTw.x + dsx_ddp.z * ddpz_dTw.x,
					dsx_ddp.x * ddpx_dTw.y + dsx_ddp.y * ddpy_dTw.y + dsx_ddp.z * ddpz_dTw.y,
					dsx_ddp.x * ddpx_dTw.z + dsx_ddp.y * ddpy_dTw.z + dsx_ddp.z * ddpz_dTw.z
				};
				float3 dsy_dTw = {
					dsy_ddp.x * ddpx_dTw.x + dsy_ddp.y * ddpy_dTw.x + dsy_ddp.z * ddpz_dTw.x,
					dsy_ddp.x * ddpx_dTw.y + dsy_ddp.y * ddpy_dTw.y + dsy_ddp.z * ddpz_dTw.y,
					dsy_ddp.x * ddpx_dTw.z + dsy_ddp.y * ddpy_dTw.z + dsy_ddp.z * ddpz_dTw.z
				};
				float3 dsx_dnormal = {
					dsx_ddp.x * ddpx_dnormal.x + dsx_ddp.y * ddpy_dnormal.x + dsx_ddp.z * ddpz_dnormal.x,
					dsx_ddp.x * ddpx_dnormal.y + dsx_ddp.y * ddpy_dnormal.y + dsx_ddp.z * ddpz_dnormal.y,
					dsx_ddp.x * ddpx_dnormal.z + dsx_ddp.y * ddpy_dnormal.z + dsx_ddp.z * ddpz_dnormal.z
				};
				float3 dsy_dnormal = {
					dsy_ddp.x * ddpx_dnormal.x + dsy_ddp.y * ddpy_dnormal.x + dsy_ddp.z * ddpz_dnormal.x,
					dsy_ddp.x * ddpx_dnormal.y + dsy_ddp.y * ddpy_dnormal.y + dsy_ddp.z * ddpz_dnormal.y,
					dsy_ddp.x * ddpx_dnormal.z + dsy_ddp.y * ddpy_dnormal.z + dsy_ddp.z * ddpz_dnormal.z
				};

				// final 
				float3 dL_dTu = {
					dL_ds.x * dsx_dTu.x,
					dL_ds.x * dsx_dTu.y, 
					dL_ds.x * dsx_dTu.z
				};
				float3 dL_dTv = {
					dL_ds.y * dsy_dTv.x,
					dL_ds.y * dsy_dTv.y,
					dL_ds.y * dsy_dTv.z,
				};
				float3 dL_dTw = {
					dL_ds.x * dsx_dTw.x + dL_ds.y * dsy_dTw.x + dL_dD * dD_dlambda2 * dlambda2_dTw.x,
					dL_ds.x * dsx_dTw.y + dL_ds.y * dsy_dTw.y + dL_dD * dD_dlambda2 * dlambda2_dTw.y,
					dL_ds.x * dsx_dTw.z + dL_ds.y * dsy_dTw.z + dL_dD * dD_dlambda2 * dlambda2_dTw.z
				};
				float3 dL_dnormal = {
					dL_ds.x * dsx_dnormal.x + dL_ds.y * dsy_dnormal.x + dL_dD * dD_dlambda2 * dlambda2_dnormal.x,
					dL_ds.x * dsx_dnormal.y + dL_ds.y * dsy_dnormal.y + dL_dD * dD_dlambda2 * dlambda2_dnormal.y,
					dL_ds.x * dsx_dnormal.z + dL_ds.y * dsy_dnormal.z + dL_dD * dD_dlambda2 * dlambda2_dnormal.z
				};
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
				atomicAdd(&dL_dtransMat_2dtemp[global_id * 3 + 0], fabs(dL_dTw.x));
				atomicAdd(&dL_dtransMat_2dtemp[global_id * 3 + 1], fabs(dL_dTw.y));
				atomicAdd(&dL_dtransMat_2dtemp[global_id * 3 + 2], fabs(dL_dTw.z)); // 每个gs在覆盖的pix上三各方向的梯度绝对值
				atomicAdd((&dL_dnormal3D[global_id * 3 + 0]), dL_dnormal.x);
				atomicAdd((&dL_dnormal3D[global_id * 3 + 1]), dL_dnormal.y);
				atomicAdd((&dL_dnormal3D[global_id * 3 + 2]), dL_dnormal.z);
			// float beta_pix = -(pixf.x - float(W) / 2.0) / float(W) * 2.0 * pi
			// float alpha_pix = beam_inclinations[H-1-pix.y]
			// float rho_r = sqrtf(Tw.x * Tw.x + Tw.y * Tw.y + Tw.z * Tw.z);

			// // 真值方法
			// float3 p = {
			// 	np.cos(alpha_pix) * np.cos(beta_pix),
			// 	np.cos(alpha_pix) * np.sin(beta_pix),
			// 	np.sin(alpha_pix),
			// };
				float dL_dmean2dx = fabs(dL_dTw.x*sin(beta_temp)*cos(alpha_temp)/float(W)*2.0*pi) + \
									fabs(dL_dTw.y*cos(beta_temp)*cos(alpha_temp)/float(W)*2.0*pi);
				dL_dmean2dx = dL_dmean2dx * rho_r * 0.5 * float(W);
				float dL_dmean2dy = fabs(dL_dTw.x*sin(alpha_temp)*cos(beta_temp)*grad_alpha) + \
									fabs(dL_dTw.y*sin(alpha_temp)*sin(beta_temp)*grad_alpha) + \
									fabs(dL_dTw.z*cos(alpha_temp)*grad_alpha);
				dL_dmean2dy = dL_dmean2dy * rho_r * 0.5 * float(H);
				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2dx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2dy); // not scaled
				atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dmean2dx)); 
				atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dmean2dy)); 
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * 40 * d.x;
				const float dG_ddely = -G * FilterInvSquare * 100 * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * 0.5 * W); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * 0.5 * H); // not scaled
				atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * 0.5 * W)); 
				atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * 0.5 * H)); 
				// float beta = pi - atan2(point.y, point.x);
				// float p_c = beta / (2 * pi / float(W)); 
				// float alpha = atan2(point.z, sqrt(point.x*point.x + point.y*point.y)); 
				// int p_r_int = find_closest_label(beam_inclinations, alpha, H);
				float rho_xy2 = sqrt(Tw.x*Tw.x + Tw.y*Tw.y);
				float ddelx_dpx = float(W)/(2*pi)*Tw.y/(rho_xy2*rho_xy2);
				float ddelx_dpy = -1.0*float(W)/(2*pi)*Tw.x/(rho_xy2*rho_xy2);
				float ddely_dpx = grad_alpha*(-1.0)*Tw.z*Tw.x/(rho_r*rho_r*rho_xy2);
				float ddely_dpy = grad_alpha*(-1.0)*Tw.z*Tw.y/(rho_r*rho_r*rho_xy2);
				float ddely_dpz =  grad_alpha*rho_xy2/(rho_r*rho_r);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dz * (Tw.x/rho_r) + dL_dG * (dG_ddelx * ddelx_dpx + dG_ddely * ddely_dpx));  
				atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dz * (Tw.y/rho_r) + dL_dG * (dG_ddelx * ddelx_dpy + dG_ddely * ddely_dpy)); 
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dz * (Tw.z/rho_r) + dL_dG * (dG_ddely * ddely_dpz));
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

__device__ void compute_cylinder_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H,
	const float3* dL_dnormals,
	float4* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float* dL_dtransMat_2dtemp,
	float* depth)
{
	// Compute transformation matrix
	float3 p_orig = p_origs[idx];
	glm::vec4 rot = rots[idx];
	glm::vec2 scale = scales[idx];
	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, 1.0f);
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);

	// X0,Y0,Z0 
	float rho_r = sqrtf(p_view.x * p_view.x + p_view.z * p_view.z)+1e-16;

	glm::mat3 L = R * S;
	float3 normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

	glm::mat3x4 world2view = glm::mat3x4(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
		viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
		viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14]); 

	glm::mat3x4 M = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat3 T = glm::transpose(M) * world2view;

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+3], dL_dTs[idx*9+6],
		dL_dTs[idx*9+1], dL_dTs[idx*9+4], dL_dTs[idx*9+7],
		dL_dTs[idx*9+2], dL_dTs[idx*9+5], dL_dTs[idx*9+8]
	);
	float4 dL_dmean2D = dL_dmean2Ds[idx];

	
	if (Ts_precomp != nullptr) {
		printf("Ts_precomp error, to check!!");
		return;
	};

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = world2view * glm::transpose(dL_dT); 

	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	depth[idx] = sqrtf(p_view.x * p_view.x + p_view.z * p_view.z); 
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	int W, int H,
	const float* beam_inclinations,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	float* dL_dtransMat_2dtemp,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float4* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float* gs_depth)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float * Ts_precomp = (scales) ? nullptr : transMats;

	// compute_transmat_aabb(
	compute_cylinder_transmat_aabb(

		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H,
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots,
		dL_dtransMat_2dtemp,
		gs_depth
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	int W, int H,
	const float* beam_inclinations,
	const glm::vec3* campos, 
	float4* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dtransMat_2dtemp,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float* depth)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		W, H,
		beam_inclinations,
		campos,	
		dL_dtransMats,
		dL_dtransMat_2dtemp,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		depth
	);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("backward preprocess Error %s \n",cudaGetErrorString(err));
	}
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* beam_inclinations,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float* dL_dtransMat,
	float* dL_dtransMat_2dtemp,
	float4* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		beam_inclinations,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dtransMat_2dtemp,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors
		);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("backward render Error %s \n",cudaGetErrorString(err));
	}
}
