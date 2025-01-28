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
#include "config.h"
namespace cg = cooperative_groups;
__device__ __constant__ float pi = 3.14159265358979323846f;

__device__ float3 norm_f3(float3 v) {
	if(v.x * v.x + v.y * v.y + v.z * v.z==0) return v;
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); // 计算向量的长度
    if (length > 0.0f) {
        v.x /= length;
        v.y /= length;
        v.z /= length;
    }
    return v;
}

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

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const int width, int height,
	const float3* means,
	const float* beam_inclinations,
	const int* radii,
	const float* cov3Ds,
	const float* view_matrix,
	const glm::vec3* campos,
	const float4* dL_dmean2D,
	const float3* dL_dbasis_u1,
	const float3* dL_dbasis_u2,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };


	float3 d = transformPoint4x3(mean, view_matrix);
	// float3 d = { 
	// 	mean.x - (*campos).x, 
	// 	mean.y - (*campos).y,
	// 	mean.z - (*campos).z,
	// };
	float dist = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
	float3 dir = norm_f3(d);
	float3 uint_sphere_d = {dir.x, dir.y, dir.z};
	// 构造平面（已知法向）  找到平面内的2个正交向量基
	float3 u1 = {dir.y, -dir.x, 0};
	u1 = norm_f3(u1);
	// 叉乘得到第二个基
	float3 u2 = {
		dir.y * u1.z - dir.z * u1.y,
		dir.z * u1.x - dir.x * u1.z,
		dir.x * u1.y - dir.y * u1.x,
	};

	glm::mat3 J = glm::mat3( // 构造投影矩阵
		u1.x, u1.y, u1.z,
		u2.x, u2.y, u2.z,
		0, 0, 0
	);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]
	);
	glm::mat3 T = W * J;

	// glm::mat3 cov2D =  T * Vrk * glm::transpose(T); 
	glm::mat3 cov2D =  glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	float _a = cov2D[0][0] += 0.01f;
	float _b = cov2D[0][1];
	float _c = cov2D[1][1] += 0.01f;
	float a = 1/(dist*dist) * _a;
	float b = 1/(dist*dist) * _b;
	float c = 1/(dist*dist) * _c;
	//---------------------------------------------------------------------------------------

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
	float3 dL_dcov_mean;
	// ---------------------------- 损失相对于协方差矩阵的梯度 ----------------------------------
	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-1*c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z); //见手稿推导第四页
		dL_dc = denom2inv * (-1*a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		float dist4 = dist*dist*dist*dist;
		dL_dcov_mean.x = dL_da * (-2*d.x*_a) / dist4 + dL_db * (-2*d.x*_b) / dist4 + dL_dc * (-2*d.x*_c) / dist4;
		dL_dcov_mean.y = dL_da * (-2*d.y*_a) / dist4 + dL_db * (-2*d.y*_b) / dist4 + dL_dc * (-2*d.y*_c) / dist4;
		dL_dcov_mean.z = dL_da * (-2*d.z*_a) / dist4 + dL_db * (-2*d.z*_b) / dist4 + dL_dc * (-2*d.z*_c) / dist4;

		dL_da = 1/(dist*dist) * dL_da;
		dL_dc = 1/(dist*dist) * dL_dc;
		dL_db = 1/(dist*dist) * dL_db;


		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	//cam： cov2D = transpose(T) * transpose(Vrk) * T;   
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;
	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ01 = W[1][0] * dL_dT00 + W[1][1] * dL_dT01 + W[1][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ10 = W[0][0] * dL_dT10 + W[0][1] * dL_dT11 + W[0][2] * dL_dT12;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	dL_dJ00 = dL_dJ00 + dL_dbasis_u1[idx].x;  // u1.x = T00  以此类推
	dL_dJ01 = dL_dJ01 + dL_dbasis_u1[idx].y;
	dL_dJ02 = dL_dJ02 + dL_dbasis_u1[idx].z;
	dL_dJ10 = dL_dJ10 + dL_dbasis_u2[idx].x;
	dL_dJ11 = dL_dJ11 + dL_dbasis_u2[idx].y;
	dL_dJ12 = dL_dJ12 + dL_dbasis_u2[idx].z;

	// d.x = meanx - cam.x  /  d.y = meany - cam.y  /  d.z = meanz - cam.z
	// dir.x = d.x / sqrt(d.x*d.x + d.y*d.y + d.z*d.z)
	// 这里还是在view坐标系 在整个backward的最后会转世界坐标系
	float d_sum2 = d.x*d.x + d.y*d.y + d.z*d.z ;
	float inv_d_sum32 = 1.0f / (sqrt(d_sum2 * d_sum2 * d_sum2)+ 1e-9);
	float ddirx_ddx = (d_sum2 - d.x*d.x) * inv_d_sum32;//-1*dir.x*dir.x * 0.5*dir.x * (-2)*(d.y*d.y+d.z*d.z)/(d.x*d.x*d.x);
	float ddirx_dmeanx = ddirx_ddx; //ddirx_ddx * ddx_dmeanx + ddirx_ddy * ddy_dmeanx + ddirx_ddz * ddz_dmeanx   其中 ddx_dmeanx = 1 / ddy_dmeanx = 0 / ddz_dmeanx = 0
	float ddirx_ddy = (-d.x*d.y) * inv_d_sum32;//-1*dir.x*dir.x * 0.5*dir.x * 2*d.y/(dir.x*dir.x);
	float ddirx_dmeany = ddirx_ddy;//ddirx_ddx * ddx_dmeany + ddirx_ddy * ddy_dmeany + ddirx_ddz * ddz_dmeany 其中 ddx_dmeany = 0 / ddy_dmeany = 1 / ddz_dmeany = 0
	float ddirx_ddz = (-d.x*d.z) * inv_d_sum32;//-1*dir.x*dir.x * 0.5*dir.x * 2*d.z/(dir.x*dir.x);
	float ddirx_dmeanz = ddirx_ddz;//ddirx_ddx * ddx_dmeanz + ddirx_ddy * ddy_dmeanz + ddirx_ddz * ddz_dmeanz 其中 ddx_dmeanz = 0 / ddy_dmeanz = 0 / ddz_dmeanz = 1
	// dir.y = d.y / sqrt(d.x*d.x + d.y*d.y + d.z*d.z)
	float ddiry_ddx = (-d.x*d.y) * inv_d_sum32;
	float ddiry_dmeanx = ddiry_ddx; //ddiry_ddx * ddx_dmeanx + ddiry_ddy * ddy_dmeanx + ddiry_ddz * ddz_dmeanx   其中 ddx_dmeanx = 1 / ddy_dmeanx = 0 / ddz_dmeanx = 0
	float ddiry_ddy = (d_sum2 - d.y*d.y) * inv_d_sum32;
	float ddiry_dmeany = ddiry_ddy;//ddiry_ddx * ddx_dmeany + ddiry_ddy * ddy_dmeany + ddiry_ddz * ddz_dmeany   其中 ddx_dmeany = 0 / ddy_dmeany = 1 / ddz_dmeany = 0
	float ddiry_ddz = (-d.y*d.z) * inv_d_sum32;//-1*dir.y*dir.y * 0.5*dir.y * 2*d.z/(d.y*d.y);
	float ddiry_dmeanz = ddiry_ddz;//ddiry_ddx * ddx_dmeanz + ddiry_ddy * ddy_dmeanz + ddiry_ddz * ddz_dmeanz   其中 ddx_dmeanz = 0 / ddy_dmeanz = 0 / ddz_dmeanz = 1
	// dir.z = d.z / sqrt(d.x*d.x + d.y*d.y + d.z*d.z)
	float ddirz_ddx = (-d.x*d.z) * inv_d_sum32;
	float ddirz_dmeanx = ddirz_ddx;//ddirz_ddx * ddx_dmeanx + ddirz_ddy * ddy_dmeanx + ddirz_ddz * ddz_dmeanx 其中 ddx_dmeanx = 1 / ddy_dmeanx = 0 / ddz_dmeanx = 0
	float ddirz_ddy = (-d.y*d.z) * inv_d_sum32;
	float ddirz_dmeany = ddirz_ddy;//ddirz_ddx * ddx_dmeany + ddirz_ddy * ddy_dmeany + ddirz_ddz * ddz_dmeany 其中 ddx_dmeany = 0 / ddy_dmeany = 1 / ddz_dmeany = 0
	float ddirz_ddz = (d_sum2 - d.z*d.z) * inv_d_sum32;
	float ddirz_dmeanz = ddirz_ddz;//ddirz_ddx * ddx_dmeanz + ddirz_ddy * ddy_dmeanz + ddirz_ddz * ddz_dmeanz 其中 ddx_dmeanz = 0 / ddy_dmeanz = 0 / ddz_dmeanz = 1

	// J00 = dir.y / sqrt(dir.x*dir.x + dir.y*dir.y)
	float dir_sum2  = dir.x*dir.x + dir.y*dir.y ;
	float inv_dir_sum32 = 1.0f / (sqrt(dir_sum2 * dir_sum2 * dir_sum2)+ 1e-9);
	float dJ00_ddiry = (dir.x*dir.x) * inv_dir_sum32;
	float dJ00_ddirx = (-dir.y*dir.x) * inv_dir_sum32;
	// J01 = -dir.x / sqrt(dir.x*dir.x + dir.y*dir.y)
	float dJ01_ddirx = (-dir.y*dir.y) * inv_dir_sum32;
	float dJ01_ddiry = (dir.x*dir.y) * inv_dir_sum32;
	// J02 = 0
	// J10 = dir.z * dir.x / sqrt(dir.x*dir.x + dir.y*dir.y)
	float dJ10_ddirx = dir.z*dir.y*dir.y*inv_dir_sum32;
	float dJ10_ddiry = -dir.x*dir.y*dir.z*inv_dir_sum32;
	float dJ10_ddirz = dir.x / (sqrt(dir_sum2)+ 1e-9);
	// J11 = dirz * dir.y / sqrt(dir.x*dir.x + dir.y*dir.y);
	float dJ11_ddirx = -dir.x*dir.y*dir.z*inv_dir_sum32;
	float dJ11_ddiry = dir.z*dir.x*dir.x*inv_dir_sum32;
	float dJ11_ddirz = dir.y / (sqrt(dir_sum2)+ 1e-9);
	// J12 = -sqrt(dir.x*dir.x + dir.y*dir.y);
	float dJ12_ddirx = -dir.x / (sqrt(dir_sum2)+ 1e-9);
	float dJ12_ddiry = -dir.y / (sqrt(dir_sum2)+ 1e-9);

	float dJ00_dmeanx = dJ00_ddirx * ddirx_dmeanx + dJ00_ddiry * ddiry_dmeanx;
	float dJ01_dmeanx = dJ01_ddirx * ddirx_dmeanx + dJ01_ddiry * ddiry_dmeanx;
	float dJ10_dmeanx = dJ10_ddirx * ddirx_dmeanx + dJ10_ddiry * ddiry_dmeanx + dJ10_ddirz * ddirz_dmeanx;
	float dJ11_dmeanx = dJ11_ddirx * ddirx_dmeanx + dJ11_ddiry * ddiry_dmeanx + dJ11_ddirz * ddirz_dmeanx;
	float dJ12_dmeanx = dJ12_ddirx * ddirx_dmeanx + dJ12_ddiry * ddiry_dmeanx;
	float dL_dmeanx = dL_dcov_mean.x + dL_dJ00 * dJ00_dmeanx + dL_dJ01 * dJ01_dmeanx + dL_dJ10 * dJ10_dmeanx + dL_dJ11 * dJ11_dmeanx +dL_dJ12 * dJ12_dmeanx;

	float dJ00_dmeany = dJ00_ddirx * ddirx_dmeany + dJ00_ddiry * ddiry_dmeany;
	float dJ01_dmeany = dJ01_ddirx * ddirx_dmeany + dJ01_ddiry * ddiry_dmeany;
	float dJ10_dmeany = dJ10_ddirx * ddirx_dmeany + dJ10_ddiry * ddiry_dmeany + dJ10_ddirz * ddirz_dmeany;
	float dJ11_dmeany = dJ11_ddirx * ddirx_dmeany + dJ11_ddiry * ddiry_dmeany + dJ11_ddirz * ddirz_dmeany;
	float dJ12_dmeany = dJ12_ddirx * ddirx_dmeany + dJ12_ddiry * ddiry_dmeany;
	float dL_dmeany = dL_dcov_mean.y + dL_dJ00 * dJ00_dmeany + dL_dJ01 * dJ01_dmeany + dL_dJ10 * dJ10_dmeany + dL_dJ11 * dJ11_dmeany +dL_dJ12 * dJ12_dmeany;

	float dJ00_dmeanz = dJ00_ddirx * ddirx_dmeanz + dJ00_ddiry * ddiry_dmeanz;
	float dJ01_dmeanz = dJ01_ddirx * ddirx_dmeanz + dJ01_ddiry * ddiry_dmeanz;
	float dJ10_dmeanz = dJ10_ddirx * ddirx_dmeanz + dJ10_ddiry * ddiry_dmeanz + dJ10_ddirz * ddirz_dmeanz;
	float dJ11_dmeanz = dJ11_ddirx * ddirx_dmeanz + dJ11_ddiry * ddiry_dmeanz + dJ11_ddirz * ddirz_dmeanz;
	float dJ12_dmeanz = dJ12_ddirx * ddirx_dmeanz + dJ12_ddiry * ddiry_dmeanz;
	float dL_dmeanz = dL_dcov_mean.z + dL_dJ00 * dJ00_dmeanz + dL_dJ01 * dJ01_dmeanz + dL_dJ10 * dJ10_dmeanz + dL_dJ11 * dJ11_dmeanz +dL_dJ12 * dJ12_dmeanz;

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	float3 dL_dmean = {dL_dmeanx, dL_dmeany, dL_dmeanz};	// 这里还是在view坐标系 在整个backward的最后会转世界坐标系
	dL_dmeans[idx] = dL_dmean;
}
// Backward pass for the conversion of scale and rotation to a  
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S*R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const int width, int height,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* proj,
	const glm::vec3* campos,
	const float* beam_inclinations,
	const float4* dL_dmean2D,
	const float3* dL_dsphere_means3D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepths,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	glm::vec3 v_campos = *campos;
	float3 p_view = transformPoint4x3(m, viewmatrix);
	float3 p_campos = {p_view.x, p_view.y, p_view.z};
	float dist = sqrt(p_campos.x*p_campos.x + p_campos.y*p_campos.y + p_campos.z*p_campos.z);
	if (dist <= 0) 
		return;
	float ddist_dmeanx = p_campos.x / dist; 
	float ddist_dmeany = p_campos.y / dist;
	float ddist_dmeanz = p_campos.z / dist;

	// d.x = meanx - cam.x  /  d.y = meany - cam.y  /  d.z = meanz - cam.z
	// dir.x = d.x / sqrt(d.x*d.x + d.y*d.y + d.z*d.z)
	float p_sum2 = p_campos.x*p_campos.x + p_campos.y*p_campos.y + p_campos.z*p_campos.z ;
	float inv_p_sum32 = 1.0f / sqrt(p_sum2*p_sum2*p_sum2); //dist*dist*dist;
	float dspx_dpx = (p_sum2 - p_campos.x*p_campos.x) * inv_p_sum32;
	float dspx_dmeanx = dspx_dpx; 
	float dspx_dpy = (-p_campos.x*p_campos.y) * inv_p_sum32;
	float dspx_dmeany = dspx_dpy;
	float dspx_dpz = (-p_campos.x*p_campos.z) * inv_p_sum32;
	float dspx_dmeanz = dspx_dpz;

	float dspy_dpx = (-p_campos.x*p_campos.y) * inv_p_sum32;
	float dspy_dmeanx = dspy_dpx; 
	float dspy_dpy = (p_sum2 - p_campos.y*p_campos.y) * inv_p_sum32;
	float dspy_dmeany = dspy_dpy;
	float dspy_dpz = (-p_campos.y*p_campos.z) * inv_p_sum32;
	float dspy_dmeanz = dspy_dpz;

	float dspz_dpx = (-p_campos.x*p_campos.z) * inv_p_sum32;
	float dspz_dmeanx = dspz_dpx;//ddirz_ddx * ddx_dmeanx + ddirz_ddy * ddy_dmeanx + ddirz_ddz * ddz_dmeanx 其中 ddx_dmeanx = 1 / ddy_dmeanx = 0 / ddz_dmeanx = 0
	float dspz_dpy = (-p_campos.y*p_campos.z) * inv_p_sum32;
	float dspz_dmeany = dspz_dpy;//ddirz_ddx * ddx_dmeany + ddirz_ddy * ddy_dmeany + ddirz_ddz * ddz_dmeany 其中 ddx_dmeany = 0 / ddy_dmeany = 1 / ddz_dmeany = 0
	float dspz_dpz = (p_sum2 - p_campos.z*p_campos.z) * inv_p_sum32;
	float dspz_dmeanz = dspz_dpz;//ddirz_ddx * ddx_dmeanz + ddirz_ddy * ddy_dmeanz + ddirz_ddz * ddz_dmeanz 其中 ddx_dmeanz = 0 / ddy_dmeanz = 0 / ddz_dmeanz = 1

	float3 view_dL_dmean;
	view_dL_dmean.x = dL_dmeans[idx].x + dL_dsphere_means3D[idx].x * dspx_dmeanx + dL_dsphere_means3D[idx].y * dspy_dmeanx + dL_dsphere_means3D[idx].z * dspz_dmeanx + dL_ddepths[idx] * ddist_dmeanx;//dL_dpc * dpc_dmeanx + dL_dpr * dpr_dmeanx + dL_ddist * ddist_dmeanx;
	view_dL_dmean.y = dL_dmeans[idx].y + dL_dsphere_means3D[idx].x * dspx_dmeany + dL_dsphere_means3D[idx].y * dspy_dmeany + dL_dsphere_means3D[idx].z * dspz_dmeany + dL_ddepths[idx] * ddist_dmeany;// dL_pc * dpc_dmeany + dL_dpr * dpr_dmeany + dL_ddist * ddist_dmeany;
	view_dL_dmean.z = dL_dmeans[idx].z + dL_dsphere_means3D[idx].x * dspx_dmeanz + dL_dsphere_means3D[idx].y * dspy_dmeanz + dL_dsphere_means3D[idx].z * dspz_dmeanz  + dL_ddepths[idx] * ddist_dmeanz;// dL_dpr * dpr_dmeanz + dL_ddist * ddist_dmeanz;

	// float3 view_dL_dmeans = {dL_dmeans[idx].x + dL_dmean.x , dL_dmeans[idx].y + dL_dmean.y , dL_dmeans[idx].z + dL_dmean.z}
	float3 dL_dmean = transformVec4x3Transpose({ view_dL_dmean.x, view_dL_dmean.y, view_dL_dmean.z }, viewmatrix);
	dL_dmeans[idx].x = dL_dmean.x;
	dL_dmeans[idx].y = dL_dmean.y;
	dL_dmeans[idx].z = dL_dmean.z;

	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__  depths,
	const float3* __restrict__ basis_u1,
	const float3* __restrict__ basis_u2,
	const float3* __restrict__ sphere_means3D,
	const float* __restrict__ beam_inclinations,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dout_depths,
	const float* __restrict__ dL_dout_occs,
	float4* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dsphere_means3D,
	float3* __restrict__ dL_dbasis_u1,
	float3* __restrict__ dL_dbasis_u2,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float3 collected_basis_u1[BLOCK_SIZE];
	__shared__ float3 collected_basis_u2[BLOCK_SIZE];
	__shared__ float3 collected_sphere_means3D[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_red = 0;
	float accum_reo = 0;
	float dL_dpixel[C];
	float dL_dout_depth = 0;
	float dL_dout_occ = 0;
	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id]; // 拿到对应pix_id像素的rgb
		dL_dout_depth = dL_dout_depths[pix_id]; // 对应像素的深度梯度
		dL_dout_occ = dL_dout_occs[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = W/2.0; // 规范化到同一尺度下 [0,1]
	const float ddely_dy = H/2.0;

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
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_basis_u1[block.thread_rank()] = basis_u1[coll_id];
			collected_basis_u2[block.thread_rank()] = basis_u2[coll_id];
			collected_sphere_means3D[block.thread_rank()] = sphere_means3D[coll_id];
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

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];

			float3 unit_sphere_xyz = collected_sphere_means3D[j]; //投影矩阵的中心 （减了雷达pos又除了深度到单位球）
			float3 u1 = collected_basis_u1[j]; // 投影平面的基向量
			float3 u2 = collected_basis_u2[j];
			float alp = beam_inclinations[H-1-pix.y];
			float beta = -(pixf.x - W / 2.0) / W * 2.0 * pi;
			const float3 uint_sphere_pixf = {cos(alp) * cos(beta), cos(alp) * sin(beta), sin(alp)};
			const float3 uint_sphere_d = {unit_sphere_xyz.x - uint_sphere_pixf.x, unit_sphere_xyz.y - uint_sphere_pixf.y, unit_sphere_xyz.z - uint_sphere_pixf.z}; // 规范空间下的相对位置
			// d = a*u1 + b*u2 //uint_sphere_d在切平面上的投影可以用两个基向量表示 a = (_d*u1)/(u1*u1) b = (_d*u2)/(u2*u2)
			const float u1_u1 = u1.x*u1.x + u1.y*u1.y + u1.z*u1.z;
			const float u2_u2 = u2.x*u2.x + u2.y*u2.y + u2.z*u2.z;
			const float _d_u1 = uint_sphere_d.x * u1.x + uint_sphere_d.y * u1.y + uint_sphere_d.z * u1.z;
			const float _d_u2 = uint_sphere_d.x * u2.x + uint_sphere_d.y * u2.y + uint_sphere_d.z * u2.z;
			const float2 d = {_d_u1/u1_u1, _d_u2/u2_u2}; 

			// const float2 d = { xy.x - pixf.x, 1*(xy.y - pixf.y) };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			// ----------------- dL_dalpha 和 dL_dcolors 对gs的颜色和不透明度的求导（推导见手稿） ---------------------------------
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;// 透明度只有一维 所以三个通道的grad加到一起
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			// float depth_cos_theta = unit_sphere_xyz.x * uint_sphere_pixf.x + unit_sphere_xyz.y * uint_sphere_pixf.y + unit_sphere_xyz.z * uint_sphere_pixf.z;
			// if(depth_cos_theta == 0) printf("[Backward Error]:depth_cos_theta shouldn't be zero!!");
			const float dep = collected_depths[j];// / depth_cos_theta; 
			accum_red = last_alpha * last_depth + (1.f - last_alpha) * accum_red;
			last_depth = dep;
			dL_dalpha += (dep-accum_red) * dL_dout_depth;
			atomicAdd(&(dL_ddepths[global_id]), dchannel_dcolor * dL_dout_depth );/// depth_cos_theta

			// Propagate gradients w.r.t. color ray-splat alphas
			accum_reo = last_alpha * 1.0 + (1.f - last_alpha) * accum_reo;
			dL_dalpha += (1 - accum_reo) * dL_dout_occ;


			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;// 没有background的不需要考虑这一步


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;  // d.x
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z  - gdx * con_o.y;

			// // 直接用商的求导法则公式 求对于投影平面的导数
			const float ddx_du1x = (uint_sphere_d.x * u1_u1 - _d_u1 * 2 * u1.x) / (u1_u1 * u1_u1);
			const float ddx_du1y = (uint_sphere_d.y * u1_u1 - _d_u1 * 2 * u1.y) / (u1_u1 * u1_u1);
			const float ddx_du1z = (uint_sphere_d.z * u1_u1 - _d_u1 * 2 * u1.z) / (u1_u1 * u1_u1);
			const float ddy_du2x = (uint_sphere_d.x * u2_u2 - _d_u2 * 2 * u2.x) / (u2_u2 * u2_u2);
			const float ddy_du2y = (uint_sphere_d.y * u2_u2 - _d_u2 * 2 * u2.y) / (u2_u2 * u2_u2);
			const float ddy_du2z = (uint_sphere_d.z * u2_u2 - _d_u2 * 2 * u2.z) / (u2_u2 * u2_u2);

			atomicAdd(&dL_dbasis_u1[global_id].x, dL_dG * dG_ddelx * ddx_du1x); 
			atomicAdd(&dL_dbasis_u1[global_id].y, dL_dG * dG_ddelx * ddx_du1y); 
			atomicAdd(&dL_dbasis_u1[global_id].z, dL_dG * dG_ddelx * ddx_du1z); 
			atomicAdd(&dL_dbasis_u2[global_id].x, dL_dG * dG_ddely * ddy_du2x); 
			atomicAdd(&dL_dbasis_u2[global_id].y, dL_dG * dG_ddely * ddy_du2y); 
			atomicAdd(&dL_dbasis_u2[global_id].z, dL_dG * dG_ddely * ddy_du2z); 

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); //* ddelx_dx lidar 不涉及 ddelx_dx
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely);  //* ddely_dy 投影平面上的梯度 可能还和pix平面不一致 但这个可以通过手动调整外面的参数 问题应该不大
			// Homodirectional Gradient
            // atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * collected_depths[j]/ u1_u1 ));//
            // atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * collected_depths[j]/ u2_u2 ));//
			
			const float ddx_dsx = u1.x/u1_u1;
			const float ddx_dsy = u1.y/u1_u1;
			const float ddx_dsz = u1.z/u1_u1;
			const float ddy_dsx = u2.x/u2_u2;
			const float ddy_dsy = u2.y/u2_u2;
			const float ddy_dsz = u2.z/u2_u2;
			const float dG_dsx = dG_ddelx*ddx_dsx + dG_ddely*ddy_dsx;
			const float dG_dsy = dG_ddelx*ddx_dsy + dG_ddely*ddy_dsy;
			const float dG_dsz = dG_ddelx*ddx_dsz + dG_ddely*ddy_dsz;
			// const float depth_cos_theta2 = depth_cos_theta*depth_cos_theta;
			float dmx = fabs(dG_ddelx*ddx_dsx) + fabs(dG_ddely*ddy_dsx);
			float dmy = fabs(dG_ddelx*ddx_dsy) + fabs(dG_ddely*ddy_dsy);
			float dmz = fabs(dG_ddelx*ddx_dsz) + fabs(dG_ddely*ddy_dsz);
			float dL_dsphere_means3Dx = dL_dG * dG_dsx;
			float dL_dsphere_means3Dy = dL_dG * dG_dsy;
			float dL_dsphere_means3Dz = dL_dG * dG_dsz;
			atomicAdd(&dL_dsphere_means3D[global_id].x, dL_dsphere_means3Dx ); //+ (-1 * uint_sphere_pixf.x / depth_cos_theta2)* dchannel_dcolor * dL_dout_depth
			atomicAdd(&dL_dsphere_means3D[global_id].y, dL_dsphere_means3Dy ); // + (-1 * uint_sphere_pixf.y / depth_cos_theta2)* dchannel_dcolor * dL_dout_depth
			atomicAdd(&dL_dsphere_means3D[global_id].z, dL_dsphere_means3Dz ); // + (-1 * uint_sphere_pixf.z / depth_cos_theta2)* dchannel_dcolor * dL_dout_depth
			// atomicAdd(&dL_dmean2D[global_id].z, sqrtf(dmx*dmx + dmy*dmy + dmz*dmz));//
			atomicAdd(&dL_dmean2D[global_id].z, sqrtf(dL_dsphere_means3Dx*dL_dsphere_means3Dx + dL_dsphere_means3Dy*dL_dsphere_means3Dy + dL_dsphere_means3Dz*dL_dsphere_means3Dz));//
            atomicAdd(&dL_dmean2D[global_id].w, 0.0);//

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const int width, int height,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float* beam_inclinations,
	const float4* dL_dmean2D,
	const float3* dL_dsphere_means3D,
	const float3* dL_dbasis_u1,
	const float3* dL_dbasis_u2,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepths,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		width, height,
		means3D,
		beam_inclinations,
		radii,
		cov3Ds,
		viewmatrix,
		campos,
		(float4*)dL_dmean2D,
		(float3*) dL_dbasis_u1,
		(float3*) dL_dbasis_u2,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	cudaDeviceSynchronize();
	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		width, height,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		beam_inclinations,
		(float4*)dL_dmean2D,
		(float3*)dL_dsphere_means3D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepths,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("preprocess Error %s \n",cudaGetErrorString(err));
	}
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* depths,
	const float3* basis_u1,
	const float3* basis_u2,
	const float3* sphere_means3D,
	const float* beam_inclinations,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dout_depths,
	const float* dL_dout_occ,
	float4* dL_dmean2D,
	float3* dL_dsphere_means3D,
	float3* dL_dbasis_u1,
	float3* dL_dbasis_u2,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		depths,
		basis_u1,
		basis_u2,
		sphere_means3D,
		beam_inclinations,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dout_depths,
		dL_dout_occ,
		dL_dmean2D,
		dL_dsphere_means3D,
		dL_dbasis_u1,
		dL_dbasis_u2,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths
		);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("preprocess Error %s \n",cudaGetErrorString(err));
	}
}