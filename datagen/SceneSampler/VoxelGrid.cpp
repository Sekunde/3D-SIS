
#include "stdafx.h"

#include "VoxelGrid.h"


void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());


	//std::cout << "camera to world" << std::endl << cameraToWorld << std::endl;
	//std::cout << "world to camera" << std::endl << worldToCamera << std::endl;

	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {

				//transform to current frame
				vec3f pf = worldToCamera * voxelToWorld(vec3i(i, j, k));
				//vec3f p = worldToCamera * (m_gridToWorld * ((vec3f(i, j, k) + 0.5f)));

				//project into depth image
				vec3f p = skeletonToDepth(intrinsic, pf);

				vec3i pi = math::round(p);

				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) {
					float d = depthImage(pi.x, pi.y);

					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {

						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);

						if (sdf > -truncation) {
							if (sdf >= 0.0f) {
								sdf = fminf(truncation, sdf);
							}
							else {
								sdf = fmaxf(-truncation, sdf);
							}
							const float integrationWeightSample = 3.0f;
							const float depthWorldMin = 0.4f;
							const float depthWorldMax = 4.0f;
							float depthZeroOne = (d - depthWorldMin) / (depthWorldMax - depthWorldMin);
							float weightUpdate = std::max(integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);

							Voxel& v = (*this)(i, j, k);
							if (v.sdf == -std::numeric_limits<float>::infinity()) {
								v.sdf = sdf;
							}
							else {
								v.sdf = (v.sdf * (float)v.weight + sdf * weightUpdate) / (float)(v.weight + weightUpdate);
							}
							v.weight = (uchar)std::min((int)v.weight + (int)weightUpdate, (int)std::numeric_limits<unsigned char>::max());
						}
					}
				}

			}
		}
	}
}

void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned short>& semantics)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {

				//transform to current frame
				vec3f p = worldToCamera * voxelToWorld(vec3i(i, j, k));

				//project into depth image
				p = skeletonToDepth(intrinsic, p);

				vec3i pi = math::round(p);
				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) {
					const float d = depthImage(pi.x, pi.y);
					const unsigned short sem = semantics(pi.x, pi.y);

					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {

						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);

						if (sdf > -truncation) {
							Voxel& v = (*this)(i, j, k);
							if (std::abs(sdf) <= std::abs(v.sdf)) {
								if (sdf >= 0.0f || v.sdf <= 0.0f) {
									v.sdf = sdf;
									v.color[0] = sem & 0xff; //ushort to vec2uc
									v.color[1] = (sem >> 8) & 0xff;
									v.weight = 1;
								}
							}
							//std::cout << "v: " << v.sdf << " " << (int)v.weight << std::endl;
						}
					}
				}

			}
		}
	}
}

TriMeshf VoxelGrid::computeSemanticsMesh(float sdfThresh) const {
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (unsigned int z = 0; z < getDimZ(); z++) {
		for (unsigned int y = 0; y < getDimY(); y++) {
			for (unsigned int x = 0; x < getDimX(); x++) {
				if (std::fabs((*this)(x, y, z).sdf) < sdfThresh) nVoxels++;
			}
		}
	}
	size_t nVertices = nVoxels * 8; //no normals
	size_t nIndices = nVoxels * 12;
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < getDimZ(); z++) {
		for (size_t y = 0; y < getDimY(); y++) {
			for (size_t x = 0; x < getDimX(); x++) {
				const Voxel& v = (*this)(x, y, z);
				if (std::fabs(v.sdf) < sdfThresh) {
					vec3f p((float)x, (float)y, (float)z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);
					const unsigned short sem = (((unsigned short)v.color[1]) << 8) | v.color[0];

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette(sem);
							triMesh.m_vertices.back().color = vec4f(vec3f(c.x, c.y, c.z) / 255.0f);
						}
					}
					for (size_t i = 0; i < 12; i++) {
						indices[i] += vertIdxBase;
						triMesh.m_indices.emplace_back(indices[i]);
					}
				}
			}
		}
	}
	triMesh.setHasColors(true);

	return triMesh;
}