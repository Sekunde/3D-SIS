#include "stdafx.h"
#include "VoxelGrid.h"

void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned char>& label, const BaseImage<unsigned char>& instance, bool debugOut /*= false*/)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	if (debugOut) {
		PointCloudf pc;
		for (const auto& p : depthImage) {
			if (p.value > 0 && p.value != -std::numeric_limits<float>::infinity()) {
				vec3f cam = depthToSkeleton(intrinsic, p.x, p.y, p.value);
				vec3f worldpos = cameraToWorld * cam;
				pc.m_points.push_back(m_worldToGrid * worldpos);
				RGBColor c = RGBColor::colorPalette(instance(p.x, p.y));
				pc.m_colors.push_back(vec4f(c));
			}
		}
		PointCloudIOf::saveToFile("points.ply", pc);

		TriMeshf trimesh = computeInstanceMesh(m_voxelSize);
		MeshIOf::saveToFile("before-integrate.ply", trimesh.computeMeshData());
		trimesh = computeInstanceMesh(m_voxelSize * 2.0f);
		MeshIOf::saveToFile("before-integrate-1.5.ply", trimesh.computeMeshData());
	}

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
					unsigned char lbl = label(pi.x, pi.y);
					unsigned char inst = instance(pi.x, pi.y);

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

							if (std::fabs(v.sdf) <= 2.0f * m_voxelSize) {
								if (std::fabs(sdf) <= 2.0f * m_voxelSize && (v.color.r == 0 || (lbl != 0))) {
									v.color.r = lbl;
									v.color.g = inst;
								}
							}
						}
					}
				}

			}
		}
	}

	if (debugOut) {
		TriMeshf trimesh = computeInstanceMesh(m_voxelSize);
		MeshIOf::saveToFile("integrated.ply", trimesh.computeMeshData()); 
		trimesh = computeInstanceMesh(m_voxelSize * 2.0f);
		MeshIOf::saveToFile("integrated-1.5.ply", trimesh.computeMeshData());
		std::cout << "waiting..." << std::endl;
		getchar();
	}
}


vec2ui VoxelGrid::countOccupancyAABB(const bbox3f& aabb, unsigned int weightThresh, float sdfThresh, unsigned short instanceId, MaskType& mask) const {
	vec2ui occ(0, 0);
	bbox3i bounds(math::floor(aabb.getMin()), math::ceil(aabb.getMax()));
	MLIB_ASSERT(bounds.getExtent() != vec3f(0.0f));
	bounds.setMin(math::min(math::max(bounds.getMin(), 0), vec3i((int)getDimX() - 1, (int)getDimY() - 1, (int)getDimZ() - 1)));
	bounds.setMax(math::min(bounds.getMax(), vec3i((int)getDimX(), (int)getDimY(), (int)getDimZ())));
	mask.allocate(bounds.getExtent()); mask.setValues(0);

	const vec3i boundsMin = bounds.getMin();
	const vec3i boundsMax = bounds.getMax();
	for (int k = boundsMin.z; k < boundsMax.z; k++) {
		for (int j = boundsMin.y; j < boundsMax.y; j++) {
			for (int i = boundsMin.x; i < boundsMax.x; i++) {
				const Voxel& v = (*this)(i, j, k);
				if (v.weight >= weightThresh && std::fabs(v.sdf) <= sdfThresh) {
					occ.y++;
					const unsigned short instId = v.color.g;
					if (instId == instanceId) {
						occ.x++;
						vec3i coordBg = vec3i(i, j, k) - boundsMin;
						if (!mask.isValidCoordinate(coordBg))
							throw MLIB_EXCEPTION("bad coord compute for mask");
						mask(coordBg) = 1;
					}
				}
			}  // i
		}  // j
	}  // k

	return occ;
}

TriMeshf VoxelGrid::computeLabelMesh(float sdfThresh) const {
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
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);
					const unsigned char sem = v.color.r;

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for empty
						}
						else if (sem == 255) {
							triMesh.m_vertices.back().color = vec4f(0.5f, 0.5f, 0.5f, 1.0f); //gray for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette((unsigned int)sem);
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


TriMeshf VoxelGrid::computeInstanceMesh(float sdfThresh) const {
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
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);
					const unsigned char sem = v.color.g;

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for empty
						}
						else if (sem == 255) {
							triMesh.m_vertices.back().color = vec4f(0.5f, 0.5f, 0.5f, 1.0f); //gray for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette((unsigned int)sem);
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