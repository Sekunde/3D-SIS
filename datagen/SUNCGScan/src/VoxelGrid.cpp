
#include "stdafx.h"

#include "VoxelGrid.h"


void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned short>& semantics)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {

				//transform to current frame
				vec3f p = worldToCamera * voxelToWorld(vec3i(i, j, k));
				//vec3f p = worldToCamera * (m_gridToWorld * ((vec3f(i, j, k) + 0.5f)));

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
					vec3f p(x, y, z);
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


vec2ui VoxelGrid::countOccupancyAABB(const bbox3f& aabb, unsigned int weightThresh, float sdfThresh, unsigned short instanceId, const TriMeshf& sceneMeshGrid, MaskType& mask, bool bComputeCompleteMask) const {
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
					const unsigned short instId = (((unsigned short)v.color[1]) << 8) | v.color[0];
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

	if (bComputeCompleteMask && occ.x > 0) { // compute complete mask
		bbox3f boundsf(bounds.getMin(), bounds.getMax());
		TriMeshf scenePart;
		scenePart.m_vertices = sceneMeshGrid.m_vertices;
		for (unsigned int i = 0; i < sceneMeshGrid.m_indices.size(); i++) {
			const auto& ind = sceneMeshGrid.m_indices[i];
			const vec3f& p0 = sceneMeshGrid.m_vertices[ind[0]].position;
			const vec3f& p1 = sceneMeshGrid.m_vertices[ind[1]].position;
			const vec3f& p2 = sceneMeshGrid.m_vertices[ind[2]].position;
			const unsigned short& id1 = (unsigned short)sceneMeshGrid.m_vertices[ind[0]].color.w % 1000;
			const unsigned short& id2 = (unsigned short)sceneMeshGrid.m_vertices[ind[1]].color.w % 1000;
			const unsigned short& id3 = (unsigned short)sceneMeshGrid.m_vertices[ind[2]].color.w % 1000;
			if ((boundsf.intersects(p0) || boundsf.intersects(p1) || boundsf.intersects(p2))
				&& (id1 == instanceId) && (id2 == instanceId) && (id3 == instanceId))
				scenePart.m_indices.push_back(ind);
		}
		BinaryGrid3 bgPart(mask.getDimensions());
		scenePart.voxelize(bgPart, mat4f::translation(-boundsMin), false, false);
		occ.y = (unsigned int)bgPart.getNumOccupiedEntries();
		for (auto& v : mask) {
			if (bgPart.isVoxelSet(v.x, v.y, v.z) && v.value == 0) v.value = 2;
		}
	}
	return occ;
}
