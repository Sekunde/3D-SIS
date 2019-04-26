#pragma once

#include "GlobalAppState.h"
#include "BBInfo.h"

namespace BBHelper {

	// -------------------- AABBs ------------------ //
	static void readAABBsFromFile(const std::string& file, std::vector<BBInfo>& aabbs) {
		aabbs.clear();
		BinaryDataStreamFile ifs(file, false);
		UINT64 numAABBs;
		ifs.readData((BYTE*)&numAABBs, sizeof(UINT64));
		aabbs.resize(numAABBs);
		for (unsigned int i = 0; i < numAABBs; i++) {
			vec3f anchor, axisX, axisY, axisZ;
			ifs.readData((BYTE*)anchor.getData(), sizeof(vec3f));
			ifs.readData((BYTE*)axisX.getData(), sizeof(vec3f));
			ifs.readData((BYTE*)axisY.getData(), sizeof(vec3f));
			ifs.readData((BYTE*)axisZ.getData(), sizeof(vec3f));
			const vec3f max = anchor + axisX + axisY + axisZ;
			unsigned short label;
			ifs.readData((BYTE*)&label, sizeof(unsigned short));
			aabbs[i].aabb = bbox3f(anchor, max);
			aabbs[i].labelId = label;
			ifs >> aabbs[i].mask;

#ifdef SUNCG
			vec3f canonicaMin, canonicalMax;
			ifs.readData((BYTE*)canonicaMin.getData(), sizeof(vec3f));
			ifs.readData((BYTE*)canonicalMax.getData(), sizeof(vec3f));
			aabbs[i].aabbCanonical = bbox3f(canonicaMin, canonicalMax);
			ifs.readData((BYTE*)&aabbs[i].angleCanonical, sizeof(float));
			ifs >> aabbs[i].maskCanonical;
#endif
		}
		ifs.close();
	}

	static std::pair<MeshDataf, MeshDataf> visualizeAABBs(const std::vector<BBInfo>& aabbs, const mat4f& worldToGrid, const vec3ui& gridDim) {
		MeshDataf meshAabbs;
		BinaryGrid3 bgMaskPartial(gridDim), bgMaskComplete(gridDim);
		const float radius = 0.5f; // grid space
		for (unsigned int i = 0; i < aabbs.size(); i++) {
			bbox3f aabbGrid = aabbs[i].aabb;
			aabbGrid.transform(worldToGrid);

			const MaskType& mask = aabbs[i].mask;
			const RGBColor c = RGBColor::colorPalette(i);
			// aabb
			for (const LineSegment3f& e : aabbGrid.getEdges()) meshAabbs.merge(Shapesf::cylinder(e.p0(), e.p1(), radius, 10, 10, c).computeMeshData());
			// mask
			const mat4f cubeToWorld = aabbGrid.cubeToWorldTransform();
			const vec3f maskDims(mask.getDimensions());
			for (unsigned int z = 0; z < mask.getDimZ(); z++) {
				for (unsigned int y = 0; y < mask.getDimY(); y++) {
					for (unsigned int x = 0; x < mask.getDimX(); x++) {
						const auto maskval = mask(x, y, z);
						if (maskval > 0) {
							const vec3i coordWorld = vec3i(x, y, z) + math::floor(aabbGrid.getMin());
							if (!bgMaskPartial.isValidCoordinate(coordWorld))
								throw MLIB_EXCEPTION("[BBHelper::visualizeAABBs] bad grid coord for mask element");
							if (maskval == 1) {
								bgMaskPartial.setVoxel(coordWorld);
							}
							else {
								bgMaskComplete.setVoxel(coordWorld);
							}
						}
					}  // z
				}  // y
			}  // x

		}  // aabbs/masks
		MeshDataf meshMasks = TriMeshf(bgMaskPartial, mat4f::identity(), false, vec4f(0.0f, 1.0f, 0.0f, 1.0f)).computeMeshData();
		meshMasks.merge(TriMeshf(bgMaskComplete, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		return std::make_pair(meshAabbs, meshMasks);
	}
#ifdef SUNCG
	static std::pair<MeshDataf, MeshDataf> visualizeAABBCanonicals(const std::vector<BBInfo>& aabbs, const mat4f& worldToGrid, const vec3ui& gridDim) {
		MeshDataf meshAabbs;
		BinaryGrid3 bgMaskPartial(gridDim), bgMaskComplete(gridDim);
		const float radius = 0.5f; // grid space
		for (unsigned int i = 0; i < aabbs.size(); i++) {
			OBB3f obbGrid = OBB3f(aabbs[i].aabbCanonical);
			const vec3f center = aabbs[i].aabbCanonical.getCenter();
			mat4f transform = mat4f::translation(center) * mat4f::rotationY(aabbs[i].angleCanonical) * mat4f::translation(-center);
			obbGrid = (worldToGrid * transform) * obbGrid;

			const MaskType& mask = aabbs[i].mask;
			const RGBColor c = RGBColor::colorPalette(i);
			// aabb
			for (const LineSegment3f& e : obbGrid.getEdges()) meshAabbs.merge(Shapesf::cylinder(e.p0(), e.p1(), radius, 10, 10, c).computeMeshData());
			// mask
			const mat4f obbToWorld = obbGrid.getOBBToWorld();
			const vec3f maskDims(mask.getDimensions());
			for (unsigned int z = 0; z < mask.getDimZ(); z++) {
				for (unsigned int y = 0; y < mask.getDimY(); y++) {
					for (unsigned int x = 0; x < mask.getDimX(); x++) {
						const auto maskval = mask(x, y, z);
						if (maskval > 0) {
							const vec3f coordObb = vec3f((float)x / maskDims.x, (float)y / maskDims.y, (float)z / maskDims.z); // in unit cube
							const vec3i coordWorld = math::round(obbToWorld * coordObb);
							if (!bgMaskPartial.isValidCoordinate(coordWorld))
								throw MLIB_EXCEPTION("[BBHelper::visualizeAABBCanonicals] bad grid coord for mask element");
							if (maskval == 1) {
								bgMaskPartial.setVoxel(coordWorld);
								//bgMaskComplete.setVoxel(coordWorld); // for vis we don't want this
							}
							else {
								bgMaskComplete.setVoxel(coordWorld);
							}
						}
					}  // z
				}  // y
			}  // x

		}  // aabbs/masks
		MeshDataf meshMasks = TriMeshf(bgMaskPartial, mat4f::identity(), false, vec4f(0.0f, 1.0f, 0.0f, 1.0f)).computeMeshData();
		meshMasks.merge(TriMeshf(bgMaskComplete, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		return std::make_pair(meshAabbs, meshMasks);
	}
#endif

}  // namespace BBHelper