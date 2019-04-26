#pragma once
#include "BBInfo.h"
#include "VoxelGrid.h"

class BBHelper {
public:

	// -------------------- AABBs ------------------ //

	static void exportAABBsToFile(const std::vector<BBInfo>& aabbs, const std::string& outFile, const mat4f& worldToGrid = mat4f::identity()) {
		// sanity check
		{
			for (const auto& bb : aabbs) {
				if (!bb.aabb.isValid()) {
					std::cout << "warning: found invalid aabb in " << outFile << ", skipping" << std::endl;
					return;
				}
				if (bb.mask.getDimX() > 10000) {
					std::cout << "warning: found invalid mask in " << outFile << ", skipping" << std::endl;
					return;
				}
			}
		}

		BinaryDataStreamFile ofs(outFile, true);
		UINT64 numAabbs = (UINT64)aabbs.size();
		ofs.writeData((const BYTE*)&numAabbs, sizeof(UINT64)); // #obbs in scene
		for (unsigned int i = 0; i < aabbs.size(); i++) {
			const BBInfo& o = aabbs[i];
			// aabb as origin point + 3 vectors + (ushort) label id
			bbox3f aabb(o.aabb); aabb.transform(worldToGrid);
			vec3f anchor = aabb.getMin();
			ofs.writeData((const BYTE*)anchor.getData(), sizeof(vec3f));
			// x
			const vec3f axisX = aabb.getExtentX() * vec3f::eX;
			ofs.writeData((const BYTE*)axisX.array, sizeof(vec3f));
			// y
			const vec3f axisY = aabb.getExtentY() * vec3f::eY;
			ofs.writeData((const BYTE*)axisY.array, sizeof(vec3f));
			// z
			const vec3f axisZ = aabb.getExtentZ() * vec3f::eZ;
			ofs.writeData((const BYTE*)axisZ.array, sizeof(vec3f));

			ofs.writeData((const BYTE*)&o.labelId, sizeof(unsigned short));
			ofs << o.mask;
		}
		ofs.close();
	}

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

	//static void computeValidAabbs(const TriMeshf& sceneMeshGrid, const VoxelGrid& grid, const std::vector<BBInfo>& aabbs, const std::vector<MaskType>& origmasks, 
	//	const vec3f& sceneBoxMin, unsigned int scenePad, float occThresh, unsigned int minNumOcc, 
	//	std::vector<BBInfo>& validAabbs) {
	//	const mat4f worldToGrid = grid.getWorldToGrid();
	//	const unsigned int weightThresh = 0;
	//	const float sdfThresh = 1.0f;

	//	validAabbs.clear();
	//	for (unsigned int i = 0; i < aabbs.size(); i++) {
	//		const auto& o = aabbs[i];
	//		bbox3f aabb(o.aabb); aabb.transform(worldToGrid); // in grid space
	//		// check if valid (enough occupancy in volume)
	//		MaskType mask;
	//		const vec2ui occ = grid.countOccupancyAABB(aabb, weightThresh, sdfThresh, o.instanceId, sceneMeshGrid, mask, true);
	//		//if (occ.y == 0) throw MLIB_EXCEPTION("invalid aabb - no occupancy found");
	//		if (occ.y == 0) continue;
	//		float percent = (float)occ.x / (float)occ.y;
	//		if (occ.x >= minNumOcc && percent >= occThresh) {
	//			validAabbs.push_back(o);
	//			validAabbs.back().mask = mask;

	//			////debugging
	//			//TriMeshf meshBg(bg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f));
	//			//meshBg.transform(mat4f::translation(math::floor(aabb.getMin())));
	//			//MeshIOf::saveToFile("test1.ply", meshBg.computeMeshData());
	//			//int a = 5;
	//			////debugging
	//		}
	//	}
	//}

	static void computeMasks(const VoxelGrid& grid, std::vector<BBInfo>& aabbs) {
		const mat4f worldToGrid = grid.getWorldToGrid();
		const unsigned int weightThresh = 0;
		const float sdfThresh = 2.0f;

		for (auto& o : aabbs) {
			bbox3f aabb(o.aabb); aabb.transform(worldToGrid); // in grid space
			// check if valid (enough occupancy in volume)
			const vec2ui occ = grid.countOccupancyAABB(aabb, weightThresh, sdfThresh, o.instanceId, o.mask);

			////debugging
			//MeshDataf bbMesh;
			//for (const auto& e : aabb.getEdges()) bbMesh.merge(Shapesf::cylinder(e.p0(), e.p1(), 0.02f, 10, 10, vec4f(0.0f, 1.0f, 0.0f, 1.0f)).computeMeshData());
			//MeshIOf::saveToFile("bb.ply", bbMesh);
			//BinaryGrid3 maskBg(o.mask.getDimensions());
			//for (const auto& v : o.mask) {
			//	if (v.value > 0) maskBg.setVoxel(v.x, v.y, v.z);
			//}
			//MeshDataf maskMesh = TriMeshf(maskBg).computeMeshData();
			//maskMesh.applyTransform(mat4f::translation(aabb.getMin()));
			//MeshIOf::saveToFile("mask.ply", maskMesh);
			////debugging
			//std::cout << "waiting..." << std::endl;
			//getchar();
		}
	}


};  // class BBHelper