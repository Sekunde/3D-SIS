#pragma once
#include "BBInfo.h"
#include "VoxelGrid.h"

class BBHelper {
public:
	// -------------------- AABBs ------------------ //
	
	static TriMeshf convertAnnGridToMesh(const Grid3<unsigned short>& annotated)
	{
		TriMeshf triMesh;

		// Pre-allocate space
		size_t nVoxels = 0;
		for (const auto& v : annotated) {
			if (v.value != 0) nVoxels++;
		}
		size_t nVertices = nVoxels * 8; //no normals
		size_t nIndices = nVoxels * 12;
		triMesh.m_vertices.reserve(nVertices);
		triMesh.m_indices.reserve(nIndices);
		// Temporaries
		vec3f verts[24];
		vec3ui indices[12];
		vec3f normals[24];
		for (size_t z = 0; z < annotated.getDimZ(); z++) {
			for (size_t y = 0; y < annotated.getDimY(); y++) {
				for (size_t x = 0; x < annotated.getDimX(); x++) {
					unsigned short val = annotated(x, y, z);
					if (val != 0) {
						vec3f p(x, y, z);
						vec3f pMin = p - 0.45f;//0.5f;
						vec3f pMax = p + 0.45f;//0.5f;
						bbox3f bb(pMin, pMax);
						bb.makeTriMesh(verts, indices);

						unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
						for (size_t i = 0; i < 8; i++) {
							triMesh.m_vertices.emplace_back(verts[i]);
							if (val == 255) {
								triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 0.0f); //black for no annotation
							}
							else {
								RGBColor color = RGBColor::colorPalette(val);
								triMesh.m_vertices.back().color = color;
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

	static std::tuple<MeshDataf, MeshDataf, MeshDataf> visualizeAABBs_Instance(const std::vector<BBInfo>& aabbs, const mat4f& worldToGrid, const vec3ui& gridDim) {
		MeshDataf meshAabbs;
		Grid3<unsigned short> bgMaskPartial(gridDim);
		bgMaskPartial.setValues(0);
		Grid3<unsigned short> bgMaskComplete(gridDim);
		bgMaskComplete.setValues(0);

		const float radius = 0.5f; // grid space
		for (unsigned int i = 0; i < aabbs.size(); i++) {
			bbox3f aabbGrid = aabbs[i].aabb;
			aabbGrid.transform(worldToGrid);

			const MaskType& mask = aabbs[i].mask;
			const RGBColor c = RGBColor::colorPalette(aabbs[i].instanceId);
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
								bgMaskPartial(coordWorld) = aabbs[i].instanceId;
								bgMaskComplete(coordWorld) = aabbs[i].instanceId; // for vis we don't want this
							}
							else {
								bgMaskComplete(coordWorld) = aabbs[i].instanceId;
							}
						}
					}  // z
				}  // y
			}  // x
			
		}  // aabbs/masks
		MeshDataf meshMasksPartial = BBHelper::convertAnnGridToMesh(bgMaskPartial).computeMeshData();
		MeshDataf meshMasksComplete = BBHelper::convertAnnGridToMesh(bgMaskComplete).computeMeshData();
		return std::make_tuple(meshAabbs, meshMasksPartial, meshMasksComplete);
	}
	static void exportAABBsToFile(const std::vector<BBInfo>& aabbs, const std::string& outFile, const mat4f& worldToGrid = mat4f::identity()) {
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

			// -- NEW PART --
			bbox3f aabbCanonical(o.aabbCanonical); aabbCanonical.transform(worldToGrid);
			vec3f canonicaMin = aabbCanonical.getMin(), canonicalMax = aabbCanonical.getMax();
			ofs.writeData((const BYTE*)canonicaMin.getData(), sizeof(vec3f));
			ofs.writeData((const BYTE*)canonicalMax.getData(), sizeof(vec3f));
			ofs.writeData((const BYTE*)&o.angleCanonical, sizeof(float));
			ofs << o.maskCanonical;
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

			// -- NEW PART --
			vec3f canonicaMin, canonicalMax;
			ifs.readData((BYTE*)canonicaMin.getData(), sizeof(vec3f));
			ifs.readData((BYTE*)canonicalMax.getData(), sizeof(vec3f));
			aabbs[i].aabbCanonical = bbox3f(canonicaMin, canonicalMax);
			ifs.readData((BYTE*)&aabbs[i].angleCanonical, sizeof(float));
			ifs >> aabbs[i].maskCanonical;
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
		MeshDataf meshMasks = TriMeshf(bgMaskPartial, mat4f::identity(), false, vec4f(1.0, 0.0f, 0.0f, 1.0f)).computeMeshData();
		meshMasks.merge(TriMeshf(bgMaskComplete, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		return std::make_pair(meshAabbs, meshMasks);
	}
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

	static void computeValidAabbs(const TriMeshf& sceneMeshGrid, const VoxelGrid& grid, const std::vector<BBInfo>& aabbs, const std::vector<MaskType>& origmasks, 
		const vec3f& sceneBoxMin, unsigned int scenePad, float occThresh, unsigned int minNumOcc, 
		std::vector<BBInfo>& validAabbs) {
		const mat4f worldToGrid = grid.getWorldToGrid();
		const unsigned int weightThresh = 0;
		const float sdfThresh = 1.0f;

		validAabbs.clear();
		for (unsigned int i = 0; i < aabbs.size(); i++) {
			const auto& o = aabbs[i];
			bbox3f aabb(o.aabb); aabb.transform(worldToGrid); // in grid space
			// check if valid (enough occupancy in volume)
			MaskType mask;
			const vec2ui occ = grid.countOccupancyAABB(aabb, weightThresh, sdfThresh, o.instanceId, sceneMeshGrid, mask, true);

			if (occ.y == 0) continue;
			float percent = (float)occ.x / (float)occ.y;
			if (occ.x >= minNumOcc && percent >= occThresh) {
				validAabbs.push_back(o);
				validAabbs.back().mask = mask;
				validAabbs.back().maskCanonical = origmasks[i];
			}
		}
	}

	static void computeMasks(const TriMeshf& sceneMeshGrid, const VoxelGrid& grid, const std::vector<BBInfo>& aabbs, const vec3f& sceneBoxMin, unsigned int scenePad,
		std::vector<MaskType>& masks) {
		const mat4f worldToGrid = grid.getWorldToGrid();
		const unsigned int weightThresh = 0;
		const float sdfThresh = 1.0f;

		masks.clear();
		for (const auto& o : aabbs) {
			bbox3f aabb(o.aabb); aabb.transform(worldToGrid); // in grid space
			// check if valid (enough occupancy in volume)
			MaskType mask;
			const vec2ui occ = grid.countOccupancyAABB(aabb, weightThresh, sdfThresh, o.instanceId, sceneMeshGrid, mask, false);
			masks.push_back(mask);
		}
	}

	// -------------------- 2D bboxes ------------------ //

	static std::vector<std::pair<bbox2i, unsigned short>> compute2DBboxes(const BaseImage<unsigned short>& semanticInstance, const BaseImage<unsigned short>& semanticLabel)
	{
		std::vector<std::pair<bbox2i, unsigned short>> bboxes;
		// determine number of different instances
		std::unordered_map<unsigned short, unsigned short> instToSemMap;
		for (const auto& p : semanticInstance) {
			if (p.value > 0) {
				unsigned short sem = semanticLabel(p.x, p.y);
				if (sem != 1 && sem != 2 && sem != 20 && sem != 22)  { // ignore structural semantic elements
					if (instToSemMap.find(p.value) == instToSemMap.end())
						instToSemMap[p.value] = sem;
				}
			}
		}
		for (const auto& inst : instToSemMap) {
			vec2i bboxmin(semanticInstance.getDimX(), semanticInstance.getDimY()), bboxmax(0, 0);
			for (const auto& p : semanticInstance) {
				if (p.value == inst.first) {
					if ((int)p.x < bboxmin.x) bboxmin.x = (int)p.x;
					if ((int)p.y < bboxmin.y) bboxmin.y = (int)p.y;
					if ((int)p.x > bboxmax.x) bboxmax.x = (int)p.x;
					if ((int)p.y > bboxmax.y) bboxmax.y = (int)p.y;
				}
			}
			bboxes.push_back(std::make_pair(bbox2i(bboxmin, bboxmax), inst.second));
		}
		return bboxes;
	}
	static void write2DBboxesToFile(const std::string& outputFile, const std::vector<std::pair<bbox2i, unsigned short>>& bboxes)
	{
		std::ofstream ofs(outputFile);
		if (!ofs.is_open()) throw MLIB_EXCEPTION("failed to open " + outputFile + " for write");
		ofs << bboxes.size() << std::endl; // #bboxes
		for (const auto& bb : bboxes)
			ofs << bb.first.getMin() << " " << bb.first.getMax() << "\t" << bb.second << std::endl;
		ofs.close();
	}

};  // class BBHelper