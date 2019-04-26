#pragma once

#include "mLibInclude.h"
#include "BBInfo.h"

struct Voxel {
	Voxel() {
		sdf = -std::numeric_limits<float>::infinity();
		freeCtr = 0;
		color = vec3uc(0, 0, 0);
		weight = 0;
	}

	float			sdf;
	unsigned int	freeCtr;
	vec3uc			color;
	uchar			weight;
};

class VoxelGrid : public Grid3 < Voxel >
{
public:

	VoxelGrid(const vec3l& dim, const mat4f& worldToGrid, float voxelSize, float depthMin, float depthMax) : Grid3(dim.x, dim.y, dim.z) {
		m_voxelSize = voxelSize;
		m_depthMin = depthMin;
		m_depthMax = depthMax;
		m_worldToGrid = worldToGrid;
		m_gridToWorld = m_worldToGrid.getInverse();

		m_trunaction = m_voxelSize * 2.5f;
		m_truncationScale = m_voxelSize;
		m_weightUpdate = 1;
	}

	~VoxelGrid() {

	}

	void reset() {
#pragma omp parallel for
		for (int i = 0; i < (int)getNumElements(); i++) {
			getData()[i] = Voxel();
		}
	}

	void integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned short>& semantics);

	//vec2ui countOccupancyOBB(const OBB3f& obb, unsigned int weightThresh, float sdfThresh, unsigned short instanceId, const TriMeshf& sceneMeshGrid, MaskType& mask) const;
	vec2ui countOccupancyAABB(const bbox3f& aabb, unsigned int weightThresh, float sdfThresh, unsigned short instanceId, const TriMeshf& sceneMeshGrid, MaskType& mask, bool bComputeCompleteMask) const;

	//! normalizes the SDFs (divides by the voxel size)
	void normalizeSDFs()  {
		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					Voxel& v = (*this)(i, j, k);
					if (v.sdf != -std::numeric_limits<float>::infinity() && v.sdf != 0.0f) {
						v.sdf /= m_voxelSize;
					}

				}
			}
		}
		m_voxelSize = 1.0f;
	}

	//! returns all the voxels on the isosurface
	std::vector<Voxel> getSurfaceVoxels(unsigned int weightThresh, float sdfThresh) const {

		std::vector<Voxel> res;
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					const Voxel& v = (*this)(i, j, k);
					if (v.weight >= weightThresh && std::abs(v.sdf) < sdfThresh) {
						res.push_back(v);
					}
				}
			}
		}

		return res;
	}

	BinaryGrid3 toBinaryGridFree(unsigned int freeThresh) const {
		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					if ((*this)(i, j, k).freeCtr >= freeThresh) {
						res.setVoxel(i, j, k);
					}
				}
			}
		}
		return res;
	}

	BinaryGrid3 toBinaryGridOccupied(unsigned int weightThresh, float sdfThresh) const {

		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					if ((*this)(i, j, k).weight >= weightThresh && std::abs((*this)(i, j, k).sdf) < sdfThresh) {
						//if ((*this)(i, j, k).weight >= weightThresh && (*this)(i, j, k).sdf < sdfThresh && (*this)(i, j, k).sdf >= 0) {
						res.setVoxel(i, j, k);
					}
				}
			}
		}
		return res;
	}


	TriMeshf computeSemanticsMesh(float sdfThresh) const;
	
	void saveToFile(const std::string& filename, float origVoxelSize, const bbox3f& bbox) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)&origVoxelSize, sizeof(float)); //TODO: also include m_voxelSize??
		vec3f bmin = bbox.getMin();
		vec3f bmax = bbox.getMax();
		ofs.write((const char*)&bmin.x, sizeof(float));
		ofs.write((const char*)&bmin.y, sizeof(float));
		ofs.write((const char*)&bmin.z, sizeof(float));
		ofs.write((const char*)&bmax.x, sizeof(float));
		ofs.write((const char*)&bmax.y, sizeof(float));
		ofs.write((const char*)&bmax.z, sizeof(float));
		//dense data
		std::vector<float> values(getNumElements());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			values[i] = getData()[i].sdf;
		}
		ofs.write((const char*)values.data(), sizeof(float)*values.size());
		ofs.close();
	}
	void saveNormalizedToFile(const std::string& filename, float origVoxelSize, const bbox3f& bbox) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)&origVoxelSize, sizeof(float)); //TODO: also include m_voxelSize??
		vec3f bmin = bbox.getMin();
		vec3f bmax = bbox.getMax();
		ofs.write((const char*)&bmin.x, sizeof(float));
		ofs.write((const char*)&bmin.y, sizeof(float));
		ofs.write((const char*)&bmin.z, sizeof(float));
		ofs.write((const char*)&bmax.x, sizeof(float));
		ofs.write((const char*)&bmax.y, sizeof(float));
		ofs.write((const char*)&bmax.z, sizeof(float));
		//dense data
		std::vector<float> values(getNumElements());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			values[i] = getData()[i].sdf;
			if (values[i] != -std::numeric_limits<float>::infinity() && values[i] != 0.0f)
				values[i] /= m_voxelSize;
		}
		ofs.write((const char*)values.data(), sizeof(float)*values.size());
		ofs.close();
	}
	void loadFromFile(const std::string& filename, float& origVoxelSize, bbox3f& bbox) {
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs.is_open()) throw MLIB_EXCEPTION("[VoxelGrid::loadFromFile] " + filename + " does not exist!");
		//metadata
		UINT64 dimX, dimY, dimZ;
		ifs.read((char*)&dimX, sizeof(UINT64));
		ifs.read((char*)&dimY, sizeof(UINT64));
		ifs.read((char*)&dimZ, sizeof(UINT64));
		ifs.read((char*)&origVoxelSize, sizeof(float));
		m_voxelSize = 1.0f; //TODO: also include m_voxelSize??
		vec3f bmin, bmax;
		ifs.read((char*)&bmin.x, sizeof(float));
		ifs.read((char*)&bmin.y, sizeof(float));
		ifs.read((char*)&bmin.z, sizeof(float));
		ifs.read((char*)&bmax.x, sizeof(float));
		ifs.read((char*)&bmax.y, sizeof(float));
		ifs.read((char*)&bmax.z, sizeof(float));
		bbox = bbox3f(bmin, bmax);
		//dense data
		allocate(dimX, dimY, dimZ);
		std::vector<float> values(getNumElements());
		ifs.read((char*)values.data(), sizeof(float)*values.size());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			Voxel& v = getData()[i];
			v.sdf = values[i];
			if (v.sdf >= -1) v.weight = 1;
			else v.weight = 0;
		}
		ifs.close();
	}


	void saveSemanticsToFile(const std::string& filename) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		//dense data
		std::vector<unsigned short> values(getNumElements());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			const Voxel& v = getData()[i];
			values[i] = (((unsigned short)v.color[1]) << 8) | v.color[0];
		}
		ofs.write((const char*)values.data(), sizeof(unsigned short)*values.size());
		ofs.close();
	}
	//! just for debugging
	void loadSemanticsFromFile(const std::string& filename) {
		std::ifstream ifs(filename, std::ios::binary);
		//metadata
		UINT64 dimX, dimY, dimZ;
		ifs.read((char*)&dimX, sizeof(UINT64));
		ifs.read((char*)&dimY, sizeof(UINT64));
		ifs.read((char*)&dimZ, sizeof(UINT64));
		if (getDimX() != dimX || getDimY() != dimY || getDimZ() != dimZ)
			throw MLIB_EXCEPTION("[loadSemanticsFromFile] mismatch dimensions");
		//dense data
		std::vector<unsigned short> values(getNumElements());
		ifs.read((char*)values.data(), sizeof(unsigned short)*values.size());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			Voxel& v = getData()[i];
			v.color[0] = values[i] & 0xff; //ushort to vec2uc
			v.color[1] = (values[i] >> 8) & 0xff;
		}
		ifs.close();
	}

	void setWorldToGrid(const mat4f& worldToGrid) {
		m_worldToGrid = worldToGrid;
		m_gridToWorld = worldToGrid.getInverse();
	}

	mat4f getGridToWorld() const {
		return m_gridToWorld;
	}

	mat4f getWorldToGrid() const {
		return m_worldToGrid;
	}

	vec3i worldToVoxel(const vec3f& p) const {
		return math::round((m_worldToGrid * p));
	}

	vec3f worldToVoxelFloat(const vec3f& p) const {
		return (m_worldToGrid * p);
	}

	vec3f voxelToWorld(vec3i& v) const {
		return m_gridToWorld * (vec3f(v));
	}

	float getVoxelSize() const {
		return m_voxelSize;
	}


	bool trilinearInterpolationSimpleFastFast(const vec3f& pos, float& dist, vec3uc& color) const {
		const float oSet = m_voxelSize;
		const vec3f posDual = pos - vec3f(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
		vec3f weight = frac(worldToVoxelFloat(pos));

		dist = 0.0f;
		vec3f colorFloat = vec3f(0.0f, 0.0f, 0.0f);

		Voxel v; vec3f vColor;
		v = getVoxel(posDual + vec3f(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, oSet, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;

		color = vec3uc(math::round(colorFloat.x), math::round(colorFloat.y), math::round(colorFloat.z));//v.color;

		return true;
	}

	//! propagates distance values
	void improveSDF(unsigned int numIter) {
		for (unsigned int iter = 0; iter < numIter; iter++) {
			bool hasUpdate = false;
			for (size_t k = 0; k < getDimZ(); k++) {
				for (size_t j = 0; j < getDimY(); j++) {
					for (size_t i = 0; i < getDimX(); i++) {
						if (checkDistToNeighborAndUpdate(i, j, k)) {
							hasUpdate = true;
						}
					}
				}
			}

			if (!hasUpdate) break;
		}
	}

	//! bools checks if there is a neighbor with a smaller distance (+ the dist to the current voxel); if then it updates the distances and returns true
	bool checkDistToNeighborAndUpdate(size_t x, size_t y, size_t z) {
		//if ((*this)(x, y, z).sdf == -std::numeric_limits<float>::infinity()) return false;
		const bool origUnseen = (*this)(x, y, z).sdf == -std::numeric_limits<float>::infinity();

		bool foundBetter = false;
		for (size_t k = 0; k < 3; k++) {
			for (size_t j = 0; j < 3; j++) {
				for (size_t i = 0; i < 3; i++) {
					if (k == 1 && j == 1 && i == 1) continue;	//don't consider itself
					vec3ul n(x - 1 + i, y - 1 + j, z - 1 + k);
					if (isValidCoordinate(n.x, n.y, n.z)) {
						float d = (vec3f((float)x, (float)y, (float)z) - vec3f((float)n.x, (float)n.y, (float)n.z)).length();
						float nSDF = (*this)(n.x, n.y, n.z).sdf;
						int sgn = math::sign(nSDF);
						if (sgn != 0) {	//don't know that to do in this case...
							float dToN = nSDF + sgn*d;
							if (origUnseen && dToN > 0) continue;

							if (std::abs(dToN) < std::abs((*this)(x, y, z).sdf)) {
								(*this)(x, y, z).sdf = dToN;
								foundBetter = true;
							}
						}
					}
				}
			}
		}
		return foundBetter;

	}

	vec3f getSurfaceNormal(size_t x, size_t y, size_t z) const {
		float SDFx = (*this)(x + 1, y, z).sdf - (*this)(x - 1, y, z).sdf;
		float SDFy = (*this)(x, y + 1, z).sdf - (*this)(x, y - 1, z).sdf;
		float SDFz = (*this)(x, y, z + 1).sdf - (*this)(x, y, z - 1).sdf;
		if (SDFx == 0 && SDFy == 0 && SDFz == 0) {// Don't divide by zero!
			return vec3f(SDFx, SDFy, SDFz);
		}
		else {
			return vec3f(SDFx, SDFy, SDFz).getNormalized();
		}
	}

	mat3f getNormalCovariance(int x, int y, int z, int radius, float weightThreshold, float sdfThreshold) const {
		// Compute neighboring surface normals
		std::vector<vec3f> normals;
		for (int k = -radius; k <= radius; k++)
			for (int j = -radius; j <= radius; j++)
				for (int i = -radius; i <= radius; i++)
					if ((*this)(x + i, y + j, z + k).weight >= weightThreshold && std::abs((*this)(x + i, y + j, z + k).sdf) < sdfThreshold)
						normals.push_back(getSurfaceNormal(x + i, y + j, z + k));

		// Find covariance matrix
		float Ixx = 0; float Ixy = 0; float Ixz = 0;
		float Iyy = 0; float Iyz = 0; float Izz = 0;
		for (int i = 0; i < normals.size(); i++) {
			Ixx = Ixx + normals[i].x*normals[i].x;
			Ixy = Ixy + normals[i].x*normals[i].y;
			Ixz = Ixz + normals[i].x*normals[i].z;
			Iyy = Iyy + normals[i].y*normals[i].y;
			Iyz = Iyz + normals[i].y*normals[i].z;
			Izz = Izz + normals[i].z*normals[i].z;
		}
		float scale = 10.0f / ((float)normals.size()); // Normalize and upscale
		return mat3f(Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz)*scale;
	}

	Voxel getVoxel(const vec3f& worldPos) const {
		vec3i voxelPos = worldToVoxel(worldPos);

		if (isValidCoordinate(voxelPos.x, voxelPos.y, voxelPos.z)) {
			return (*this)(voxelPos.x, voxelPos.y, voxelPos.z);
		}
		else {
			return Voxel();
		}
	}


	float getDepthMin() const {
		return m_depthMin;
	}

	float getDepthMax() const {
		return m_depthMax;
	}

	float getTruncation(float d) const {
		return m_trunaction + d * m_truncationScale;
	}

	float getMaxTruncation() const {
		return getTruncation(m_depthMax);
	}
private:

	float frac(float val) const {
		return (val - floorf(val));
	}

	vec3f frac(const vec3f& val) const {
		return vec3f(frac(val.x), frac(val.y), frac(val.z));
	}

	BoundingBox3<int> computeFrustumBounds(const mat4f& intrinsic, const mat4f& rigidTransform, unsigned int width, unsigned int height) const {

		std::vector<vec3f> cornerPoints(8);

		cornerPoints[0] = depthToSkeleton(intrinsic, 0, 0, m_depthMin);
		cornerPoints[1] = depthToSkeleton(intrinsic, width - 1, 0, m_depthMin);
		cornerPoints[2] = depthToSkeleton(intrinsic, width - 1, height - 1, m_depthMin);
		cornerPoints[3] = depthToSkeleton(intrinsic, 0, height - 1, m_depthMin);

		cornerPoints[4] = depthToSkeleton(intrinsic, 0, 0, m_depthMax);
		cornerPoints[5] = depthToSkeleton(intrinsic, width - 1, 0, m_depthMax);
		cornerPoints[6] = depthToSkeleton(intrinsic, width - 1, height - 1, m_depthMax);
		cornerPoints[7] = depthToSkeleton(intrinsic, 0, height - 1, m_depthMax);

		BoundingBox3<int> box;
		for (unsigned int i = 0; i < 8; i++) {

			vec3f pl = math::floor(rigidTransform * cornerPoints[i]);
			vec3f pu = math::ceil(rigidTransform * cornerPoints[i]);
			box.include(worldToVoxel(pl));
			box.include(worldToVoxel(pu));
		}

		box.setMin(math::max(box.getMin(), 0));
		box.setMax(math::min(box.getMax(), vec3i((int)getDimX() - 1, (int)getDimY() - 1, (int)getDimZ() - 1)));

		return box;
	}

	static vec3f depthToSkeleton(const mat4f& intrinsic, unsigned int ux, unsigned int uy, float depth) {
		if (depth == 0.0f || depth == -std::numeric_limits<float>::infinity()) return vec3f(-std::numeric_limits<float>::infinity());

		float x = ((float)ux - intrinsic(0, 2)) / intrinsic(0, 0);
		float y = ((float)uy - intrinsic(1, 2)) / intrinsic(1, 1);

		return vec3f(depth*x, depth*y, depth);
	}

	static vec3f skeletonToDepth(const mat4f& intrinsics, const vec3f& p) {

		float x = (p.x * intrinsics(0, 0)) / p.z + intrinsics(0, 2);
		float y = (p.y * intrinsics(1, 1)) / p.z + intrinsics(1, 2);

		return vec3f(x, y, p.z);
	}

	float m_voxelSize;
	mat4f m_worldToGrid;
	mat4f m_gridToWorld; //inverse of worldToGrid
	float m_depthMin;
	float m_depthMax;


	float			m_trunaction;
	float			m_truncationScale;
	unsigned int	m_weightUpdate;
};