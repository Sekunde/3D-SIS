#include "stdafx.h"
#include "omp.h"
#include <random>

#include "GlobalAppState.h"
#include "LabelUtil.h"
#include "BBHelper.h"
#include "VoxelGrid.h"

struct ChunkInfo {
	bbox3i location;
	unsigned int numBoxesHitByImages;
	unsigned int numBoxes;
	float percentBoxSurfaceHit;
	std::string filename;
};
struct Chunk {
	DistanceField3f data;
	std::vector<BBInfo> bboxes;
	std::vector<float> bboxInChunk;
	std::vector<unsigned int> nearestImages;
	mat4f worldToChunk;
	std::vector<MaskType> masks;

	Chunk(const vec3ui& dim, float default) {
		data.allocate(dim);
		data.setValues(default);
	}
};

mat4f loadMatrixFromFile(const std::string& filename);
void loadSceneGridFromFile(const std::string& sceneFile, DistanceField3f& grid, bbox3f& sceneBounds, float& origVoxelSize);
int validateChunk(Chunk& chunk);
void loadFrameInfo(const std::string& framePath, std::vector<DepthImage32>& depths, std::vector<mat4f>& poses, mat4f& worldToGrid, unsigned int frameSkip = 1);
void loadFrameInfo(const std::string& framePath, std::vector<DepthImage32>& depths, std::vector<mat4f>& poses, mat4f& worldToGrid, const std::vector<unsigned int>& frameIds);
void visualizeChunk(const std::string& filename, const Chunk& chunk, bool bVisIndividual = false);
void saveChunkToFile(const std::string& filename, const Chunk& chunk);
void loadChunkFromFile(const std::string& filename, Chunk& chunk);
void processScene(const std::string& sceneFile, const std::string& bboxFile, const std::string& framePath,
	const LabelUtil& suncgLabelMap, const LabelUtil& nyuLabelMap, unsigned int maxNumNearestImages,
	const std::string& outputFile, float defaultValue);
void processSceneChunks(const std::string& sceneFile, const std::string& bboxFile, const std::string& framePath,
	const vec3i& chunkDim, const LabelUtil& suncgLabelMap, const LabelUtil& nyuLabelMap, unsigned int maxNumNearestImages,
	const std::string& outPrefix, unsigned int sampleFactor, float defaultValue, bool bRotate90);

TriMeshf visualizeColorGrid(const Grid3<vec3uc>& grid);

void loadGlobalAppState(const std::string& fileNameDescGlobalApp) {
	if (!util::fileExists(fileNameDescGlobalApp)) {
		throw MLIB_EXCEPTION("cannot find parameter file " + fileNameDescGlobalApp);
	}

	std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
	ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
	GlobalAppState::get().readMembers(parameterFileGlobalApp);
	GlobalAppState::get().print();
}

std::vector<std::string> readLinesFromFile(const std::string& filename) {
	std::ifstream ifs(filename); std::string line;
	if (!ifs.is_open()) throw MLIB_EXCEPTION("failed to open " + filename + " for read");
	std::vector<std::string> lines;
	while (std::getline(ifs, line)) lines.push_back(line);
	return lines;
}

void readCamerasFromFile(const std::string& filename, std::vector<std::vector<Cameraf>>& trajectories) {
	BinaryDataStreamFile s(filename, false);
	s >> trajectories; //s >> scanBounds;
	s.close();
}

void filterChunkForInsideBboxes(Chunk& chunk)
{
	std::vector<unsigned int> indicesToRemove;
	for (unsigned int i = 0; i < chunk.bboxInChunk.size(); i++) {
		if (chunk.bboxInChunk[i] < 1) indicesToRemove.push_back(i);
	}
	for (int i = (int)indicesToRemove.size() - 1; i >= 0; i--) {
		chunk.bboxes.erase(chunk.bboxes.begin() + indicesToRemove[i]);
		chunk.bboxInChunk.erase(chunk.bboxInChunk.begin() + indicesToRemove[i]);
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	try {

		const std::string fileNameDescGlobalApp = "zParameters.txt";
		loadGlobalAppState(fileNameDescGlobalApp);

		// params
		const GlobalAppState& gas = GlobalAppState::get();
		const std::string& sceneFileList = gas.s_sceneFileList;
		const std::string scenePath = gas.s_scenePath;
		const std::string aabbPath = gas.s_AABBPath;
		const std::string framePath = gas.s_framePath;
		const std::string outputPath = gas.s_outputPath;
		const std::string ext = ".scsdf";
		const unsigned int numTrajectories = gas.s_maxNumTrajectories;
		const unsigned int numAugment = gas.s_numAugment;
		const unsigned int sampleFactor = gas.s_sampleFactor;
		const unsigned int maxNumNearestImages = gas.s_maxNumNearestImages;
		const bool bGenerateTrain = gas.s_generateTrain;
		const bool bRotate90 = gas.s_bRotate90;
		const bool bGenerateTestOnly = gas.s_bGenerateTestOnly;
		const float defaultValue = -std::numeric_limits<float>::infinity();

		const vec3i chunkDim = gas.s_chunkDim;
		LabelUtil suncgLabelMap, nyuLabelMap;
		if (!gas.s_suncgLabelMapFile.empty()) {
			suncgLabelMap.init(gas.s_suncgLabelMapFile, gas.s_suncgLabelMapTo, gas.s_suncgLabelMapFrom);
			nyuLabelMap.init(gas.s_nyuLabelMapFile, gas.s_nyuLabelMapFrom, gas.s_nyuLabelMapTo);
		}

		if (!util::directoryExists(outputPath)) util::makeDirectory(outputPath);
		const std::vector<std::string> scenes = readLinesFromFile(sceneFileList);
		std::cout << "Found " << scenes.size() << " scenes" << std::endl;

		const int maxNumThreads = omp_get_max_threads();
		std::vector<std::vector<float>> avgNumBoxes(maxNumThreads);
		std::vector<std::vector<float>> avgPercentBoxesHitByImages(maxNumThreads);
		std::vector<std::vector<float>> avgPercentBoxSurfHit(maxNumThreads);
		std::vector<std::vector<unsigned int>> labelStats(maxNumThreads);

		for (int i = 0; i < (int)scenes.size(); i++) {		
			for (unsigned int t = 0; t < numTrajectories; t++) {
				for (unsigned int a = 0; a < numAugment + 1; a++) {
					
					const std::string suffix = (a == 0) ? "" : std::to_string(a - 1);
					const std::string filename = scenes[i] + "__" + std::to_string(t) + "__" + suffix;
					const std::string sceneFile = scenePath + "/" + filename + ext;
					const std::string bboxFile = aabbPath + "/" + filename + ".aabbs";
#ifdef SUNCG
					const std::string sceneFramePath = framePath + "/" + scenes[i] + "__" + std::to_string(t) + "__" + suffix;
#else
					const std::string sceneFramePath = framePath + "/" + scenes[i];
#endif


					if (bGenerateTestOnly) {
						if (!util::fileExists(sceneFile))
							continue;
					}
					else {
						if (!util::fileExists(sceneFile) || !util::fileExists(bboxFile))
							continue;
					}

					const int thread = omp_get_thread_num();
					const std::string outPrefix = outputPath + "/" + filename + "_";

					if (bGenerateTrain) {
						if (util::fileExists(outPrefix + "0.chunk")) {
							if (i % 20 == 0)
								std::cout << scenes[i] << " exists, skipping" << std::endl;
							continue;
						}

						processSceneChunks(sceneFile, bboxFile, sceneFramePath, chunkDim, suncgLabelMap, nyuLabelMap,
							maxNumNearestImages, outPrefix, sampleFactor, defaultValue, bRotate90);
						
					}
					else {
						const std::string outputFile = outputPath + "\\" + filename + ".scene";
						if (util::fileExists(outputFile)) {
							std::cout << scenes[i] << " exists, skipping";
							continue;
						}
						processScene(sceneFile, bboxFile, sceneFramePath, suncgLabelMap, nyuLabelMap, maxNumNearestImages, outputFile, defaultValue);
					}

					if (i % 20 == 0)
						std::cout << "[ " << (i + 1) << " | " << scenes.size() << " ] " << scenes[i] << " (" << (t*numAugment + a + 1) << " | " << (numTrajectories * (numAugment + 1)) << ")" << std::endl;
				} // augment
			} // trajectories
		}  // scenes
		std::cout << std::endl;
	}
	catch (const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	return 0;
}

void loadSceneGridFromFile(const std::string& sceneFile, DistanceField3f& grid, bbox3f& sceneBounds, float& origVoxelSize)
{
	std::ifstream ifs(sceneFile, std::ios::binary);
	if (!ifs.is_open()) throw MLIB_EXCEPTION("failed to open " + sceneFile + " for read");
	//metadata
	UINT64 dimX, dimY, dimZ;
	ifs.read((char*)&dimX, sizeof(UINT64));
	ifs.read((char*)&dimY, sizeof(UINT64));
	ifs.read((char*)&dimZ, sizeof(UINT64));
	ifs.read((char*)&origVoxelSize, sizeof(float));
	vec3f bmin, bmax;
	ifs.read((char*)&bmin.x, sizeof(float));
	ifs.read((char*)&bmin.y, sizeof(float));
	ifs.read((char*)&bmin.z, sizeof(float));
	ifs.read((char*)&bmax.x, sizeof(float));
	ifs.read((char*)&bmax.y, sizeof(float));
	ifs.read((char*)&bmax.z, sizeof(float));
	sceneBounds = bbox3f(bmin, bmax);
	//dense data
	grid.allocate(dimX, dimY, dimZ);
	ifs.read((char*)grid.getData(), sizeof(float)*grid.getNumElements());
	ifs.close();
}

float computeIntersectionVolume(const bbox3f& b0, const bbox3f& b1)
{
	vec3f min, max;
	min.x = std::max(b0.getMinX(), b1.getMinX());
	min.y = std::max(b0.getMinY(), b1.getMinY());
	min.z = std::max(b0.getMinZ(), b1.getMinZ());
	max.x = std::min(b0.getMaxX(), b1.getMaxX());
	max.y = std::min(b0.getMaxY(), b1.getMaxY());
	max.z = std::min(b0.getMaxZ(), b1.getMaxZ());
	return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
}

// if b0 inside of b1
bool isInside(const bbox3f& b0, const bbox3f& b1)
{
	return (b0.getMinX() >= b1.getMinX() && b0.getMinY() >= b1.getMinY() && b0.getMinZ() >= b1.getMinZ()
		&& b0.getMaxX() <= b1.getMaxX() && b0.getMaxY() <= b1.getMaxY() && b0.getMaxZ() <= b1.getMaxZ());
}

template<typename T>
vec3<T> getCoord(const vec3<T>& coord, unsigned int rotId)
{
	return getCoord(coord.x, coord.y, coord.z, rotId);
}

template<typename T>
vec3<T> getCoord(T x, T y, T z, unsigned int rotId)
{
	switch (rotId) {
	case 4:
	case 0:
		return vec3<T>(x, y, z);
		break;
	case 1:
		return vec3<T>(z, y, -x);
		break;
	case 2:
		return vec3<T>(-x, y, -z);
		break;
	case 3:
		return vec3<T>(-z, y, x);
		break;
	default:
		throw MLIB_EXCEPTION("invalid rotation id: " + std::to_string(rotId));
	};
}

MaskType rotateMask(const MaskType& mask, unsigned int rotId)
{
	if (rotId == 0) return mask;
	const vec3i dims(mask.getDimensions());

	vec3i rdims = getCoord(dims, rotId);
	vec3i roffset = vec3i(0, 0, 0);
	for (unsigned int i = 0; i < 3; i++) {
		if (i == 1) continue;
		if (rdims[i] < -0.01f) roffset[i] = -rdims[i] - 1;
	}
	rdims = math::abs(rdims);
	MaskType rotated(rdims);
	for (const auto& v : mask) {
		const vec3i coord = getCoord((int)v.x, (int)v.y, (int)v.z, rotId) + roffset;
		rotated(coord) = v.value;
	}
	return rotated;
}

void extractChunk(const DistanceField3f& grid, const std::vector<BBInfo>& bboxes, const vec3i& start, unsigned int rotId, Chunk& chunk)
{
	MLIB_ASSERT(chunk.data.getNumElements() > 0);
	const auto dim = chunk.data.getDimensions();
	const auto gridDim = grid.getDimensions();
	vec3i rotGridDim = getCoord(vec3i(gridDim), rotId);
	vec3i offsetGridDim = vec3i(0, 0, 0);
	vec3i invOffsetGridDim = vec3i(0, 0, 0);
	for (unsigned int i = 0; i < 3; i++) {
		if (i == 1) continue;
		if (rotGridDim[i] < -0.01f) offsetGridDim[i] = -rotGridDim[i];
		else invOffsetGridDim[i] = (int)gridDim[i];
	}
	rotGridDim = math::abs(rotGridDim);

	// check for bboxes inside 
	bbox3f chunkBox(start, start + dim);
	float chunkVol = (chunkBox.getMaxX() - chunkBox.getMinX()) * (chunkBox.getMaxY() - chunkBox.getMinY()) * (chunkBox.getMaxZ() - chunkBox.getMinZ());
	for (unsigned int i = 0; i < bboxes.size(); i++) {
		vec3f bb0 = getCoord(bboxes[i].aabb.getMin(), rotId) + vec3f(offsetGridDim);
		vec3f bb1 = getCoord(bboxes[i].aabb.getMax(), rotId) + vec3f(offsetGridDim);
		bbox3f bb(math::min(bb0, bb1), math::max(bb0, bb1));
		if (chunkBox.intersects(bb)) {
			// intersection volume with chunk 
			float vol = computeIntersectionVolume(chunkBox, bb);
			float boxVol = (bb.getMaxX() - bb.getMinX()) * (bb.getMaxY() - bb.getMinY()) * (bb.getMaxZ() - bb.getMinZ());
			MLIB_ASSERT(vol > 0 && (vol - chunkVol) <= 0.001f);
			float portion = 1.0f;
			if (!isInside(bboxes[i].aabb, chunkBox)) portion = vol / boxVol;
			//if (!isInside(bb, chunkBox)) continue;

			//chunk.bboxIdxes.push_back(i);
			chunk.bboxInChunk.push_back(portion);
			// bbox in chunk space
			chunk.bboxes.push_back(bboxes[i]);
			auto& chunkbb = chunk.bboxes.back();
			chunkbb.aabb = bb;
			chunkbb.aabb.transform(mat4f::translation(-start));
#ifdef SUNCG
			chunkbb.aabbCanonical.transform(mat4f::translation(-start));
#endif
			//mask
			chunkbb.mask = rotateMask(bboxes[i].mask, rotId);
		}
	}

	for (unsigned int z = 0; z < dim.z; z++) {
		for (unsigned int y = 0; y < dim.y; y++) {
			for (unsigned int x = 0; x < dim.x; x++) {
				const vec3i loc = -offsetGridDim + start + vec3i(x, y, z); // rot grid space
				vec3i coord = getCoord(loc, 4 - rotId);   // unrot
				if (rotId == 1) coord.x -= 1;//coord += invOffsetGridDim - 1;
				else if (rotId == 2) { coord.x -= 1; coord.z -= 1; }
				else if (rotId == 3) coord.z -= 1;
				if (grid.isValidCoordinate(coord))
					chunk.data(x, y, z) = grid(coord);
			} // x
		} // y
	} // z

	//visualizeChunk("debug/test.ply", chunk);
	//int a = 5;
}

void saveChunkToFile(const std::string& filename, const Chunk& chunk)
{
	std::ofstream ofs(filename, std::ios::binary);
	if (!ofs.is_open()) throw MLIB_EXCEPTION("failed to open file " + filename + " for write");
	//metadata
	UINT64 dimX = chunk.data.getDimX(), dimY = chunk.data.getDimY(), dimZ = chunk.data.getDimZ();
	ofs.write((const char*)&dimX, sizeof(UINT64));
	ofs.write((const char*)&dimY, sizeof(UINT64));
	ofs.write((const char*)&dimZ, sizeof(UINT64));
	//chunk data
	ofs.write((const char*)chunk.data.getData(), sizeof(float)*chunk.data.getNumElements());
	//bboxes
	unsigned int numBboxes = (unsigned int)chunk.bboxes.size();
	ofs.write((const char*)&numBboxes, sizeof(unsigned int));
	for (unsigned int i = 0; i < numBboxes; i++) {
		const BBInfo& bbInfo = chunk.bboxes[i];
		unsigned int labelId = (unsigned int)bbInfo.labelId;
		const vec3f bboxMin = bbInfo.aabb.getMin();
		const vec3f bboxMax = bbInfo.aabb.getMax();
		ofs.write((const char*)bboxMin.array, sizeof(vec3f));
		ofs.write((const char*)bboxMax.array, sizeof(vec3f));
		ofs.write((const char*)&labelId, sizeof(unsigned int));
	}
	//masks
	unsigned int numMasks = numBboxes;
	ofs.write((const char*)&numMasks, sizeof(unsigned int)); // redundant
	for (unsigned int i = 0; i < numMasks; i++) {
		unsigned int labelId = (unsigned int)chunk.bboxes[i].labelId;
		ofs.write((const char*)&labelId, sizeof(unsigned int)); // redundant
		const MaskType& mask = chunk.bboxes[i].mask;
		UINT64 dimX = mask.getDimX(), dimY = mask.getDimY(), dimZ = mask.getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)mask.getData(), sizeof(unsigned short)*mask.getNumElements());
	}
	//info
	//portion of bboxes inside volume
	ofs.write((const char*)&numBboxes, sizeof(unsigned int));
	ofs.write((const char*)chunk.bboxInChunk.data(), sizeof(float)*numBboxes);
	// image info
	ofs.write((const char*)chunk.worldToChunk.getData(), sizeof(mat4f));
	unsigned int numImages = (unsigned int)chunk.nearestImages.size();
	ofs.write((const char*)&numImages, sizeof(unsigned int));
	ofs.write((const char*)chunk.nearestImages.data(), sizeof(unsigned int)*numImages);
	ofs.close();

#ifdef SUNCG
	//canonical bboxes
	ofs.write((const char*)&numBboxes, sizeof(unsigned int));
	for (unsigned int i = 0; i < numBboxes; i++) {
		const BBInfo& bbInfo = chunk.bboxes[i];
		unsigned int labelId = (unsigned int)bbInfo.labelId;
		const vec3f bboxMin = bbInfo.aabbCanonical.getMin();
		const vec3f bboxMax = bbInfo.aabbCanonical.getMax();
		ofs.write((const char*)bboxMin.array, sizeof(vec3f));
		ofs.write((const char*)bboxMax.array, sizeof(vec3f));
		ofs.write((const char*)&bbInfo.angleCanonical, sizeof(float));
		ofs.write((const char*)&labelId, sizeof(unsigned int));
		const MaskType& mask = bbInfo.maskCanonical;
		UINT64 dimX = mask.getDimX(), dimY = mask.getDimY(), dimZ = mask.getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)mask.getData(), sizeof(unsigned short)*mask.getNumElements());
	}
#endif
}


void loadChunkFromFile(const std::string& filename, Chunk& chunk)
{
	std::ifstream ifs(filename, std::ios::binary);
	if (!ifs.is_open())
		throw MLIB_EXCEPTION("failed to open file " + filename + " for read");
	//metadata
	UINT64 dimX, dimY, dimZ;
	ifs.read((char*)&dimX, sizeof(UINT64));
	ifs.read((char*)&dimY, sizeof(UINT64));
	ifs.read((char*)&dimZ, sizeof(UINT64));
	chunk.data.allocate(dimX, dimY, dimZ);
	//chunk data
	ifs.read((char*)chunk.data.getData(), sizeof(float)*chunk.data.getNumElements());
	//bboxes
	unsigned int numBboxes;
	ifs.read((char*)&numBboxes, sizeof(unsigned int));
	chunk.bboxes.resize(numBboxes);
	for (unsigned int i = 0; i < numBboxes; i++) {
		BBInfo& bbInfo = chunk.bboxes[i];
		unsigned int labelId;
		vec3f bboxMin;
		vec3f bboxMax;
		ifs.read((char*)bboxMin.array, sizeof(vec3f));
		ifs.read((char*)bboxMax.array, sizeof(vec3f));
		ifs.read((char*)&labelId, sizeof(unsigned int));
		bbInfo.aabb = bbox3f(bboxMin, bboxMax);
		bbInfo.labelId = (unsigned short)labelId;
	}
	//masks
	unsigned int numMasks;
	ifs.read((char*)&numMasks, sizeof(unsigned int)); // redundant
	MLIB_ASSERT(numMasks == numBboxes);
	for (unsigned int i = 0; i < numMasks; i++) {
		unsigned int labelId;
		ifs.read((char*)&labelId, sizeof(unsigned int)); // redundant
		auto& mask = chunk.bboxes[i].mask;
		UINT64 dimX, dimY, dimZ;
		ifs.read((char*)&dimX, sizeof(UINT64));
		ifs.read((char*)&dimY, sizeof(UINT64));
		ifs.read((char*)&dimZ, sizeof(UINT64));
		mask.allocate(dimX, dimY, dimZ);
		ifs.read((char*)mask.getData(), sizeof(unsigned short)*mask.getNumElements());
	}
	//info
	//portion of bboxes inside volume
	ifs.read((char*)&numBboxes, sizeof(unsigned int));
	chunk.bboxInChunk.resize(numBboxes);
	ifs.read((char*)chunk.bboxInChunk.data(), sizeof(float)*numBboxes);
	// image info
	ifs.read((char*)chunk.worldToChunk.getData(), sizeof(mat4f));
	unsigned int numImages = (unsigned int)chunk.nearestImages.size();
	ifs.read((char*)&numImages, sizeof(unsigned int));
	if (numImages > 0) {
		chunk.nearestImages.resize(numImages);
		ifs.read((char*)chunk.nearestImages.data(), sizeof(unsigned int)*numImages);
	}
	ifs.close();
}

void mapLabels(std::vector<BBInfo>& bboxes, const LabelUtil& suncgLabelMap, const LabelUtil& nyuLabelMap)
{
	for (auto& bbox : bboxes) {
		std::string label;
		bool bValid = suncgLabelMap.getLabelForId(bbox.labelId, label);
		if (!bValid)
			throw MLIB_EXCEPTION("failed to find label for suncg id " + std::to_string(bbox.labelId));
		unsigned short id;
		bValid = nyuLabelMap.getIdForLabel(label, id);
		if (!bValid)
			throw MLIB_EXCEPTION("failed to find id for nyu label " + label);
		bbox.labelId = id;
	}
}

void visualizeChunk(const std::string& filename, const Chunk& chunk, bool bVisIndividual /*= false*/)
{
	BinaryGrid3 bg = chunk.data.computeBinaryGrid(1.0f);
	MeshIOf::saveToFile(filename, TriMeshf(bg).computeMeshData());

	const float radius = 0.5f; // grid space
	bg.clearVoxels();
	BinaryGrid3 bgComplete(bg.getDimensions());
	MeshDataf meshBoxes;
	for (unsigned int i = 0; i < chunk.bboxes.size(); i++) {
		const MaskType& mask = chunk.bboxes[i].mask;
		const RGBColor c = RGBColor::colorPalette(i);
		// aabb
		MeshDataf meshBbox;
		for (const LineSegment3f& e : chunk.bboxes[i].aabb.getEdges()) meshBbox.merge(Shapesf::cylinder(e.p0(), e.p1(), radius, 10, 10, c).computeMeshData());
		meshBoxes.merge(meshBbox);
		// mask
		BinaryGrid3 curBg(bg.getDimensions());
		const vec3f maskDims(mask.getDimensions());

		for (unsigned int z = 0; z < mask.getDimZ(); z++) {
			for (unsigned int y = 0; y < mask.getDimY(); y++) {
				for (unsigned int x = 0; x < mask.getDimX(); x++) {
					const auto val = mask(x, y, z);
					if (val > 0) {
						const vec3i coordWorld = vec3i(x, y, z) + math::floor(chunk.bboxes[i].aabb.getMin());
						if (bg.isValidCoordinate(coordWorld)) {
							if (val == 1) {
								bg.setVoxel(coordWorld);
								curBg.setVoxel(coordWorld);
								//bgComplete.setVoxel(coordWorld); // don't need for visualze
							}
							else {
								bgComplete.setVoxel(coordWorld);
							}
						}
						else {
							if (std::fabs(chunk.bboxInChunk[i] - 1.0f) < 0.001f)
								throw MLIB_EXCEPTION("[visualizeChunk] bad world coord for mask element");
						}
					}
				}  // z
			}  // y
		}  // x

		if (bVisIndividual) {
			const unsigned int classId = chunk.bboxes[i].labelId;
			MeshIOf::saveToFile(util::removeExtensions(filename) + "_" + std::to_string(i) + "_class" + std::to_string(classId) + "_BBOXES.ply", meshBbox);
			MeshIOf::saveToFile(util::removeExtensions(filename) + "_" + std::to_string(i) + "_class" + std::to_string(classId) + "_MASKS.ply", TriMeshf(curBg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		}
	}
	MeshIOf::saveToFile(util::removeExtensions(filename) + "_BBOXES.ply", meshBoxes);
	MeshDataf meshMask = TriMeshf(bg, mat4f::identity(), false, vec4f(1.0f, 0.0f, 0.0f, 1.0f)).computeMeshData();
	meshMask.merge(TriMeshf(bgComplete, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
	MeshIOf::saveToFile(util::removeExtensions(filename) + "_MASKS.ply", meshMask);
	for (unsigned int i = 0; i < chunk.bboxInChunk.size(); i++) {
		std::cout << "[ box " << i << " ] overlap " << chunk.bboxInChunk[i] << std::endl;
	}
}

mat4f loadMatrixFromFile(const std::string& filename) {
	mat4f m;
	std::ifstream ifs(filename);
	ifs >> m._m00 >> m._m01 >> m._m02 >> m._m03;
	ifs >> m._m10 >> m._m11 >> m._m12 >> m._m13;
	ifs >> m._m20 >> m._m21 >> m._m22 >> m._m23;
	ifs >> m._m30 >> m._m31 >> m._m32 >> m._m33;
	ifs.close();
	return m;
}

void loadFrameInfo(const std::string& framePath, std::vector<DepthImage32>& depths, std::vector<mat4f>& poses, mat4f& worldToGrid, unsigned int frameSkip /*= 1*/) {
	depths.clear();
	poses.clear();

	const std::string depthPath = framePath + "/depth";
	const std::string posePath = framePath + "/pose";
	worldToGrid = loadMatrixFromFile(framePath + "/world2grid.txt");

	Directory dir(posePath);
	const auto& files = dir.getFiles();
	if (files.empty()) throw MLIB_EXCEPTION("failed to find poses: " + framePath);
	depths.reserve(files.size());
	poses.reserve(files.size());
	for (unsigned int f = 0; f < files.size()*frameSkip; f += frameSkip) {
		const std::string depthFile = depthPath + "/" + std::to_string(f) + ".png";
		const std::string poseFile = posePath + "/" + std::to_string(f) + ".txt";
		if (!util::fileExists(depthFile)) {
			//throw MLIB_EXCEPTION("no depth file for pose file: " + poseFile);
			depths.push_back(DepthImage32());
			poses.push_back(mat4f::zero(-std::numeric_limits<float>::infinity()));
		}
		else {
			DepthImage16 depth16;
			FreeImageWrapper::loadImage(depthFile, depth16);
			depth16.resize(80, 60);
			depths.push_back(DepthImage32(depth16));
			poses.push_back(loadMatrixFromFile(poseFile));
		}
	}

}

void loadFrameInfo(const std::string& framePath, std::vector<DepthImage32>& depths, std::vector<mat4f>& poses, mat4f& worldToGrid, const std::vector<unsigned int>& frameIds) {
	depths.clear();
	poses.clear();
	depths.reserve(frameIds.size());
	poses.reserve(frameIds.size());
	const std::string depthPath = framePath + "/depth";
	const std::string posePath = framePath + "/pose";
	for (const auto f : frameIds) {
		const std::string depthFile = depthPath + "/" + std::to_string(f) + ".png";
		const std::string poseFile = posePath + "/" + std::to_string(f) + ".txt";
		if (!util::fileExists(depthFile)) throw MLIB_EXCEPTION("no depth file for pose file: " + poseFile);
		DepthImage16 depth16; FreeImageWrapper::loadImage(depthFile, depth16);
		depth16.resize(80, 60);
		depths.push_back(DepthImage32(depth16));
		poses.push_back(loadMatrixFromFile(poseFile));
	}

}


vec2ui findNearestImages(const DistanceField3f& grid, const std::vector<BBInfo>& bboxes, const std::vector<float>& bboxInChunk, const mat4f& sampleGridToWorld, float voxelSize, const vec3ui& blockDim,
	const std::vector<DepthImage32>& depths, const std::vector<mat4f>& poses, unsigned int numNearestImages, std::vector<unsigned int>& nearestImages) {

#ifdef SUNCG
	mat4f intrinsic = mat4f(554.256f, 0.0f, 319.5f, 0.0f,
		0.0f, 554.256f, 239.5f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
#else
	mat4f intrinsic = mat4f(577.870605f, 0.0f, 319.5f, 0.0f,
		0.0f, 577.870605f, 239.5f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
#endif
	intrinsic(0, 0) *= 80.0f / 640.0f;
	intrinsic(1, 1) *= 60.0f / 480.0f;
	intrinsic(0, 2) *= (80.0f - 1.0f) / (640.0f - 1.0f);
	intrinsic(1, 2) *= (60.0f - 1.0f) / (480.0f - 1.0f);

	//todo this is a stupid hack
	std::unordered_set<unsigned int> validClasses = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39 };

	nearestImages.clear();
	VoxelGrid dummy(blockDim, sampleGridToWorld, voxelSize, 0.4f, 4.0f);
	auto comp = [](const std::pair<unsigned int, std::pair<std::unordered_map<unsigned int, std::unordered_set<vec3i>>, unsigned int>>& a, const std::pair<unsigned int, std::pair<std::unordered_map<unsigned int, std::unordered_set<vec3i>>, unsigned int>>& b) {
		if (a.second.first.size() != b.second.first.size())
			return a.second.first.size() < b.second.first.size();
		return a.second.second < b.second.second;
	};
	std::priority_queue<std::pair<unsigned int, std::pair<std::unordered_map<unsigned int, std::unordered_set<vec3i>>, unsigned int>>, std::vector<std::pair<unsigned int, std::pair<std::unordered_map<unsigned int, std::unordered_set<vec3i>>, unsigned int>>>, decltype(comp)> best(comp);
	for (unsigned int f = 0; f < poses.size(); f++) {
		if (poses[f](0, 0) == -std::numeric_limits<float>::infinity()) continue;
		const DepthImage32& depth = depths[f];
		std::pair<std::unordered_map<unsigned int, std::unordered_set<vec3i>>, unsigned int> validBoxHit = dummy.countProjectionsInBoxes(intrinsic, poses[f], depth, bboxes, bboxInChunk); // for each box hit, which voxels
		if (!validBoxHit.first.empty()) {
			best.push(std::make_pair(f, validBoxHit));
		}
	}

	if (best.empty()) {
		return vec2ui(0, 0);
	}
	std::vector<unsigned int> geoCountPerInstance(bboxes.size(), 0);
	for (unsigned int i = 0; i < bboxes.size(); i++) {
		for (const auto& v : bboxes[i].mask) {
			if (v.value != 0) geoCountPerInstance[i]++;
		}
	}

	// greedy max box coverage
	const unsigned int num = std::min((unsigned int)best.size(), numNearestImages);
	nearestImages.push_back(best.top().first);
	std::unordered_map<unsigned int, std::unordered_set<vec3i>> boxSet = best.top().second.first;

	best.pop();
	if (!best.empty()) {
		std::vector< std::pair<unsigned int, std::pair<std::unordered_map<unsigned int, std::unordered_set<vec3i>>, unsigned int>> > tmp;
		size_t numBest = best.size();
		for (unsigned int i = 0; i < numBest; i++) {
			const auto& top = best.top();
			tmp.push_back(top);
			best.pop();
		}

		while (nearestImages.size() < num) {
			float maxInstCov = 0.0f;
			unsigned int max = 0, maxIdx = 0;
			for (unsigned int i = 0; i < tmp.size(); i++) {
				std::unordered_map<unsigned int, std::unordered_set<vec3i>> current = boxSet;
				for (const auto& instVoxels : tmp[i].second.first) {
					auto it = current.find(instVoxels.first);
					if (it != current.end())
						it->second.insert(instVoxels.second.begin(), instVoxels.second.end());
					else
						current[instVoxels.first] = instVoxels.second;
				}
				float curInstCov = 0.0f;
				unsigned int norm = 0;
				for (unsigned int c = 0; c < bboxes.size(); c++) {
					if (bboxInChunk[c] < IMAGE_FIND_BBOX_INTERSECT_THRESH || validClasses.find(bboxes[c].labelId) == validClasses.end())
						continue;
					norm++;
					const auto it = current.find(c);
					if (it != current.end())
						curInstCov += (float)it->second.size() / (float)geoCountPerInstance[c];
				}
				curInstCov /= (float)norm;
				if (curInstCov > maxInstCov) {
					maxInstCov = curInstCov;
					maxIdx = i;
				}
			}

			for (const auto& instVoxels : tmp[maxIdx].second.first) {
				auto it = boxSet.find(instVoxels.first);
				if (it != boxSet.end())
					it->second.insert(instVoxels.second.begin(), instVoxels.second.end());
				else
					boxSet[instVoxels.first] = instVoxels.second;
			}
			nearestImages.push_back(tmp[maxIdx].first);
			tmp.erase(tmp.begin() + maxIdx);
			if (tmp.empty()) break;
		}
	}
	std::unordered_set<vec3i> voxelsHit;
	for (const auto& inst : boxSet) voxelsHit.insert(inst.second.begin(), inst.second.end());

	return vec2ui((unsigned int)boxSet.size(), (unsigned int)voxelsHit.size());
}


void processSceneChunks(const std::string& sceneFile, const std::string& bboxFile, const std::string& framePath,
	const vec3i& chunkDim, const LabelUtil& suncgLabelMap, const LabelUtil& nyuLabelMap, unsigned int maxNumNearestImages,
	const std::string& outPrefix, unsigned int sampleFactor, float defaultValue, bool bRotate90)
{
	// todo: move params out
	const int scenePad = 16;
	const int pad = 6;
	const int offset = scenePad - pad;

	// load scene
	DistanceField3f grid; float voxelSize; bbox3f sceneBounds;
	loadSceneGridFromFile(sceneFile, grid, sceneBounds, voxelSize);
	const vec3ul sceneDim = grid.getDimensions();
	// load bboxes/masks
	std::vector<BBInfo> bboxes;
	BBHelper::readAABBsFromFile(bboxFile, bboxes);

	// double check masks
	for (unsigned int i = 0; i < bboxes.size(); i++) {
		const MaskType& m = bboxes[i].mask;
		if (m.getDimX() == 0 || m.getDimY() == 0 || m.getDimZ() == 0) {
			std::cout << "warning: empty masks in scene " << sceneFile << std::endl;
			break;
		}
	}

	if (suncgLabelMap.isInitialized())
		mapLabels(bboxes, suncgLabelMap, nyuLabelMap);
	// load frames
	std::vector<DepthImage32> depths; std::vector<mat4f> poses; mat4f worldToGrid;


#ifdef SUNCG
	loadFrameInfo(framePath, depths, poses, worldToGrid);
	vec3ul voxelDim = math::round(sceneBounds.getExtent() / voxelSize);
	voxelDim += scenePad * 2;
	worldToGrid = mat4f::scale(1.0f / voxelSize) * mat4f::translation(-sceneBounds.getMin() + scenePad*voxelSize);
#else
	const unsigned frameSkip = 20;
	loadFrameInfo(framePath, depths, poses, worldToGrid, frameSkip); //TODO fix this hack
#endif


	const unsigned int numRots = bRotate90 ? 4 : 1;


	std::vector<ChunkInfo> infos;
	const vec3f center = vec3f(grid.getDimensions()) * 0.5f;
	//for (unsigned int rot = 1; rot < numRots; rot++) {
	for (unsigned int rot = 0; rot < numRots; rot++) {
		const auto newDim = getCoord(vec3i(grid.getDimensions()), rot);
		vec3f newCenter = getCoord(center, rot);
		if (newCenter.x < 0) newCenter.x -= newDim.x;
		if (newCenter.z < 0) newCenter.z -= newDim.z;
		const mat4f rotationGrid = mat4f::translation(newCenter) * mat4f::rotationY(90.0f * rot) * mat4f::translation(-center);
		const vec3i endDim = math::abs(getCoord(vec3i(sceneDim), rot));
		for (unsigned int z = offset; z < endDim.z - offset; z += sampleFactor) {
			unsigned int y = scenePad; {
				for (unsigned int x = offset; x < endDim.x - offset; x += sampleFactor) {
					Chunk chunk(chunkDim, defaultValue);
					extractChunk(grid, bboxes, vec3ui(x, y, z), rot, chunk);
					if (chunk.bboxes.empty()) continue;

					const mat4f sampleGridToWorld = mat4f::translation(-vec3f((float)x, (float)y, (float)z)) * rotationGrid * worldToGrid;  // world -> grid -> sample grid
					chunk.worldToChunk = sampleGridToWorld.getInverse();
					vec2ui stat = findNearestImages(chunk.data, chunk.bboxes, chunk.bboxInChunk, sampleGridToWorld, voxelSize, chunkDim, depths, poses, maxNumNearestImages, chunk.nearestImages);
#ifndef SUNCG
					for (auto& im : chunk.nearestImages) im *= frameSkip;
#endif
					if (chunk.nearestImages.empty()) continue;

					const std::string filename = outPrefix + std::to_string(infos.size()) + ".chunk";

					// sanity check
					bool empty = true;
					for (const auto& v : chunk.data) {
						if (std::fabs(v.value) <= 1) {
							empty = false;
							break;
						}
					}
					if (empty) {
						std::cerr << "warning: found empty chunk data, skipping" << std::endl;
						continue;
					}
					int validity = validateChunk(chunk);
					if (validity > -1) {
						saveChunkToFile(filename, chunk);
						unsigned int numBoxSurf = 0;
						for (const auto& bb : chunk.bboxes) {
							for (const auto& v : bb.mask) {
								if (v.value == 1) numBoxSurf++;
							}
						}
						float percentBoxSurface = (float)stat[1] / (float)numBoxSurf;
						infos.push_back(ChunkInfo{ bbox3i(vec3i(x, y, z), vec3i(x, y, z) + chunkDim), stat[0], (unsigned int)chunk.bboxes.size(), percentBoxSurface, filename });
					}

				} // x
			} // y
		} // z
	} // rots

}

void processScene(const std::string& sceneFile, const std::string& bboxFile, const std::string& framePath,
	const LabelUtil& suncgLabelMap, const LabelUtil& nyuLabelMap, unsigned int maxNumNearestImages,
	const std::string& outputFile, float defaultValue)
{

	// todo: move params out
	const int scenePad = 16;
	const int pad = 6;
	const int offset = scenePad - pad;
	const bool bGenerateTestOnly = GlobalAppState::get().s_bGenerateTestOnly;

	// load scene
	DistanceField3f grid; float voxelSize; bbox3f sceneBounds;
	loadSceneGridFromFile(sceneFile, grid, sceneBounds, voxelSize);
	const vec3ul sceneDim = grid.getDimensions();
	// load bboxes/masks
	std::vector<BBInfo> bboxes;
	if (!bGenerateTestOnly) {
		BBHelper::readAABBsFromFile(bboxFile, bboxes);
		if (suncgLabelMap.isInitialized()) mapLabels(bboxes, suncgLabelMap, nyuLabelMap);
	}
	// load frames
	std::vector<DepthImage32> depths; std::vector<mat4f> poses; mat4f worldToGrid;
	loadFrameInfo(framePath, depths, poses, worldToGrid);
	vec3ul voxelDim = math::round(sceneBounds.getExtent() / voxelSize);
	voxelDim += scenePad * 2;
	worldToGrid = mat4f::scale(1.0f / voxelSize) * mat4f::translation(-sceneBounds.getMin() + scenePad*voxelSize);

	// crop off some padding here
	Chunk scene(grid.getDimensions() - vec3i(2 * offset, scenePad + offset, 2 * offset), defaultValue);
	for (unsigned int z = offset; z < grid.getDimZ() - offset; z++) {
		for (unsigned int y = scenePad; y < grid.getDimY() - offset; y++) {
			for (unsigned int x = offset; x < grid.getDimX() - offset; x++) {
				scene.data(x - offset, y - scenePad, z - offset) = grid(x, y, z);
			}  // x
		}  // y
	}  // z

	bbox3f sceneBox(vec3f((float)offset, (float)scenePad, (float)offset), vec3f(grid.getDimensions()) - vec3f((float)offset, (float)offset, (float)offset));
	for (unsigned int i = 0; i < bboxes.size(); i++) {
		// intersection volume with chunk 
		float vol = computeIntersectionVolume(sceneBox, bboxes[i].aabb);
		float boxVol = (bboxes[i].aabb.getMaxX() - bboxes[i].aabb.getMinX()) * (bboxes[i].aabb.getMaxY() - bboxes[i].aabb.getMinY()) * (bboxes[i].aabb.getMaxZ() - bboxes[i].aabb.getMinZ());
		float portion = 1.0f;
		if (!isInside(bboxes[i].aabb, sceneBox)) portion = vol / boxVol;

		scene.bboxInChunk.push_back(portion);
		// bbox in cropped scene space
		scene.bboxes.push_back(bboxes[i]);
		auto& bb = scene.bboxes.back();
		bb.aabb.transform(mat4f::translation(-sceneBox.getMin()));
		bb.mask = bboxes[i].mask;
	}

	const mat4f gridToWorld = mat4f::translation(-sceneBox.getMin()) * worldToGrid;  // world -> grid -> cropped grid
	scene.worldToChunk = gridToWorld.getInverse();
	int validity = validateChunk(scene);
	if (validity > -1) {
		saveChunkToFile(outputFile, scene);
		std::cout << "saved to:" << outputFile << std::endl;
	}
	else {
		std::cout << "warning: found invalid scene: " << outputFile << std::endl;
	}
}

// -1: bad, discard
// 0: good
// 1: fixed
int validateChunk(Chunk& chunk)
{
	bool ok = true;
	const vec3ui dim = chunk.data.getDimensions();
	for (unsigned int i = 0; i < chunk.bboxes.size(); i++) {
		const float val = chunk.bboxInChunk[i];
		if (std::fabs(val - 1.0f) > 0.001f) continue;
		bbox3f& bbox = chunk.bboxes[i].aabb;
		const vec3i bboxMin = math::floor(bbox.getMin());
		const vec3i bboxMax = math::ceil(bbox.getMax());
		const vec3i extent2 = bboxMax - bboxMin;
		vec3i maskDim = chunk.bboxes[i].mask.getDimensions();
		if (maskDim != extent2) {
			return -1;
		}
	}
	if (!ok) return 1;
	return 0;
}


TriMeshf visualizeColorGrid(const Grid3<vec3uc>& grid)
{
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (const auto& v : grid) {
		if (!(v.value[0] == 0 && v.value[1] == 0 && v.value[2] == 0)) nVoxels++;
	}
	size_t nVertices = nVoxels * 8; //no normals
	size_t nIndices = nVoxels * 12;
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < grid.getDimZ(); z++) {
		for (size_t y = 0; y < grid.getDimY(); y++) {
			for (size_t x = 0; x < grid.getDimX(); x++) {
				const vec3uc& val = grid(x, y, z);
				if (!(val[0] == 0 && val[1] == 0 && val[2] == 0)) {
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						triMesh.m_vertices.back().color = vec4f(vec3f(val) / 255.0f, 1.0f);
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

