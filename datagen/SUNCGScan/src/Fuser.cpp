
#include "stdafx.h"

#include "GlobalAppState.h"
#include "Fuser.h"
#include "MarchingCubes.h"
#include "CameraUtil.h"
#include "BBHelper.h"
#include "LabelUtil.h"


Fuser::Fuser(ml::ApplicationData& _app) : m_app(_app)
{

}

Fuser::~Fuser()
{

}


bbox3f Fuser::fuse(const std::string& outputFile, Scene& scene, const std::vector<Cameraf>& cameraTrajectory, const std::string& outputOBBFile, const std::string& outputAABBFile, bool isAugment, const bbox3f& scanBounds /*= bbox3f()*/, bool debugOut /*= false*/)
{
	std::vector<Cameraf> cameras = cameraTrajectory;

	const auto& gas = GlobalAppState::get();
	const bool bAddDepthNoise = gas.s_addNoiseToDepth;

	const float voxelSize = gas.s_voxelSize;
	const float depthMin = gas.s_renderNear;
	const float depthMax = gas.s_renderFar;
	const vec3f dims = gas.s_gridExtents;
	const unsigned int scenePad = gas.s_scenePadding;
	const bool bGenerateImages = gas.s_bGenerateImages;
	const std::string outputImagePath = gas.s_outputImagePath + "/" + util::removeExtensions(util::fileNameFromPath(outputFile));
	if (bGenerateImages && !util::directoryExists(outputImagePath)) util::makeDirectory(outputImagePath);

	const unsigned int validObbMinOcc = gas.s_validObbMinNumOcc;
	const float obbOccThresh = gas.s_validObbOccThresh;

	const unsigned int imageWidth = gas.s_renderWidth;
	const unsigned int imageHeight = gas.s_renderHeight;

	const bbox3f sceneBbox = scene.getBoundingBox();
	std::vector<DXGI_FORMAT> formats = { DXGI_FORMAT_R32G32B32A32_FLOAT };
	m_renderTarget.init(m_app.graphics.castD3D11(), imageWidth, imageHeight, formats);
	const float viewEvalMaxDepth = gas.s_viewEvalMaxDepth;

	PointCloudf pc; BoundingBox3f boundsAll; //debugging 
	vec3f average = vec3f::origin; unsigned int norm = 0; //center of object points within viewEvalMaxDepth from camera

	bbox3f bounds = sceneBbox; float debugScale = 1.0f;

	if (!bounds.isValid()) throw MLIB_EXCEPTION("invalid bounds for fuse");
	vec3ul voxelDim = math::round(bounds.getExtent() / voxelSize);
	const vec3ul volumeVoxDim = math::round(gas.s_gridExtents / voxelSize);
	mat4f worldToGrid;
	//account for scene padding
	voxelDim += scenePad * 2;
	worldToGrid = mat4f::scale(1.0f / voxelSize) * mat4f::translation(-bounds.getMin() + scenePad*voxelSize);

	if (debugOut) { //debugging check mesh
		bbox3f mbounds(bounds.getMin() - voxelSize * 5, bounds.getMax() + voxelSize * 5);
		MeshDataf md; scene.computeSceneMesh(md);
		MeshDataf meshInBounds = md;
		meshInBounds.applyTransform(worldToGrid);
		MeshIOf::saveToFile(util::removeExtensions(outputFile) + "_GT-BLOCK-MESH.ply", meshInBounds);
	}

	const std::string outPathColor = outputImagePath + "/color2";
	const std::string outPathDepth = outputImagePath + "/depth";
	const std::string outPathLabel = outputImagePath + "/label";
	const std::string outPathInstance = outputImagePath + "/instance";
	const std::string outPathPose = outputImagePath + "/pose";
	if (bGenerateImages) {
		if (!util::directoryExists(outPathColor)) util::makeDirectory(outPathColor);
		if (!util::directoryExists(outPathDepth)) util::makeDirectory(outPathDepth);
		if (!util::directoryExists(outPathLabel)) util::makeDirectory(outPathLabel);
		if (!util::directoryExists(outPathInstance)) util::makeDirectory(outPathInstance);
		if (!util::directoryExists(outPathPose)) util::makeDirectory(outPathPose);
		std::ofstream ofs(outputImagePath + "/world2grid.txt");
		for (unsigned int r = 0; r < 4; r++) {
			for (unsigned int c = 0; c < 4; c++)
				ofs << worldToGrid(r, c) << " ";
			ofs << std::endl;
		}
		ofs.close();
	}

	VoxelGrid grid(voxelDim, worldToGrid, voxelSize, depthMin, depthMax);
	for (size_t i = 0; i < cameras.size(); i++) {
		const auto& c = cameras[i];
		Scene::RenderedView view = scene.renderToImage(c, imageWidth, imageHeight);
		if (bAddDepthNoise) {
			addNoiseToDepth(view.depth);
			// filter the depth map
			view.depth = CameraUtil::bilateralFilter(view.depth, 2.0f, 0.1f);
		}
		const mat4f intrinsic = c.getIntrinsic(imageWidth, imageHeight);
		const mat4f extrinsic = c.getExtrinsic();
		grid.integrate(intrinsic, extrinsic, view.depth, view.semanticInstance);
		std::cout << "\r[ " << (i + 1) << " | " << cameras.size() << " ]";
		
		if (bGenerateImages) {
			BaseImage<unsigned char> label(view.semanticLabel.getWidth(), view.semanticLabel.getHeight());
			for (auto& p : label) {
				unsigned short val = view.semanticLabel(p.x, p.y);
				if (val > 40) throw MLIB_EXCEPTION("bad semantic label " + std::to_string(val));
				p.value = (unsigned char)val;
			}
			if (debugOut) {
				FreeImageWrapper::saveImage(outPathLabel + "/" + std::to_string(i) + "_vis.png", view.getColoredSemanticLabel());
				FreeImageWrapper::saveImage(outPathInstance + "/" + std::to_string(i) + "_vis.png", view.getColoredSemanticInstance());
			}

			FreeImageWrapper::saveImage(outPathColor + "/" + std::to_string(i) + ".jpg", ColorImageR8G8B8(view.color));
			FreeImageWrapper::saveImage(outPathDepth + "/" + std::to_string(i) + ".png", DepthImage16(view.depth));
			FreeImageWrapper::saveImage(outPathLabel + "/" + std::to_string(i) + ".png", label);
			FreeImageWrapper::saveImage(outPathInstance + "/" + std::to_string(i) + ".png", view.semanticInstance);
			{
				std::ofstream ofs(outPathPose + "/" + std::to_string(i) + ".txt");
				for (unsigned int r = 0; r < 4; r++) {
					for (unsigned int c = 0; c < 4; c++)
						ofs << extrinsic(r, c) << " ";
					ofs << std::endl;
				}
				ofs.close();
			}
			{ // compute 2d bboxes
				std::vector<std::pair<bbox2i, unsigned short>> bboxes2d = BBHelper::compute2DBboxes(view.semanticInstance, view.semanticLabel);
				if (!bboxes2d.empty()) BBHelper::write2DBboxesToFile(outPathInstance + "/" + std::to_string(i) + ".txt", bboxes2d);
			}
		}
	}

	std::cout << "normalizing + improving... ";
	grid.normalizeSDFs();
	grid.improveSDF(4);
	grid.saveToFile(outputFile, voxelSize, bounds); //TODO also have rotations???

	if (!outputAABBFile.empty()) {
		MeshDataf sceneMesh;
		scene.computeSceneMesh(sceneMesh);
		sceneMesh.makeTriMesh();
		TriMeshf sceneMeshGrid(sceneMesh);
		sceneMeshGrid.transform(worldToGrid);

		const std::vector<BBInfo>& aabbs = scene.getObjectBBs();
		std::vector<MaskType> origmasks;
		const std::string cacheFile = isAugment ? gas.s_outputPath + "/cache/" + util::splitOnFirst(util::removeExtensions(util::fileNameFromPath(outputAABBFile)), "__").first  + "__0__.masks" :
			gas.s_outputPath + "/cache/" + util::removeExtensions(util::fileNameFromPath(outputAABBFile)) + ".masks";
		if (!isAugment) {
			BBHelper::computeMasks(sceneMeshGrid, grid, aabbs, scene.getBoundingBox().getMin(), scenePad, origmasks);
			BinaryDataStreamFile ofs(cacheFile, true);
			ofs << origmasks;
			ofs.close();
		}
		else {
			BinaryDataStreamFile ifs(cacheFile, false);
			ifs >> origmasks;
			ifs.close();
		}
		std::vector<BBInfo> validAabbs;
		BBHelper::computeValidAabbs(sceneMeshGrid, grid, aabbs, origmasks, scene.getBoundingBox().getMin(), scenePad, obbOccThresh, validObbMinOcc, validAabbs);
		BBHelper::exportAABBsToFile(validAabbs, outputAABBFile, worldToGrid);

		if (debugOut) {
			const std::tuple<MeshDataf, MeshDataf, MeshDataf> meshes = BBHelper::visualizeAABBs_Instance(validAabbs, worldToGrid, grid.getDimensions());
			MeshIOf::saveToFile(util::removeExtensions(outputAABBFile) + "_AABBS.ply", std::get<0>(meshes));
			MeshIOf::saveToFile(util::removeExtensions(outputAABBFile) + "_MASKSPartial.ply", std::get<1>(meshes));
			MeshIOf::saveToFile(util::removeExtensions(outputAABBFile) + "_MASKSComplete.ply", std::get<2>(meshes));
			const auto meshesCanonical = BBHelper::visualizeAABBCanonicals(validAabbs, worldToGrid, grid.getDimensions());
			MeshIOf::saveToFile(util::removeExtensions(outputAABBFile) + "_AABBS_CANONICAL.ply", meshesCanonical.first);

			std::vector<BBInfo> validAabbsCheck;
			BBHelper::readAABBsFromFile(outputAABBFile, validAabbsCheck);
			if (validAabbsCheck.size() != validAabbs.size()) {
				std::cout << "error: load/save aabbs #aabbs = " << validAabbs.size() << " vs " << validAabbsCheck.size()  << std::endl;
				getchar();
			}
			for (unsigned int i = 0; i < validAabbs.size(); i++) {
				const auto& refAabb = validAabbs[i];
				const auto& loadAabb = validAabbsCheck[i];
				if (refAabb.labelId != loadAabb.labelId) {
					std::cout << "error: load/save aabbs labelId = " << refAabb.labelId << " vs " << loadAabb.labelId << std::endl;
					getchar();
				}
				bbox3f transformedRef(refAabb.aabb); transformedRef.transform(worldToGrid);
				if ((transformedRef.getMin() - loadAabb.aabb.getMin()).length() > 0.0001f) {
					std::cout << "error: load/save aabbs min: " << transformedRef.getMin() << " vs " << loadAabb.aabb.getMin() << std::endl;
					getchar();
				}
				if ((transformedRef.getMax() - loadAabb.aabb.getMax()).length() > 0.0001f) {
					std::cout << "error: load/save aabbs max: " << transformedRef.getMax() << " vs " << loadAabb.aabb.getMax() << std::endl;
					getchar();
				}
				const auto& refBg = refAabb.mask;
				const auto& loadBg = loadAabb.mask;
				if (refBg.getNumElements() != loadBg.getNumElements()) {
					std::cout << "error: load/save masks #elem = " << refBg.getNumElements() << " vs " << loadBg.getNumElements() << std::endl;
					getchar();
				}
				for (unsigned int z = 0; z < refBg.getDimZ(); z++) {
					for (unsigned int y = 0; y < refBg.getDimY(); y++) {
						for (unsigned int x = 0; x < refBg.getDimX(); x++) {
							if (refBg(x, y, z) != loadBg(x, y, z)) {
								std::cout << "error: load/save masks at (" << x << ", " << y << ", " << z << "): " << refBg(x, y, z) << " vs " << loadBg(x, y, z) << std::endl;
								getchar();
							}
						}
					}
				}
			}
		}  // debugOut
	}  // export obbs + masks

	if (debugOut) {
		BinaryGrid3 bg = grid.toBinaryGridOccupied(1, 1.0f);
		MeshIOf::saveToFile(util::removeExtensions(outputFile) + "_OCC.ply", TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		VoxelGrid re(vec3ui(0, 0, 0), mat4f::identity(), 1.0f, 0.1f, 10.0f); bbox3f reBbox; float reVoxSize;
		re.loadFromFile(outputFile, reVoxSize, reBbox);
		BinaryGrid3 rebg = re.toBinaryGridOccupied(1, 1.0f);

		const float eps = 0.00001f;
		for (unsigned int i = 0; i < 3; i++) {
			if (std::fabs(reBbox.getMin()[i] - bounds.getMin()[i]) > eps ||
				std::fabs(reBbox.getMax()[i] - bounds.getMax()[i]) > eps) {
				std::cout << "error load/save bbox" << std::endl;
				std::cout << "saved bbox: " << bounds << std::endl;
				std::cout << "loaded bbox: " << reBbox << std::endl;
				getchar();
			}
		}
		if (voxelSize != reVoxSize) {
			std::cout << "error load/save voxel size " << voxelSize << " vs " << reVoxSize << std::endl;
			getchar();
		}
		if (grid.getVoxelSize() != re.getVoxelSize()) {
			std::cout << "error load/save voxel size " << grid.getVoxelSize() << " vs " << re.getVoxelSize() << std::endl;
			getchar();
		}
		for (const auto& v : grid) {
			const Voxel& rv = re(v.x, v.y, v.z);
			if (std::fabs(rv.sdf - v.value.sdf) > eps) {
				std::cout << "error load/save voxel sdf " << v.value.sdf << " vs " << rv.sdf << std::endl;
				getchar();
			}
			if (rebg.isVoxelSet(v.x, v.y, v.z) != bg.isVoxelSet(v.x, v.y, v.z)) {
				std::cout << "error load/save voxel bg" << std::endl;
				getchar();
			}
		}
	}
	return bounds;
}



void Fuser::render(Scene& scene, const Cameraf& camera, ColorImageR32G32B32A32& color, DepthImage32& depth)
{
#pragma omp critical 
		{
			auto &g = m_app.graphics.castD3D11();

			m_renderTarget.bind();
			m_renderTarget.clear(ml::vec4f(1.0f, 0.0f, 1.0f, 0.0f));

			scene.render(camera);

			m_renderTarget.unbind();

			m_renderTarget.captureColorBuffer(color);
			m_renderTarget.captureDepthBuffer(depth);
		}
}

void Fuser::renderDepth(Scene& scene, const Cameraf& camera, DepthImage32& depth)
{
#pragma omp critical 
		{
			auto &g = m_app.graphics.castD3D11();
			m_renderTarget.bind();
			m_renderTarget.clear(ml::vec4f(1.0f, 0.0f, 1.0f, 0.0f));
			scene.render(camera);
			m_renderTarget.unbind();
			m_renderTarget.captureDepthBuffer(depth);
		}
}

void Fuser::addNoiseToDepth(DepthImage32& depth) const {
	const auto INVALID = depth.getInvalidValue();
	for (auto& p : depth) {
		if (p.value != INVALID) {
			if (p.value > 0.4f) {
				float variance = 0.005f * p.value + 0.001f * std::pow(p.value, 2.5f);
				float noise = math::randomNormal(0.0f, variance);
				p.value += noise;
			}
			else { //min depth
				p.value = INVALID;
			}
		}
	}
}

