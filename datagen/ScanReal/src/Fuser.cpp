
#include "stdafx.h"

#include "GlobalAppState.h"
#include "Fuser.h"
#include "MarchingCubes.h"
#include "CameraUtil.h"
#include "BBHelper.h"


Fuser::Fuser(ml::ApplicationData& _app) : m_app(_app)
{

}

Fuser::~Fuser()
{

}

void Fuser::fuse(const std::string& outFile, const std::string& outAABBFile, Scene& scene,
	const std::vector<unsigned int>& frameIds, bool debugOut /*= false*/)
{
	const auto& gas = GlobalAppState::get();
	const float voxelSize = gas.s_voxelSize;
	const float depthMin = gas.s_renderNear;
	const float depthMax = gas.s_renderFar;
	const unsigned int scenePad = gas.s_scenePadding;
	const bool bGenerateTestOnly = gas.s_bGenerateTestOnly;

	const std::string outputImagePath = gas.s_outputImagePath + "/" + scene.getName();
	if (!util::directoryExists(outputImagePath)) util::makeDirectory(outputImagePath);

	const unsigned int imageWidth = gas.s_renderWidth;
	const unsigned int imageHeight = gas.s_renderHeight;

	const bbox3f bounds = scene.getBoundingBox();
	if (!bounds.isValid()) throw MLIB_EXCEPTION("invalid bounds for fuse");

	std::vector<DXGI_FORMAT> formats = { DXGI_FORMAT_R32G32B32A32_FLOAT };
	m_renderTarget.init(m_app.graphics.castD3D11(), imageWidth, imageHeight, formats);

	vec3ul voxelDim = math::round(bounds.getExtent() / voxelSize);
	//account for scene padding
	voxelDim += scenePad * 2;
	const mat4f worldToGrid = mat4f::scale(1.0f / voxelSize) * mat4f::translation(-bounds.getMin() + scenePad*voxelSize);
	const OBB3f sceneBoundsVoxels = worldToGrid * scene.getOBB();

	//world2grid export
	std::ofstream ofs(outputImagePath + "/world2grid.txt");
	for (unsigned int r = 0; r < 4; r++)
	{
		for (unsigned int c = 0; c < 4; c++)
			ofs << (worldToGrid*scene.getTransformFromOrig())(r, c) << " ";
		ofs << std::endl;
	}
	ofs.close();
	VoxelGrid grid(voxelDim, worldToGrid, voxelSize, sceneBoundsVoxels, 0.4f, 4.0f);
	BaseImage<unsigned char> semantics(imageWidth, imageHeight), instance(imageWidth, imageHeight);
	DepthImage32 depthImage(imageWidth, imageHeight);
	mat4f intrinsic, extrinsic;
	for (unsigned int i = 0; i < frameIds.size(); i++) {
		bool bValid = scene.renderDepthSemantics(m_app.graphics, frameIds[i], intrinsic, extrinsic, depthImage, semantics, instance);
		if (bValid) {
			grid.integrate(intrinsic, extrinsic, depthImage, semantics, instance);
		}
		std::cout << "\r[ " << i << " | " << frameIds.size() << " ]";
	}

	grid.normalizeSDFs();
	grid.saveToFile(outFile, voxelSize, bounds);

	std::vector<BBInfo> bboxes;
	if (!bGenerateTestOnly) { // aabbs
		bboxes = scene.getObjectBBs();
		BBHelper::computeMasks(grid, bboxes);
		BBHelper::exportAABBsToFile(bboxes, outAABBFile, worldToGrid);
	}

	if (debugOut) {
		if (!bGenerateTestOnly) {
			const auto aabbMeshes = BBHelper::visualizeAABBs(bboxes, worldToGrid, grid.getDimensions());
			MeshIOf::saveToFile(util::removeExtensions(outAABBFile) + "_AABBS.ply", aabbMeshes.first);
			MeshIOf::saveToFile(util::removeExtensions(outAABBFile) + "_MASKS.ply", aabbMeshes.second);
		}
		BinaryGrid3 bg = grid.toBinaryGridOccupied(1, 1.0f);
		MeshIOf::saveToFile(util::removeExtensions(outFile) + "_OCC.ply", TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		VoxelGrid re(vec3ui(0, 0, 0), mat4f::identity(), 1.0f, OBB3f(), 0.1f, 10.0f); bbox3f reBbox; float reVoxSize;
		re.loadFromFile(outFile, reVoxSize, reBbox);
		BinaryGrid3 rebg = re.toBinaryGridOccupied(1, 1.0f);
		if (!bGenerateTestOnly) {
			const auto meshLabel = grid.computeLabelMesh(2.0f);
			const auto meshInstance = grid.computeInstanceMesh(2.0f);
			MeshIOf::saveToFile(util::removeExtensions(outFile) + "_LABEL.ply", meshLabel.computeMeshData());
			MeshIOf::saveToFile(util::removeExtensions(outFile) + "_INSTNACE.ply", meshInstance.computeMeshData());
		}

		const float eps = 0.00001f;
		for (unsigned int i = 0; i < 3; i++) {
			if (std::fabs(reBbox.getMin()[i] - bounds.getMin()[i]) > eps ||
				std::fabs(reBbox.getMax()[i] - bounds.getMax()[i]) > eps) {
				std::cout << "error (complete) load/save bbox" << std::endl;
				std::cout << "saved bbox: " << bounds << std::endl;
				std::cout << "loaded bbox: " << reBbox << std::endl;
				getchar();
			}
		}
		if (voxelSize != reVoxSize) {
			std::cout << "error (complete) load/save voxel size " << voxelSize << " vs " << reVoxSize << std::endl;
			getchar();
		}
		if (grid.getVoxelSize() != re.getVoxelSize()) {
			std::cout << "error (complete) load/save voxel size " << grid.getVoxelSize() << " vs " << re.getVoxelSize() << std::endl;
			getchar();
		}
		for (const auto& v : grid) {
			const Voxel& rv = re(v.x, v.y, v.z);
			if (std::fabs(rv.sdf - v.value.sdf) > eps) {
				std::cout << "error (complete) load/save voxel sdf " << v.value.sdf << " vs " << rv.sdf << std::endl;
				getchar();
			}
			if (rebg.isVoxelSet(v.x, v.y, v.z) != bg.isVoxelSet(v.x, v.y, v.z)) {
				std::cout << "error (complete) load/save voxel bg" << std::endl;
				getchar();
			}
		}
		TriMeshf tri = TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f));
		const unsigned int pad = scenePad;
		tri.transform(mat4f::translation(bounds.getMin() - voxelSize * pad) * mat4f::scale(voxelSize));
		MeshIOf::saveToFile(util::removeExtensions(outFile) + "_OCC-WORLD.ply", tri.computeMeshData());
	}
}
