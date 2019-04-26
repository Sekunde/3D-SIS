#include "stdafx.h"
#include "GlobalAppState.h"
#include "Fuser.h"
#include "ViewGenerator.h"
#include "omp.h"
#include "BBHelper.h"

void loadDf(const std::string filename, DistanceField3f& df, float& dfVoxSize, bbox3f& dfBbox)
{
	std::ifstream ifs(filename, std::ios::binary);
	//metadata
	UINT64 dimX, dimY, dimZ;
	ifs.read((char*)&dimX, sizeof(UINT64));
	ifs.read((char*)&dimY, sizeof(UINT64));
	ifs.read((char*)&dimZ, sizeof(UINT64));
	ifs.read((char*)&dfVoxSize, sizeof(float));
	vec3f bmin, bmax;
	ifs.read((char*)&bmin.x, sizeof(float));
	ifs.read((char*)&bmin.y, sizeof(float));
	ifs.read((char*)&bmin.z, sizeof(float));
	ifs.read((char*)&bmax.x, sizeof(float));
	ifs.read((char*)&bmax.y, sizeof(float));
	ifs.read((char*)&bmax.z, sizeof(float));
	dfBbox = bbox3f(bmin, bmax);
	//dense data
	df.allocate(dimX, dimY, dimZ);
	ifs.read((char*)df.getData(), sizeof(float)*df.getNumElements());
	ifs.close();
}

void Visualizer::init(ApplicationData& app)
{
	LabelUtil::get().init(GlobalAppState::get().s_suncgLabelMapFile, GlobalAppState::get().s_labelMapLabelName, GlobalAppState::get().s_labelMapIdName);
	LabelUtil::get().initNyu("../fileLists/nyu40_eigen13_labels.csv", "nyu40class", "nyu40id");

	ViewGenerator::initViewStats(GlobalAppState::get().s_viewStatsFile);
	const std::string sceneFileList = GlobalAppState::get().s_sceneFileList;


	m_font.init(app.graphics, "Calibri");
	if (!sceneFileList.empty()) {
		process(app);
	}

	{
		const std::string house = GlobalAppState::get().s_suncgPath + "/house/" + GlobalAppState::get().s_meshFilenames.front() + "/house.json";
		std::cout << "loading scene... ";
		m_scene.loadFromJson(house, app.graphics, GlobalAppState::get());
		std::cout << "done!" << std::endl;
		std::cout << "scene bounds:\n" << m_scene.getBoundingBox() << std::endl;

		vec3f eye = vec3f(
			m_scene.getBoundingBox().getMinX(),
			0.5f*(m_scene.getBoundingBox().getMinY() + m_scene.getBoundingBox().getMaxY()),
			0.5f*(m_scene.getBoundingBox().getMinZ() + m_scene.getBoundingBox().getMaxZ())
			);
		eye = eye - 3.0f*(m_scene.getBoundingBox().getCenter() - eye);
		vec3f worldUp = vec3f::eY;
		m_camera = Cameraf(eye, m_scene.getBoundingBox().getCenter() - eye, worldUp, GlobalAppState::get().s_cameraFov, (float)app.window.getWidth() / app.window.getHeight(), 0.04f, 200.0f);	//in theory that should be depth min/max...
	}

	m_bEnableRecording = false;
	m_bEnableAutoRotate = false;
}

void Visualizer::process(ApplicationData& app, float scaleBounds /*= 1.0f*/) //todo something better?
{
	
	const auto& gas = GlobalAppState::get();
	const bool debugOut = gas.s_bDebugOut;
	if (debugOut) {
		std::cout << "DEBUG ENABLED (press key to continue)" << std::endl;
		getchar();
	}
	const std::string suncgPath = gas.s_suncgPath;
	const std::string outputPath = gas.s_outputPath;
	const std::string cameraPath = gas.s_cameraPath;
	const bool bGenerateOBBs = gas.s_bGenerateOBBs;
	const bool bGenerateAABBs = gas.s_bGenerateAABBs;
	const bool bGenerateCameras = gas.s_bGenerateCameras;
	const unsigned int numAugmentPartial = gas.s_numAugmentPartial;
	const std::string outputOBBPath = gas.s_outputOBBPath;
	const std::string outputAABBPath = gas.s_outputAABBPath;

	const std::string outCachePath = outputPath + "/cache";
	if (!util::directoryExists(outCachePath)) util::makeDirectory(outCachePath);
	const std::string outAugCachePath = outCachePath;
	if (!util::directoryExists(outAugCachePath)) throw MLIB_EXCEPTION("cache path (" + outAugCachePath + ") does not exist!");

	std::vector<std::string> sceneNames = readScenesFromFile(gas.s_sceneFileList);
	std::cout << "processing " << sceneNames.size() << " scenes..." << std::endl;

	if (bGenerateOBBs && !util::directoryExists(outputOBBPath)) util::makeDirectory(outputOBBPath);
	if (bGenerateAABBs && !util::directoryExists(outputAABBPath)) util::makeDirectory(outputAABBPath);
	if (bGenerateCameras && !util::directoryExists(cameraPath)) util::directoryExists(cameraPath);
	if (!util::directoryExists(outputPath)) util::makeDirectory(outputPath);

	const bool bReadCameras = !bGenerateCameras;
	if (bReadCameras) {
		Directory camDir(cameraPath);
		if (camDir.getFiles().size() < sceneNames.size()) {
			std::cout << "warning: camera dir contains " << camDir.getFiles().size() << " vs " << sceneNames.size() << " scenes" << std::endl;
			std::cout << "(press key to continue)" << std::endl;
			getchar();
		}
		std::cout << "READING CAMERAS" << std::endl;
	}
	else {
		std::cout << "GENERATING CAMERAS" << std::endl;
		std::cout << "(press key to continue)" << std::endl;
		getchar(); //just for safety
		if (!util::directoryExists(cameraPath)) util::makeDirectory(cameraPath);
	}

	Fuser fuser(app);
	for (int i = 0; i < (int)sceneNames.size(); i++) {
		bool bHasDefault = false;
		if (true) { // check if file exists already -> skip
			const std::string outFile = outputPath + "\\" + sceneNames[i] + "__0__.scsdf";
			const std::string outAugFile = outputPath + "\\" + sceneNames[i] + "__0__" + std::to_string(numAugmentPartial - 1) + ".scsdf";

			if (util::fileExists(outFile) && (numAugmentPartial == 0 || util::fileExists(outAugFile))) {
				if (i % 20 == 0)
					std::cout << "\r[ " << (i + 1) << " | " << sceneNames.size() << " ] (skip) " << sceneNames[i] << std::endl;
				continue;
			}
		}
		try {
			//load scene
			Scene scene;
			std::cout << "\r" << sceneNames[i] << ": loading... ";
			const std::string house = suncgPath + "/house/" + sceneNames[i] + "/house.json";
			scene.loadFromJson(house, app.graphics, gas);

			if (!bHasDefault) { // check if non-augmented scan exists
				processScene(scene, &fuser, outputPath, sceneNames[i], outputOBBPath, outputAABBPath, cameraPath,
					-1, bGenerateOBBs, bGenerateAABBs, bReadCameras, false, scaleBounds, debugOut);
			}
			if (numAugmentPartial > 0) { // continually randomly rotate the scene objects
				for (int augId = 0; augId < (int)numAugmentPartial; augId++) {
					const std::string augCacheFile = outAugCachePath + "/" + sceneNames[i] + "__0__" + std::to_string(augId) + ".augmentation";
					scene.augment(app.graphics, augCacheFile);
					processScene(scene, &fuser, outputPath, sceneNames[i], outputOBBPath, outputAABBPath, cameraPath,
						augId, bGenerateOBBs, bGenerateAABBs, bReadCameras, true, scaleBounds, debugOut);
				}
			}

			if (i % 20 == 0)
				std::cout << "\r[ " << (i + 1) << " | " << sceneNames.size() << " ] " << sceneNames[i] << std::endl;
		}
		catch (MLibException& e)
		{
			std::stringstream ss;
			ss << "exception caught at scene " << sceneNames[i] << " : " << e.what() << std::endl;
			std::cout << ss.str() << std::endl;
			const std::string outpath = outputPath + sceneNames[i] + "__0__";
			if (util::directoryExists(outpath)) util::deleteDirectory(outpath);
		}
	}
	std::cout << std::endl << "done!" << std::endl;
}

void Visualizer::randomizeCamera() {
	vec3f dir = vec3f((float)RNG::global.rand_closed01() - 0.5f, (float)RNG::global.rand_closed01() - 0.5f, (float)RNG::global.rand_closed01() - 0.5f).getNormalized();
	vec3f worldUp = (dir ^ vec3f(dir.z, -dir.x, dir.y)).getNormalized();

	float dist = 1.5f*m_scene.getBoundingBox().getMaxExtent();
	vec3f eye = m_scene.getBoundingBox().getCenter() - dist*dir;

	vec3f lookDir = (m_scene.getBoundingBox().getCenter() - eye).getNormalized();
	m_camera.reset(eye, lookDir, worldUp);
}

std::vector<Cameraf> Visualizer::generateRandomTrajectory(size_t numCams)
{
	std::vector<Cameraf> res(numCams);
	for (size_t i = 0; i < numCams; i++) {
		randomizeCamera();
		res[i] = m_camera;
	}
	return res;
}

void Visualizer::render(ApplicationData& app)
{
	if (m_bEnableAutoRotate) {

		randomizeCamera();
	}

	m_timer.frame();

	if (GlobalAppState::get().s_BRDF == 0)
	{
		m_scene.render(m_camera); // phong
	}
	else {
		m_scene.render(m_camera);	//ward
	}


	m_font.drawString("FPS: " + convert::toString(m_timer.framesPerSecond()), vec2i(10, 5), 24.0f, RGBColor::Red);

	if (m_bEnableRecording) {
		if (m_recordedCameras.empty()) m_recordedCameras.push_back(std::vector<Cameraf>());
		m_recordedCameras.back().push_back(m_camera);
		m_font.drawString("RECORDING ON " + std::to_string(m_recordedCameras.size()), vec2i(10, 30), 24.0f, RGBColor::Red);
	}
}


void Visualizer::resize(ApplicationData &app)
{
	m_camera.updateAspectRatio((float)app.window.getWidth() / app.window.getHeight());
}

void Visualizer::keyDown(ApplicationData& app, UINT key)
{
	//if (key == KEY_F) app.graphics.castD3D11().toggleWireframe();

	if (key == KEY_U)
	{
	}

	if (key == KEY_I)
	{
		m_scene.randomizeLighting();
	}

	if (key == KEY_Y) {
		m_bEnableAutoRotate = !m_bEnableAutoRotate;
	}

	//record trajectory
	if (key == KEY_R) {
		if (m_bEnableRecording == false) {
			m_recordedCameras.clear();
			m_bEnableRecording = true;
		}
		else {
			m_bEnableRecording = false;
		}
	}

	if (key == KEY_F1) {

		ml::D3D11RenderTarget renderTarget;
		renderTarget.init(app.graphics.castD3D11(), app.window.getWidth(), app.window.getHeight());
		renderTarget.clear();
		renderTarget.bind();

		m_scene.render(m_camera);	//render call

		renderTarget.unbind();
		ml::ColorImageR8G8B8A8 color;
		ml::DepthImage32 depth;
		renderTarget.captureColorBuffer(color);
		renderTarget.captureDepthBuffer(depth, m_camera.getProj());
		FreeImageWrapper::saveImage("color.png", color);
		FreeImageWrapper::saveImage("depth.png", ml::ColorImageR32G32B32A32(depth));

		{
			ml::PointCloudf pc;
			mat4f intr = Cameraf::graphicsToVisionProj(m_camera.getProj(), depth.getWidth(), depth.getHeight()).getInverse();
			for (auto &p : depth) {
				if (p.value != depth.getInvalidValue()) {
					vec3f point = intr * vec3f(p.x*p.value, p.y*p.value, p.value);
					pc.m_points.push_back(point);
				}
			}
			ml::PointCloudIOf::saveToFile("test.ply", pc);
		}


		{
			renderTarget.captureDepthBuffer(depth);
			mat4f projToCamera = m_camera.getProj().getInverse();
			mat4f cameraToWorld = m_camera.getView().getInverse();
			mat4f projToWorld = cameraToWorld * projToCamera;

			for (auto &p : depth) {
				if (p.value != 0.0f && p.value != 1.0f) {
					vec3f posProj = vec3f(app.graphics.castD3D11().pixelToNDC(vec2i((int)p.x, (int)p.y), depth.getWidth(), depth.getHeight()), p.value);
					vec3f posCamera = projToCamera * posProj;
					p.value = posCamera.z;
				}
				else {
					p.value = depth.getInvalidValue();
				}
			}
			FreeImageWrapper::saveImage("color1.png", color);
			FreeImageWrapper::saveImage("depth1.png", ml::ColorImageR32G32B32A32(depth));
		}

		std::cout << "screenshot taken (color.png / depth.png)" << std::endl;
	}

	if (key == KEY_Q) {
		Scene::RenderedView view = m_scene.renderToImage(m_camera, 1280, 480);
		FreeImageWrapper::saveImage("test_color.png", view.color);
		FreeImageWrapper::saveImage("test_depth.png", ColorImageR32G32B32A32(view.depth));
		FreeImageWrapper::saveImage("test_semantics.png", view.getColoredSemanticLabel());
	}

	if (key == KEY_ESCAPE) {
		PostQuitMessage(WM_QUIT);
	}
}

void Visualizer::keyPressed(ApplicationData &app, UINT key)
{
	const float distance = 0.1f;
	const float theta = 0.1f;

	if (key == KEY_S) m_camera.move(-distance);
	if (key == KEY_W) m_camera.move(distance);
	if (key == KEY_A) m_camera.strafe(-distance);
	if (key == KEY_D) m_camera.strafe(distance);
	if (key == KEY_E) m_camera.jump(-distance);
	if (key == KEY_Q) m_camera.jump(distance);

	if (key == KEY_UP) m_camera.lookUp(theta);
	if (key == KEY_DOWN) m_camera.lookUp(-theta);
	if (key == KEY_LEFT) m_camera.lookRight(theta);
	if (key == KEY_RIGHT) m_camera.lookRight(-theta);

	if (key == KEY_Z) m_camera.roll(theta);
	if (key == KEY_X) m_camera.roll(-theta);


}

void Visualizer::mouseDown(ApplicationData &app, MouseButtonType button)
{

}

void Visualizer::mouseWheel(ApplicationData &app, int wheelDelta)
{
	const float distance = 0.01f;
	m_camera.move(distance * wheelDelta);
}

void Visualizer::mouseMove(ApplicationData &app)
{
	const float distance = 0.05f;
	const float theta = 0.5f;

	vec2i posDelta = app.input.mouse.pos - app.input.prevMouse.pos;

	if (app.input.mouse.buttons[MouseButtonRight])
	{
		m_camera.strafe(distance * posDelta.x);
		m_camera.jump(distance * posDelta.y);
	}

	if (app.input.mouse.buttons[MouseButtonLeft])
	{
		m_camera.lookRight(theta * posDelta.x);
		m_camera.lookUp(theta * posDelta.y);
	}

}

void Visualizer::processScene(Scene &scene, Fuser* fuser_, const std::string& outputPath, const std::string& sceneName, const std::string& outputOBBPath, const std::string& outputAABBPath,
	const std::string& cameraPath, int numAug, bool bGenerateOBBs, bool bGenerateAABBs, bool bReadCameras, bool isAugment, float scaleBounds, bool debugOut)
{
	const auto& gas = GlobalAppState::get();
	const unsigned int maxNumTraj = gas.s_maxNumTrajectories;
	const unsigned int maxNumViews = gas.s_maxNumViews;
	const unsigned int maxNumViewPerTraj = gas.s_maxNumViewPerTraj;
	const unsigned int maxNumSampleViews = gas.s_maxNumViewsToSample;
	const unsigned int scenePad = gas.s_scenePadding;
	auto& fuser = *fuser_;

	//gen views
	std::vector<std::vector<Cameraf>> trajectories; std::vector<bbox3f> scanBounds;
	bool generateCam = !bReadCameras;

	// read cameras
	if (bReadCameras) {
		if (!util::fileExists(cameraPath + "\\" + sceneName + ".cameras")) {
			std::cout << "no camera file for " << sceneName << ", skipping" << std::endl;
			return;
		}
		else {
			BinaryDataStreamFile s(cameraPath + "\\" + sceneName + ".cameras", false);
			s >> trajectories; s >> scanBounds;
			s.close();
			for (auto& b : scanBounds)
				b = bbox3f(b.getCenter() - scaleBounds * 0.5f * b.getExtent(), b.getCenter() + scaleBounds * 0.5f * b.getExtent());
			if (trajectories.size() > maxNumTraj) {
				trajectories.resize(maxNumTraj);
				scanBounds.resize(maxNumTraj);
			}
		}
	}

	//generate cameras
	if (generateCam) {
		std::cout << "\r" << sceneName << ": generating views... ";
		ViewGenerator vg(&scene);
		trajectories = vg.genTrajectories(maxNumTraj, maxNumViewPerTraj, maxNumViews, maxNumSampleViews, true, debugOut); //sample entire scene = true
	}
	if (trajectories.empty()) {
		std::cout << "\rno trajectories for " << sceneName << ", skipping";
		return;
	}

	scanBounds.resize(trajectories.size());
	//compute sdfs
	std::cout << "\r" << sceneName << ": fusing " << trajectories.front().size() << " cameras... ";
	for (unsigned int c = 0; c < trajectories.size(); c++) {
		const std::string augId = numAug >= 0 ? std::to_string(numAug) : "";
		const std::string outFile = outputPath + "\\" + sceneName + "__" + std::to_string(c) + "__" + augId + ".scsdf";
		const std::string outOBBFile = bGenerateOBBs ? outputOBBPath + "\\" + sceneName + "__" + std::to_string(c) + "__" + augId + ".obbs" : "";
		const std::string outAABBFile = bGenerateAABBs ? outputAABBPath + "\\" + sceneName + "__" + std::to_string(c) + "__" + augId + ".aabbs" : "";

		const auto& gas = GlobalAppState::get();
		const float renderFar = gas.s_renderFar;

		for (auto& camera : trajectories[c])
			camera.setFarPlane(renderFar);
		bbox3f bounds = fuser.fuse(outFile, scene, trajectories[c], outOBBFile, outAABBFile, isAugment, scanBounds[c], debugOut);
		if (!bReadCameras) scanBounds[c] = bounds;
	}

	if (!bReadCameras || generateCam) {
		//save cameras to file
		BinaryDataStreamFile s(cameraPath + "\\" + sceneName + ".cameras", true);
		s << trajectories;
		s << scanBounds;
		s.close();
	}
}

