
#include "stdafx.h"

#include "GlobalAppState.h"
#include "Fuser.h"
#include "BBHelper.h"

#include "omp.h"

void Visualizer::init(ApplicationData& app)
{
	const auto& gas = GlobalAppState::get();
	LabelUtil::get().init(gas.s_scanLabelFile, gas.s_labelName, gas.s_labelIdName);

	const std::string sceneFileList = gas.s_sceneFileList;
	const char dataid = util::fileNameFromPath(sceneFileList).front();
	if (dataid == 's' || dataid == 'n') {
		m_scans.loadScanNet(gas.s_scanPath, gas.s_scanMeshPath, gas.s_alnPath, sceneFileList, gas.s_bGenerateTestOnly);
		m_bMatterport = false;
	}
	else if (dataid == 'm') {
		m_scans.loadMatterport(gas.s_scanPath, gas.s_scanMeshPath, sceneFileList, gas.s_maxNumSens);
		m_bMatterport = true;
	}
	else
		throw MLIB_EXCEPTION("unable to determine dataset from scene file list: " + sceneFileList);

	m_font.init(app.graphics, "Calibri");
	if (!sceneFileList.empty()) {
		process(app);
	}
	m_bEnableRecording = false;
	m_bEnableAutoRotate = false;
}

void Visualizer::process(ApplicationData& app, float scaleBounds /*= 1.0f*/) //todo something better?
{
	
	const auto& gas = GlobalAppState::get();
	const std::string scanPath = gas.s_scanPath;
	const std::string scanMeshPath = gas.s_scanMeshPath;
	const std::string outputPath = gas.s_outputPath;
	const std::string outputAABBPath = gas.s_outputAABBPath;
	const bool bGenerateTestOnly = gas.s_bGenerateTestOnly;
	const unsigned int numRotAug = gas.s_numRotAug;
	const bool debugOut = gas.s_bDebugOut;

	if (debugOut) {
		std::cout << "DEBUG ENABLED (press key to continue)" << std::endl;
		getchar();
	}

	const auto& scans = m_scans.getScans();
	std::cout << "processing " << scans.size() << " scenes..." << std::endl;
	if (!util::directoryExists(outputPath)) util::makeDirectory(outputPath);
	if (!bGenerateTestOnly && !util::directoryExists(outputAABBPath)) util::makeDirectory(outputAABBPath);

	const int numThreads = 1; //omp_get_max_threads();
	std::vector<Scene*> scenes(numThreads, NULL);
	std::vector<Fuser*> fusers(numThreads, NULL);
	for (unsigned int i = 0; i < numThreads; i++) {
		scenes[i] = new Scene;
		fusers[i] = new Fuser(app);
	}
	
	for (int i = 0; i < (int)scans.size(); i++) {
		int thread = omp_get_thread_num();
		Scene& scene = *scenes[thread];
		const auto& scanInfo = scans[i];
		for (unsigned int r = 0; r < numRotAug; r++) {
			const std::string outFile = outputPath + "/" + scanInfo.sceneName + "__" + std::to_string(r) + "__.scsdf";
			const std::string outAABBFile = outputAABBPath + "/" + scanInfo.sceneName + "__" + std::to_string(r) + "__.aabbs";
			if (true && !debugOut) { // skip if already exists
				if (util::fileExists(outFile) && (bGenerateTestOnly || util::fileExists(outAABBFile))) {
					std::cout << "\r[ " << (i + 1) << " | " << scanInfo.sceneName.size() << " ] (skip) " << scanInfo.sceneName << std::endl;
					continue;
				}
			}
			try {
				//load scene
				std::cout << "\r" << scanInfo.sceneName << ": loading... ";

				const float amountRotate = (r == 0) ? 0.0f : math::randomUniform(10.0f, 350.0f);
				const mat4f rotation = mat4f::rotationY(amountRotate) * mat4f::rotationX(270);
				scene.load(app.graphics, scanInfo, bGenerateTestOnly, rotation);

				//generate complete/incomplete traj
				std::vector<unsigned int> frameIds;
				{ //always need complete
					generateCompleteFrames(scene, frameIds, m_bMatterport);
				}
				//fuse to sdf
				std::cout << "\r" << scanInfo.sceneName << ": fusing " << frameIds.size() << "... ";
				Fuser& fuser = *fusers[thread];
				fuser.fuse(outFile, outAABBFile, scene, frameIds, debugOut);

				if (r == 0)
					std::cout << "\r[ " << (i + 1) << " | " << scans.size() << " ] " << scanInfo.sceneName << std::endl;

			}
			catch (MLibException& e)
			{
				std::stringstream ss;
				ss << "exception caught at scene " << scanInfo.sceneName << " : " << e.what() << std::endl;
				std::cout << ss.str() << std::endl;
			}
		}
	}
	for (unsigned int i = 0; i < scenes.size(); i++) {
		SAFE_DELETE(scenes[i]);
		SAFE_DELETE(fusers[i]);
	}
	std::cout << std::endl << "done!" << std::endl;
}

void Visualizer::render(ApplicationData& app)
{
	m_timer.frame();

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



