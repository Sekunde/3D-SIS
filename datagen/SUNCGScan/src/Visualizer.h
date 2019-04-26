#pragma once 

class Fuser;

class Visualizer : public ApplicationCallback
{
public:
	void init(ApplicationData &app);
	void render(ApplicationData &app);
	void keyDown(ApplicationData &app, UINT key);
	void keyPressed(ApplicationData &app, UINT key);
	void mouseDown(ApplicationData &app, MouseButtonType button);
	void mouseMove(ApplicationData &app);
	void mouseWheel(ApplicationData &app, int wheelDelta);
	void resize(ApplicationData &app);
	void randomizeCamera();
	std::vector<Cameraf> generateRandomTrajectory(size_t numCams);

	void process(ApplicationData& app, float scaleBounds = 1.0f); 
	void processScene(Scene &scene, Fuser* fuser_, const std::string& outputPath, const std::string& sceneName, const std::string& outputOBBPath, const std::string& outputAABBPath,
		const std::string& cameraPath, int numAug, bool bGenerateOBBs, bool bGenerateAABBs, bool bReadCameras, bool isAugment, float scaleBounds, bool debugOut);

private:
	std::vector<std::string> readScenesFromFile(const std::string& filename) {
		std::vector<std::string> scenes;
		std::ifstream s(filename); std::string line;
		while (std::getline(s, line)) 
			scenes.push_back(line);
		return scenes;
	}

	Scene m_scene;

	D3D11Font m_font;
	FrameTimer m_timer;

	Cameraf m_camera;

	std::vector<std::vector<Cameraf>> m_recordedCameras;
	bool m_bEnableRecording;
	bool m_bEnableAutoRotate;
};