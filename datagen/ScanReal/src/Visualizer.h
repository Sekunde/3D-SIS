#pragma once 

#include "ScansDirectory.h"

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

	void process(ApplicationData& app, float scaleBounds = 1.0f); 
private:

	void generateIncompleteFrames(const Scene& scene, const std::vector<unsigned int>& completeFrames, 
		float percentDropFrames, unsigned int dropKeep, std::vector<unsigned int>& incompleteFrames) {
		incompleteFrames.clear();
		if (completeFrames.size() == scene.getNumFrames()) {
			const size_t numFrames = scene.getNumFrames();
			const unsigned int numDrop = (unsigned int)std::round(percentDropFrames * numFrames);
			unsigned int dropStart0 = math::randomUniform(0u, (unsigned int)(numFrames - numDrop));
			unsigned int dropStart1 = math::randomUniform(0u, (unsigned int)(numFrames - numDrop));
			if (dropStart0 > dropStart1) std::swap(dropStart0, dropStart1);
			for (unsigned int f = 0; f < dropStart0; f++) incompleteFrames.push_back(f);
			if (numFrames > 1200) { // drop more
				for (unsigned int f = dropStart0 + numDrop; f < dropStart1; f++) incompleteFrames.push_back(f);
				for (unsigned int f = dropStart1 + numDrop; f < numFrames; f++) incompleteFrames.push_back(f);
			}
			else {
				for (unsigned int f = dropStart0; f < dropStart0 + numDrop; f += dropKeep) incompleteFrames.push_back(f);
				for (unsigned int f = dropStart0 + numDrop; f < numFrames; f++) incompleteFrames.push_back(f);
			}
			return;
		}
		{
			//special case matterport frame characteristics different (randomly drop frames instead of dropping consecutive)
			const float chanceDropFrame = std::min(0.8f, 2.6f * percentDropFrames);
			for (unsigned int f : completeFrames) {
				if (math::randomUniform(0.0f, 1.0f) > chanceDropFrame)
					incompleteFrames.push_back(f);
			}
		}
	}
	void generateCompleteFrames(const Scene& scene, std::vector<unsigned int>& completeFrames, bool bMatterportData) {
		completeFrames.clear();
		if (!bMatterportData) {
			completeFrames.resize(scene.getNumFrames());
			for (unsigned int i = 0; i < completeFrames.size(); i++) completeFrames[i] = i;
			return;
		}
		// matterport - filter out cameras not viewing the scene
		scene.computeTrajFramesInScene(completeFrames);
	}


	ScansDirectory m_scans;

	Scene m_scene;
	bool m_bMatterport;

	D3D11Font m_font;
	FrameTimer m_timer;

	Cameraf m_camera;

	std::vector<std::vector<Cameraf>> m_recordedCameras;
	bool m_bEnableRecording;
	bool m_bEnableAutoRotate;
};