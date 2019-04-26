#pragma once

#include "mLibInclude.h"
#include "VoxelGrid.h"

class Fuser
{
public:
	Fuser(ml::ApplicationData& _app);

	~Fuser();
	
	bbox3f fuse(const std::string& outputFile, Scene& scene, const std::vector<Cameraf>& cameraTrajectory, const std::string& outputOBBFile, const std::string& outputAABBFile, bool isAugment, const bbox3f& scanBounds = bbox3f(), bool debugOut = false);

private:

	void render(Scene& scene, const Cameraf& camera, ColorImageR32G32B32A32& color, DepthImage32& depth);
	void renderDepth(Scene& scene, const Cameraf& camera, DepthImage32& depth);

	void addNoiseToDepth(DepthImage32& depth) const;

	void includeInBoundingBox(BoundingBox3f& bb, const DepthImage16& depthImage, const Cameraf& camera);
	void boundDepthMap(float minDepth, float maxDepth, DepthImage16& depthImage);

	ml::ApplicationData& m_app;
	D3D11RenderTarget m_renderTarget;
};

