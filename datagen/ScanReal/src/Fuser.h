#pragma once

#include "mLibInclude.h"
#include "VoxelGrid.h"

class Fuser
{
public:
	Fuser(ml::ApplicationData& _app);

	~Fuser();
	
	void fuse(const std::string& outFile, const std::string& outAABBFile, Scene& scene,
		const std::vector<unsigned int>& frameIds, bool debugOut = false);
private:

	void includeInBoundingBox(BoundingBox3f& bb, const DepthImage16& depthImage, const Cameraf& camera);
	void boundDepthMap(float minDepth, float maxDepth, DepthImage16& depthImage);

	ml::ApplicationData& m_app;
	D3D11RenderTarget m_renderTarget;
};

