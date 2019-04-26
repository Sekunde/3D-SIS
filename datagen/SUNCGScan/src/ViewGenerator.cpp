
#include "stdafx.h"
#include "ViewGenerator.h"

ViewStats ViewGenerator::s_viewStats;

std::vector<std::vector<Cameraf>> ViewGenerator::genTrajectories(unsigned int maxNumTraj, unsigned int maxNumViewPerTraj, unsigned int numKeyViews, unsigned int maxNumViewSamples, bool bSampleEntireScene,
	bool bDebug /*= false*/) const {
	std::vector<std::vector<Cameraf>> trajectories;
	//get trajectory keys
	const std::vector<std::vector<Cameraf>> trajKeys = generateKeys(maxNumTraj, numKeyViews, maxNumViewSamples, true, false);

	//interpolate to trajectory
	const float camerasPerM = 2.0f;
	const float camerasPerRad = 0.01f;
	if (!trajKeys.empty()) trajectories.resize(trajKeys.size());
	for (unsigned int t = 0; t < trajKeys.size(); t++) {
		const auto& cameras = trajKeys[t];
		const unsigned int maxPerSegment = std::max(5u, maxNumViewPerTraj / (unsigned int)cameras.size());
		trajectories[t].push_back(cameras.front());
		for (unsigned int i = 0; i < cameras.size() - 3; i++) { //TODO need to add several cameras to the end/begin since these are cut off from the interp...
			const float dist = vec3f::dist(cameras[i + 1].getEye(), cameras[i + 2].getEye());
			const float angle = std::acos(math::clamp(cameras[i + 1].getLook() | cameras[i + 2].getLook(), -1.0f, 1.0f));
			const unsigned int numT = (unsigned int)std::round(dist * camerasPerM);
			const unsigned int numR = (unsigned int)std::round(angle * camerasPerRad);
			const unsigned int num = std::min(maxPerSegment, std::max(numT, numR));
			std::vector<Cameraf> cams;
			catmullRomSpline(
				cameras[i + 0],
				cameras[i + 1],
				cameras[i + 2],
				cameras[i + 3],
				num, cams);
			trajectories[t].push_back(cams.front());
			for (unsigned int j = 1; j < cams.size()-1; j++) { //check camera validity
				const vec3f eye = cams[j].getEye();
				bool bValid = !m_scene->intersectsCameraBox(cams[j], 0.2f, 0.8f);
				if (bValid) {
					if (math::randomUniform(0.0f, 1.0f) < 0.8f)
						trajectories[t].push_back(cams[j]);
				}
			}
			trajectories[t].push_back(cams.back());
		}
		for (unsigned int i = cameras.size() - 3; i < cameras.size(); i++) trajectories[t].push_back(cameras[i]);
	}

	return trajectories;
}

std::vector<std::vector<Cameraf>> ViewGenerator::generateKeys(unsigned int maxNumTraj, unsigned int numViews, unsigned int maxNumViewSamples, bool bSampleEntireScene, 
	bool bDebug /*= false*/) const {
	float fov = 60.0f;
	unsigned int renderWidth = GlobalAppState::get().s_renderWidth;
	unsigned int renderHeight = GlobalAppState::get().s_renderHeight;
	float renderNear = GlobalAppState::get().s_renderNear;
	float renderFar = GlobalAppState::get().s_renderFar;

	std::vector<std::vector<Cameraf>> trajectories;
	std::vector<std::vector<bbox3f>> roomBboxes;
	m_scene->getRoomBboxes(roomBboxes);

	const float coverageVoxSize = 1.5f;//2.0f; //meters
	const unsigned int maxLevel = 1; //only consider ground height for now //TODO
	const float minPercentPixObjects = 0.2f; //TODO params

	BoundingBox3f bb = m_scene->getBoundingBox();
	if (bb.getMinY() < 0) bb.setMinY(0.0f);
	vec3ui coverageDims = math::ceil(bb.getExtent() / coverageVoxSize);
	coverageDims.y = 1; //only consider ground height for now //TODO
	const unsigned int numCamsToTryPerCoverageVox = std::max(3u, std::min(10u, maxNumViewSamples / (coverageDims.z*coverageDims.x)));
	for (unsigned int t = 0; t < maxNumTraj; t++) {
		trajectories.push_back(std::vector<Cameraf>());
		BinaryGrid3 bgCoverage(coverageDims);
		for (unsigned int level = 0; level < maxLevel; level++) {
			//iterate thru rooms in level
			for (unsigned int r = 0; r < roomBboxes[level].size(); r++) {
				const bbox3f& roomBbox = roomBboxes[level][r];
				vec3ui roomCovDims = math::ceil(roomBbox.getExtent() / coverageVoxSize);
				roomCovDims.y = std::min(coverageDims.y, roomCovDims.y);

				//traverse in zigzag/scanline order
				for (unsigned int y = 0; y < roomCovDims.y; y++) {
					for (unsigned int z = 0; z < roomCovDims.z; z++) {
						for (unsigned int _x = 0; _x < roomCovDims.x; _x++) {
							unsigned int x = (z % 2) == 0 ? _x : roomCovDims.x - _x - 1; //iterate forward/backward
							vec3ui coverageLoc = math::round((vec3f(x, y, z) * coverageVoxSize + roomBbox.getMin() - bb.getMin()) / coverageVoxSize);
							if (!bgCoverage.isValidCoordinate(coverageLoc) || bgCoverage.isVoxelSet(coverageLoc)) continue; //already got some views here

							bbox3f curBbox(vec3f(x, y, z)*coverageVoxSize + roomBbox.getMin(), math::min(roomBbox.getMax(), (vec3f(x, y, z) + 1.0f)*coverageVoxSize + roomBbox.getMin()));
							//sample for best camera
							Cameraf bestCamera; vec2f bestScore = vec2f(0.0f, 0.0f);
							for (unsigned int c = 0; c < numCamsToTryPerCoverageVox; c++) {
								vec3f eye;
								eye.x = math::linearMap(0.0f, 1.0f, curBbox.getMinX(), curBbox.getMaxX(), math::randomUniform(0.0f, 1.0f));
								eye.z = math::linearMap(0.0f, 1.0f, curBbox.getMinZ(), curBbox.getMaxZ(), math::randomUniform(0.0f, 1.0f));
								eye.y = math::randomNormal(s_viewStats.camHeightMean, s_viewStats.camHeightStd);
								math::clamp(eye, curBbox.getMin(), curBbox.getMax());

								const float horizRot = math::randomUniform(0.0f, 360.0f);
								float vertRot = math::randomNormal(s_viewStats.camAngleMean, s_viewStats.camAngleStd);
								vec3f lookDir = mat3f::rotationY(horizRot) * mat3f::rotationX(vertRot) * vec3f::eZ;
								vec3f up = mat3f::rotationY(horizRot) * mat3f::rotationX(vertRot) * vec3f::eY;

								Cameraf curr(eye, lookDir, up, fov, (float)renderWidth / (float)renderHeight, renderNear, renderFar);
								const Scene::RenderedView view = m_scene->renderToImage(curr, renderWidth, renderHeight);

								vec2f score = evaluateView(view, bDebug);
								if (score.x > minPercentPixObjects && score.y > 0.0f && math::randomUniform(0.0f, 1.0f) <= score.y) {
									bestCamera = curr;
									bestScore = score;
									break; // found a good enough one
								}
								else if (score.x > bestScore.x || (bestScore.x == 0 && score.y > bestScore.y)) {
									bestCamera = curr;
									bestScore = score;
								}
							}
							if (bestScore.x > 0.0f || bestScore.y > 0.0f) {
								bgCoverage.setVoxel(coverageLoc);
								trajectories.back().push_back(bestCamera);

								if (bDebug) {
									std::cout << "view " << trajectories.back().size() << ", score = " << bestScore << std::endl;
									std::string debugDir = "debug/";
									if (!util::directoryExists(debugDir)) util::makeDirectory(debugDir);
									const Scene::RenderedView view = m_scene->renderToImage(bestCamera, renderWidth, renderHeight);
									FreeImageWrapper::saveImage(debugDir + "view-" + std::to_string(trajectories.back().size()) + "_color.png", view.color);
									FreeImageWrapper::saveImage(debugDir + "view-" + std::to_string(trajectories.back().size()) + "_depth.png", ColorImageR32G32B32A32(view.depth));
								}
							}
							//sample up camera (todo angie fix messy)
							if (math::randomUniform(0.0f, 1.0f) < 0.5f) {//sample for best camera
								Cameraf bestCamera; vec2f bestScore = vec2f(0.0f, 0.0f);
								for (unsigned int c = 0; c < numCamsToTryPerCoverageVox; c++) {
									vec3f eye;
									eye.x = math::linearMap(0.0f, 1.0f, curBbox.getMinX(), curBbox.getMaxX(), math::randomUniform(0.0f, 1.0f));
									eye.z = math::linearMap(0.0f, 1.0f, curBbox.getMinZ(), curBbox.getMaxZ(), math::randomUniform(0.0f, 1.0f));
									eye.y = math::randomNormal(s_viewStats.camHeightMean, s_viewStats.camHeightStd);
									math::clamp(eye, curBbox.getMin(), curBbox.getMax());

									const float horizRot = math::randomUniform(0.0f, 360.0f);
									float vertRot = -1.5f * math::randomNormal(s_viewStats.camAngleMean, s_viewStats.camAngleStd);
									vec3f lookDir = mat3f::rotationY(horizRot) * mat3f::rotationX(vertRot) * vec3f::eZ;
									vec3f up = mat3f::rotationY(horizRot) * mat3f::rotationX(vertRot) * vec3f::eY;

									Cameraf curr(eye, lookDir, up, fov, (float)renderWidth / (float)renderHeight, renderNear, renderFar);
									const Scene::RenderedView view = m_scene->renderToImage(curr, renderWidth, renderHeight);

									vec2f score = evaluateView(view, bDebug);
									if (score.x > minPercentPixObjects && score.y > 0.0f && math::randomUniform(0.0f, 1.0f) <= score.y) {
										bestCamera = curr;
										bestScore = score;
										break; // found a good enough one
									}
									else if (score.x > bestScore.x || (bestScore.x == 0 && score.y > bestScore.y)) {
										bestCamera = curr;
										bestScore = score;
									}
								}
								if (bestScore.x > 0.0f || bestScore.y > 0.0f) {
									bgCoverage.setVoxel(coverageLoc);
									trajectories.back().push_back(bestCamera);

									if (bDebug) {
										std::cout << "view " << trajectories.back().size() << ", score = " << bestScore << std::endl;
										std::string debugDir = "debug/";
										if (!util::directoryExists(debugDir)) util::makeDirectory(debugDir);
										const Scene::RenderedView view = m_scene->renderToImage(bestCamera, renderWidth, renderHeight);
										FreeImageWrapper::saveImage(debugDir + "view-" + std::to_string(trajectories.back().size()) + "_color.png", view.color);
										FreeImageWrapper::saveImage(debugDir + "view-" + std::to_string(trajectories.back().size()) + "_depth.png", ColorImageR32G32B32A32(view.depth));
									}
								}
							}
						}//x
					}//y
				}//z

			}//rooms
		}//levels
		if (trajectories.back().empty()) trajectories.pop_back();

	}//trajectories
	
	return trajectories;
}