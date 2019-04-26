
#include "GlobalAppState.h"
#include "Scene.h"


struct ViewStats {
	float camHeightMean;
	float camHeightStd;
	float camAngleMean;
	float camAngleStd;

	float percentValidDepthMean;
	float percentValidDepthStd;

	std::vector<float> depthHistMean;
	std::vector<float> depthHistStd;
	float depthHistMedEMD;
	float depthHistStdEMD;
};


class ViewGenerator {
public:
	ViewGenerator(Scene* scene) {
		m_scene = scene;
	}

	~ViewGenerator() {
	}

	//debug function
	void visualizeCameraCoverageGrid(const std::vector<Cameraf>& cameras, const std::string& filename) const {
		BoundingBox3f bb = m_scene->getBoundingBox();
		const float coverageVoxSize = 0.6f;// 0.75f; //meters
		BinaryGrid3 coverageGrid(math::ceil(bb.getExtent() / coverageVoxSize));
		MeshDataf md;
		for (const auto& cam : cameras) {
			const vec3f focus = cam.getEye() + cam.getLook() * 1.5f; //2m in front of eye
			const vec3ui coverageLoc = math::round((focus - bb.getMin()) / coverageVoxSize);
			coverageGrid.setVoxel(coverageLoc);

			bbox3f gridBox(vec3f(coverageLoc)*coverageVoxSize + bb.getMin(), (vec3f(coverageLoc) + 1)*coverageVoxSize + bb.getMin());
			md.merge(Shapesf::wireframeBox(gridBox.cubeToWorldTransform(), vec4f(0.0f, 0.0f, 1.0f, 1.0f), 0.05f).computeMeshData());
			md.merge(makeCameraMesh(cam, RGBColor::Red.toVec4f()));
		}
		md.merge(Shapesf::wireframeBox(bb.cubeToWorldTransform(), vec4f(0.0f, 1.0f, 0.0f, 1.0f), 0.05f).computeMeshData());
		std::cout << "covered " << coverageGrid.getNumOccupiedEntries() << " of " << coverageGrid.getNumElements() << std::endl;
		MeshIOf::saveToFile(filename, md);
	}

	std::vector<std::vector<Cameraf>> genViews(unsigned int maxNumTraj, unsigned int numViews, unsigned int maxNumViewSamples, bool bSampleEntireScene, bool bDebug = false) const {
		float fov = 60.0f;
		unsigned int renderWidth = GlobalAppState::get().s_renderWidth;
		unsigned int renderHeight = GlobalAppState::get().s_renderHeight;
		float renderNear = GlobalAppState::get().s_renderNear;
		float renderFar = GlobalAppState::get().s_renderFar;

		std::vector<std::vector<Cameraf>> trajectories;

		BoundingBox3f bb = m_scene->getBoundingBox();
		const float coverageVoxSize = 1.5f;//2.0f; //meters
		vec3ui coverageDims = math::ceil(bb.getExtent() / coverageVoxSize);
		coverageDims.y = 1; //don't bother with higher stuff, there won't be any scannet camera stats for this anyways
		const unsigned int numCamsToTryPerCoverageVox = std::min(10u, maxNumViewSamples / (coverageDims.z*coverageDims.x));
		const float minPercentPixObjects = 0.2f; //TODO params

		for (unsigned int t = 0; t < maxNumTraj; t++) {
			trajectories.push_back(std::vector<Cameraf>());
			//traverse in zigzag/scanline order
			for (unsigned int y = 0; y < coverageDims.y; y++) {
				for (unsigned int z = 0; z < coverageDims.z; z++) {
					for (unsigned int _x = 0; _x < coverageDims.x; _x++) {
						unsigned int x = (z % 2) == 0 ? _x : coverageDims.x - _x - 1; //iterate forward/backward
						vec3ui coverageLoc(x, y, z);
						bbox3f curBbox(vec3f(coverageLoc)*coverageVoxSize + bb.getMin(), vec3f(coverageLoc + 1)*coverageVoxSize + bb.getMin());

						Cameraf bestCamera; float bestScore = 0.0f;
						for (unsigned int c = 0; c < numCamsToTryPerCoverageVox; c++) {
							vec3f eye;
							eye.x = math::linearMap(0.0f, 1.0f, curBbox.getMinX(), curBbox.getMaxX(), math::randomUniform(0.0f, 1.0f));
							eye.z = math::linearMap(0.0f, 1.0f, curBbox.getMinZ(), curBbox.getMaxZ(), math::randomUniform(0.0f, 1.0f));
							eye.y = math::randomNormal(s_viewStats.camHeightMean, s_viewStats.camHeightStd);
							math::clamp(eye, curBbox.getMin(), curBbox.getMax());

							const float horizRot = math::randomUniform(0.0f, 360.0f);
							float vertRot = math::randomNormal(s_viewStats.camAngleMean, s_viewStats.camAngleStd);
							//if (math::randomUniform(0.0f, 1.0f) < 0.2f) vertRot = -2*vertRot; //TODO ANGIE HERE
							vec3f lookDir = mat3f::rotationY(horizRot) * mat3f::rotationX(vertRot) * vec3f::eZ;
							vec3f up = mat3f::rotationY(horizRot) * mat3f::rotationX(vertRot) * vec3f::eY;

							Cameraf curr(eye, lookDir, up, fov, (float)renderWidth / (float)renderHeight, renderNear, renderFar);
							const Scene::RenderedView view = m_scene->renderToImage(curr, renderWidth, renderHeight);

							throw MLIB_EXCEPTION("need to adapt view evaluation to nyu label mapping");
							vec2f scores = evaluateView(view, bDebug);
							float score = scores.x > minPercentPixObjects ? scores.y : 0.0f;
							if (score > 0.0f && math::randomUniform(0.0f, 1.0f) <= score) {
								bestCamera = curr;
								bestScore = score;
								break; // found a good enough one
							}
							else if (score > bestScore) {
								bestCamera = curr;
								bestScore = score;
							}
						}
						if (bestScore > 0.0f) {
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
					} //x
				} //y
			} //z
			if (bDebug) std::cout << "took " << trajectories.back().size() << " cameras of " << maxNumViewSamples << std::endl;
			if (trajectories.back().empty()) trajectories.pop_back();
		} //traj

		return trajectories;
	}

	std::vector<std::vector<Cameraf>> generateKeys(unsigned int maxNumTraj, unsigned int numViews, unsigned int maxNumViewSamples, bool bSampleEntireScene, 
		bool bDebug = false) const;

	std::vector<std::vector<Cameraf>> genTrajectories(unsigned int maxNumTraj, unsigned int maxNumViewPerTraj, unsigned int numKeyViews, unsigned int maxNumViewSamples, bool bSampleEntireScene,
		bool bDebug = false) const;

	static void initViewStats(const std::string& filename)
	{
		std::ifstream s(filename);
		if (!s.is_open()) throw MLIB_EXCEPTION("failed to open file (" + filename + ") for read");
		std::string line; bool bFoundMean = false, bFoundStddev = false; bool bFoundMed = false;

		//read header
		std::getline(s, line);
		{ //depth hist 
			float depthHistStart, depthHistInc; unsigned int depthHistBins;
			std::getline(s, line);
			const auto parts = util::split(line, ',');
			depthHistStart = util::convertTo<float>(parts[0]);
			depthHistInc = util::convertTo<float>(parts[1]);
			depthHistBins = util::convertTo<unsigned int>(parts[2]);
			s_viewStats.depthHistMean.resize(depthHistBins);
			s_viewStats.depthHistStd.resize(depthHistBins);
		}
		std::getline(s, line);
		while (std::getline(s, line)) {
			const auto parts = util::split(line, ',');
			if (parts.front() == "AVERAGE") {
				s_viewStats.camHeightMean = util::convertTo<float>(util::split(parts[1], ' ').back()); //camera translation z
				s_viewStats.camAngleMean = math::radiansToDegrees(util::convertTo<float>(parts[2])); //camera angle (+ => down)
				s_viewStats.percentValidDepthMean = util::convertTo<float>(parts[3]);
				for (unsigned int i = 0; i < s_viewStats.depthHistMean.size(); i++)
					s_viewStats.depthHistMean[i] = util::convertTo<float>(parts[5 + i]);
				bFoundMean = true;
			}
			else if (parts.front() == "STDDEV") {
				s_viewStats.camHeightStd = util::convertTo<float>(util::split(parts[1], ' ').back()); //camera translation z
				s_viewStats.camAngleStd = math::radiansToDegrees(util::convertTo<float>(parts[2])); //camera angle (+ => down)
				s_viewStats.percentValidDepthStd = util::convertTo<float>(parts[3]);
				s_viewStats.depthHistStdEMD = util::convertTo<float>(parts[4]);
				for (unsigned int i = 0; i < s_viewStats.depthHistStd.size(); i++)
					s_viewStats.depthHistStd[i] = util::convertTo<float>(parts[5 + i]);
				bFoundStddev = true;
			}
			else if (parts.front() == "MEDIAN") {
				s_viewStats.depthHistMedEMD = util::convertTo<float>(parts[4]);
				bFoundMed = true;
			}
			if (bFoundMean && bFoundStddev && bFoundMed) break;
		}
		s.close();
		if (!bFoundMean || !bFoundStddev || !bFoundMed) throw MLIB_EXCEPTION("failed to read mean/stddev from view stat file");
	}

private:

	// -- debug vis
	PointCloudf makePointCloud(const Scene::RenderedView& view) const {
		PointCloudf pc;
		const mat4f intrinsicInv = view.camera.getIntrinsic(view.depth.getWidth(), view.depth.getHeight()).getInverse();
		for (unsigned int y = 0; y < view.depth.getHeight(); y++) {
			for (unsigned int x = 0; x < view.depth.getWidth(); x++) {
				const float d = view.depth(x, y);
				if (d != view.depth.getInvalidValue()) {
					vec3f cameraPos = (intrinsicInv*vec4f((float)x*d, (float)y*d, d, 0.0f)).getVec3();
					//vec3f worldPos = view.camera.getExtrinsic() * cameraPos;
					pc.m_points.push_back(cameraPos);
					pc.m_colors.push_back(vec4f(view.color(x, y)) / 255.0f);
				}
			}
		}
		return pc;
	}
	MeshDataf makeCameraMesh(const Cameraf& cam, const vec4f& color, float size = 0.1f) const {
		MeshDataf md;
		md.merge(Shapesf::sphere(0.4f * size, cam.getEye(), 64, 64, color).computeMeshData());
		md.merge(Shapesf::cylinder(cam.getEye(), cam.getEye() + 0.5f * cam.getLook(), 0.1f * size, 10, 10, color).computeMeshData());
		md.merge(Shapesf::cylinder(cam.getEye(), cam.getEye() + 0.5f * cam.getUp(), 0.1f * size, 10, 10, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
		return md;
	}
	MeshDataf makeCamerasMesh(const std::vector<Cameraf>& cameras, float size = 0.1f) const {
		MeshDataf md;
		for (unsigned int i = 0; i < cameras.size(); i++) {
			vec4f color = BaseImageHelper::convertDepthToRGBA((float)i, 0.0f, (float)cameras.size() - 1.0f);
			md.merge(makeCameraMesh(cameras[i], color, size));
		}
		return md;
	}

	// -- view evaluation
	vec2f evaluateView(const Scene::RenderedView& view, bool bDebug = false) const {
		const float viewEvalMaxDepth = GlobalAppState::get().s_viewEvalMaxDepth;

		unsigned int numValid = 0;
		unsigned int numPixObjects = 0;

		const float histogramScale = 0.5f;	//0.5f m per bin
		const unsigned int histogramSize = 16;
		const float invHistScale = 1.0f / histogramScale;
		std::vector<float> histogram(histogramSize, 0.0f);
		for (auto d : view.depth) {
			if (d.value != view.depth.getInvalidValue()) {
				numValid++;
				unsigned int histValue = math::clamp((unsigned int)(d.value * invHistScale), 0u, histogramSize - 1);
				histogram[histValue]++;

				int semantic = view.semanticLabel(d.x, d.y);
				//if (semantic > 0 && d.value < viewEvalMaxDepth && !LabelUtil::get().isExcludedClass(semantic))
				if (semantic > 0 && d.value < viewEvalMaxDepth)
					numPixObjects++;
			}
		}

		vec2f score; //object score, depth hist score

		//--compute object score--
		score.x = (float)numPixObjects / (float)view.semanticLabel.getNumPixels();

		//--compute depth hist score--
		for (size_t i = 0; i < histogram.size(); i++) {
			histogram[i] = (float)histogram[i] / (float)view.depth.size();
			//histogram[i] = (float)histogram[i] / (float)numValid;
		}
		float emd = computeEMD(histogram, s_viewStats.depthHistMean);
		float p = computeProbability(emd, s_viewStats.depthHistMedEMD, s_viewStats.depthHistStdEMD);
		const float multFactor = score.x > 0.5f ? 2.0f : 1.5f;

		if (bDebug) { //debugging
			float minDepth = std::numeric_limits<float>::infinity();
			float maxDepth = -std::numeric_limits<float>::infinity();
			for (const auto& d : view.depth) {
				if (d.value != view.depth.getInvalidValue()) {
					if (d.value < minDepth) minDepth = d.value;
					if (d.value > maxDepth) maxDepth = d.value;
				}
			}
			std::cout << "hist, mean" << std::endl;
			float l2norm = 0.0f, l1norm = 0.0f;;
			for (unsigned int i = 0; i < histogram.size(); i++) {
				l2norm += (histogram[i] - s_viewStats.depthHistMean[i]) * (histogram[i] - s_viewStats.depthHistMean[i]);
				l1norm += std::fabs(histogram[i] - s_viewStats.depthHistMean[i]);
				std::cout << " " << histogram[i];
			}
			std::cout << std::endl;
			for (unsigned int i = 0; i < s_viewStats.depthHistMean.size(); i++)
				std::cout << " " << s_viewStats.depthHistMean[i];
			std::cout << std::endl;
			std::cout << "depth range = [ " << minDepth << ", " << maxDepth << " ]" << std::endl;
			std::cout << "%object =" << score.x << std::endl;;
			std::cout << "l2 norm =" << std::sqrt(l2norm) << std::endl;
			std::cout << "l1 norm =" << l1norm << std::endl;
			std::cout << "emd     =" << emd << std::endl;
			std::cout << "p(emd)  =" << p << std::endl;
		} //debugging

		if (p < 0.1f) score.y = 0.0f;
		score.y = std::min(1.0f, multFactor*p);
		return score;
	}

	static float computeEMD(const std::vector<float>& p, const std::vector<float>& q)
	{
		std::vector<float> emds(p.size());
		emds[0] = 0.0f;
		float dist = 0.0f;
		for (unsigned int i = 1; i < p.size(); i++) {
			emds[i] += (p[i - 1] + emds[i - 1]) - q[i - 1];
			dist += std::fabs(emds[i]);
		}
		return dist;
	}
	static float computeProbability(float score, float mean, float std)
	{
		const float z = (score - mean) / std;

		if (z < 0) return std::erfc(-z * M_SQRT1_2);
		else       return std::erfc(z * M_SQRT1_2);
	}

	// -- view interpolation
	static void catmullRomSpline(const Cameraf& c0, const Cameraf& c1, const Cameraf& c2, const Cameraf& c3, unsigned int numPoints, std::vector<Cameraf>& cameras)
	{
		const float fov = c0.getFoV();
		const float aspect = c0.getAspect();
		const float zNear = c0.getNearPlane();
		const float zFar = c0.getFarPlane();
		const float eps = 0.001f;

		//quat rotations
		const mat4f m1 = c1.getExtrinsic();
		const mat4f m2 = c2.getExtrinsic();
		const mat4f t12 = m2 * m1.getInverse();
		const quatf q1(1, 0, 0, 0); //no rotation
		const quatf q2(t12.getRotation());

		//translations
		const vec3f p0(c0.getEye());
		const vec3f p1(c1.getEye());
		const vec3f p2(c2.getEye());
		const vec3f p3(c3.getEye());

		if (vec3f::dist(p1, p2) < 0.001f) {
			cameras.resize(numPoints);
			for (unsigned int i = 0; i < numPoints; i++) {
				const quatf q = q1.slerp(q2, (float)i / (float)(numPoints - 1));
				mat4f ext = mat4f::identity();
				ext.setRotationMatrix(q.matrix3x3() * m1.getRotation());
				ext.setTranslationVector(p1);
				cameras[i] = Cameraf(ext, fov, aspect, zNear, zFar);
			}
			return;
		}

		// calculate t
		const float alpha = 0.5f;
		const float t0 = 0.0f;
		const float t1 = catmullRomTj(alpha, t0, p0, p1);
		const float t2 = catmullRomTj(alpha, t1, p1, p2);
		const float t3 = catmullRomTj(alpha, t2, p2, p3);

		std::vector<vec3f> newpoints;
		for (float t = t1; t < t2; t += (t2 - t1) / (float)numPoints) {
			const vec3f A1 = (t1 - t) / std::max(eps, t1 - t0) * p0 + (t - t0) / std::max(eps, t1 - t0) * p1;
			const vec3f A2 = (t2 - t) / std::max(eps, t2 - t1)*p1 + (t - t1) / std::max(eps, t2 - t1)*p2;
			const vec3f A3 = (t3 - t) / std::max(eps, t3 - t2)*p2 + (t - t2) / std::max(eps, t3 - t2)*p3;

			const vec3f B1 = (t2 - t) / std::max(eps, t2 - t0)*A1 + (t - t0) / std::max(eps, t2 - t0)*A2;
			const vec3f B2 = (t3 - t) / std::max(eps, t3 - t1)*A2 + (t - t1) / std::max(eps, t3 - t1)*A3;

			const vec3f C = (t2 - t) / std::max(eps, t2 - t1)*B1 + (t - t1) / std::max(eps, t2 - t1)*B2;
			newpoints.push_back(C);
		}

		cameras.resize(newpoints.size());
		for (unsigned int i = 0; i < newpoints.size(); i++) {
			const quatf q = q1.slerp(q2, (float)i / (float)(newpoints.size() - 1));
			mat4f ext = mat4f::identity();
			ext.setRotationMatrix(q.matrix3x3() * m1.getRotation());
			ext.setTranslationVector(newpoints[i]);
			cameras[i] = Cameraf(ext, fov, aspect, zNear, zFar);
		}
	}
	static float catmullRomTj(float alpha, float ti, const vec3f& pi, const vec3f& pj) {
		return std::pow(vec3f::dist(pj, pi), alpha) + ti;
	}

	Scene* m_scene;

	static ViewStats s_viewStats;
};

