#pragma once

#include "GlobalAppState.h"
#include "LabelUtil.h"
#include "ScansDirectory.h"
#include "Lighting.h"
#include "json.h"
#include "Segmentation.h"
#include "Aggregation.h"
#include "BBInfo.h"

class Scene
{
public:
	Scene() {
		m_graphics = nullptr;
		m_transformFromOrig = mat4f::identity();
	}

	~Scene() {
		clear();
	}

	void load(GraphicsDevice& g, const ScanInfo& scanInfo, bool bGenerateTestOnly, mat4f transform = mat4f::identity());

	const bbox3f& getBoundingBox() const {
		return m_bb;
	}
	const OBB3f& getOBB() const {
		return m_obb;
	}
	size_t getNumFrames() const {
		return m_linearizedSensFrameIds.size();
	}
	bool getDepthFrame(unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth = 0.0f, float maxDepth = 12.0f) const;
	bool renderSemanticsFrame(GraphicsDevice& g, unsigned int idx, const DepthImage32& origDepth, BaseImage<unsigned char>& semantics, BaseImage<unsigned char>& instance);
	bool renderDepthSemantics(GraphicsDevice& g, unsigned int idx, mat4f& intrinsic, mat4f& extrinsic, DepthImage32& depth, BaseImage<unsigned char>& semantics, BaseImage<unsigned char>& instance);

	void randomizeLighting() {
		m_lighting.randomize();
	}

	const Lighting& getLighting() const {
		return m_lighting;
	}

	void setLighting(const Lighting& l) {
		m_lighting = l;
	}

	// trajectory frameIds <-> linearized sens frame idss
	void computeTrajFramesInScene(std::vector<unsigned int>& frameIds) const {
		frameIds.clear();
		for (unsigned int i = 0; i < m_linearizedSensFrameIds.size(); i++) {
			const auto& sensFrameId = m_linearizedSensFrameIds[i];
			const mat4f& transform = m_sensDatas[sensFrameId.x]->m_frames[sensFrameId.y].getCameraToWorld();
			if (m_obb.intersects(transform.getTranslation()))
				frameIds.push_back(i);
		}

		////debugging
		//std::vector<mat4f> camerasInScene, camerasNotInScene;
		//std::unordered_set<unsigned int> frameSet(frameIds.begin(), frameIds.end());
		//for (unsigned int i = 0; i < m_linearizedSensFrameIds.size(); i++) {
		//	const auto& sensFrameId = m_linearizedSensFrameIds[i];
		//	const mat4f& transform = m_sensDatas[sensFrameId.x]->m_frames[sensFrameId.y].getCameraToWorld();
		//	if (frameSet.find(i) == frameSet.end())
		//		camerasNotInScene.push_back(transform);
		//	else
		//		camerasInScene.push_back(transform);
		//}
		//MeshIOf::saveToFile("_cam_in_scene.ply", makeCamerasMesh(camerasInScene, vec4f(0.0f, 1.0f, 0.0f, 1.0f)));
		//MeshIOf::saveToFile("_cam_not_in_scene.ply", makeCamerasMesh(camerasNotInScene, vec4f(1.0f, 1.0f, 0.0f, 1.0f)));
		//std::cout << "waiting..." << std::endl; getchar();
		////debugging
	}

	std::vector<BBInfo> getObjectBBs() const { return m_objectBBs; }

	const mat4f& getTransformFromOrig() const { return m_transformFromOrig; }
	const std::string& getName() const { return m_name; }

private:
	void clear() {
		m_sensFiles.clear();
		for (unsigned int i = 0; i < m_sensDatas.size(); i++)
			SAFE_DELETE(m_sensDatas[i]);
		m_sensDatas.clear();
		m_linearizedSensFrameIds.clear();
		m_bb.reset();
		m_obb.setInvalid();
		m_transformFromOrig = mat4f::identity();
	}

	void computeObjectIdsAndColorsPerVertex(const Aggregation& aggregation, const Segmentation& segmentation,
		MeshDataf& meshData);

	static MeshDataf makeCamerasMesh(const std::vector<mat4f>& cameras, const vec4f& eyeColor = vec4f(0.0f, 1.0f, 0.0f, 1.0f),
		const vec4f& lookColor = vec4f(1.0f, 0.0f, 0.0f, 1.0f), const vec4f& upColor = vec4f(0.0f, 0.0f, 1.0f, 1.0f)) {
		MeshDataf camMesh;
		for (const mat4f& cam : cameras) {
			const vec3f eye = cam.getTranslation();
			const vec3f look = cam.getRotation() * -vec3f::eZ;
			const vec3f up = cam.getRotation() * vec3f::eY;
			camMesh.merge(Shapesf::cylinder(eye, eye + 0.2f * look, 0.1f, 10, 10, lookColor).computeMeshData());
			camMesh.merge(Shapesf::cylinder(eye, eye + 0.2f * up, 0.1f, 10, 10, upColor).computeMeshData());
			camMesh.merge(Shapesf::sphere(0.1f, eye, 10, 10, eyeColor).computeMeshData());
		}
		return camMesh;
	}

	struct ConstantBufferCamera {
		mat4f worldViewProj;
		mat4f world;
		vec4f eye;
	};

	struct ConstantBufferMaterial {
		vec4f ambient;
		vec4f diffuse;
		vec4f specular;
		float shiny;
		vec3f dummy;
	};

	std::string m_name;
	GraphicsDevice* m_graphics;

	D3D11ShaderManager m_shaders;

	std::vector<vec2ui> m_linearizedSensFrameIds;
	std::vector<std::string> m_sensFiles;
	std::vector<SensorData*> m_sensDatas;
	bbox3f m_bb;
	OBB3f m_obb;
	D3D11TriMesh m_mesh;
	mat4f m_transformFromOrig;

	std::vector<BBInfo> m_objectBBs;

	D3D11ConstantBuffer<ConstantBufferCamera>	m_cbCamera;
	D3D11ConstantBuffer<ConstantBufferMaterial> m_cbMaterial;
	Lighting m_lighting;

	D3D11RenderTarget m_renderTarget;
};

