#pragma once

#include "GlobalAppState.h"
#include "LabelUtil.h"
#include "Lighting.h"
#include "json.h"
#include "VoxelGrid.h"
#include "BBInfo.h"

class RenderObject {
public:
	RenderObject(GraphicsDevice& g, const TriMeshf& triMesh, const Materialf& material, const mat4f& modelToWorld) {
		m_triMesh.init(g, triMesh);
		m_material = material;
		if (m_material.m_TextureFilename_Kd != "") {
			std::string texFile = m_material.m_TextureFilename_Kd;
			//that's a hack because I don't know how to load exrs (manually converted with imagemagick)
			if (util::getFileExtension(texFile) == "exr") {
				texFile = util::replace(texFile, ".exr", ".png");
			}

			if (util::fileExists(texFile)) {
				try {
					ColorImageR8G8B8A8 tex;
					FreeImageWrapper::loadImage(texFile, tex);
					m_texture.init(g, tex);
				}
				catch (const std::exception& e) {
					std::cout << texFile << " : " << e.what() << std::endl;
				}
			}
			else {
				std::cout << "can't file tex file " << texFile << std::endl;
			}
		}

		m_modelToWorld = modelToWorld;
		for (const auto& v : triMesh.getVertices())  {
			if (!isnan(v.position[0]))
				m_boundingBoxWorld.include(m_modelToWorld * v.position);
		}
	}

	~RenderObject() {
	}

	const mat4f& getModelToWorld() const {
		return m_modelToWorld;
	}

	const D3D11TriMesh& getD3D11TriMesh() const {
		return m_triMesh;
	}

	const D3D11Texture2D<vec4uc>& getD3D11Texture2D() const {
		return m_texture;
	}

	const BoundingBox3f& getBoundingBoxWorld() const {
		return m_boundingBoxWorld;
	}

	const Materialf& getMaterial() const {
		return m_material;
	}


	const bool isTextured() const {
		return m_texture.isInit();
	}

	void setModelToWorld(const mat4f& modelToWorld) {
		m_modelToWorld = modelToWorld;
		const auto& triMesh = m_triMesh.getTriMesh();
		for (const auto& v : triMesh.getVertices())  {
			if (!isnan(v.position[0]))
				m_boundingBoxWorld.include(m_modelToWorld * v.position);
		}
	}

	const unsigned short getLabelId() const {
		return (unsigned short)((int)std::round(m_triMesh.getTriMesh().m_vertices.front().color.w) / 1000);
	}

private:
	mat4f					m_modelToWorld;
	D3D11TriMesh			m_triMesh;
	D3D11Texture2D<vec4uc>	m_texture;
	BoundingBox3f			m_boundingBoxWorld;

	Materialf				m_material;
};

struct SceneInfo {
	std::string id;
	vec3f up;
	vec3f front;
	float scaleToMeters;
};



struct ObjectMaterial{
	std::unordered_map<std::string, std::string> data;
};

class SceneGraphNode {
public:
	SceneGraphNode() {
		id = "";
		type = "";
		valid = true;
		modelId = "";
		state = 0;
		transform = mat4f::identity();
		isMirrored = false;
		dimensions = vec3f(0.0f, 0.0f, 0.0f);
		hideCeiling = false;
		hideFloor = false;
		hideWalls = false;
	}

	std::string id;
	std::string type;
	bool valid;
	std::string modelId;
	bbox3f bbox;

	//for rooms
	std::vector<unsigned int> nodeIndices;
	std::vector<std::string> roomTypes;
	union {
		bool hideCeiling, hideFloor, hideWalls; //needs to match with mesh creation { "c", "f", "w" };
		bool hide[3];
	};

	//for objects
	int state;
	mat4f transform;
	bool isMirrored;
	std::vector<ObjectMaterial> materials;

	//for box
	vec3f dimensions;
};


class Scene
{
public:
	Scene() {
		m_graphics = nullptr;
	}

	~Scene() {

	}

	void loadFromJson(const std::string& filename, GraphicsDevice& g, const GlobalAppState& gas, bool printWarnings = false);

	//! debugging only
	void loadMesh(const std::string& filename, GraphicsDevice& g, const GlobalAppState& gas, bool printWarnings = false) {
		m_graphics = &g;

		m_cbCamera.init(g);
		m_cbMaterial.init(g);
		m_lighting.loadFromGlobaAppState(g, gas);

		//TODO factor our the shader loading; ideally every object has a shader
		m_shaders.init(g);
		m_shaders.registerShader("shaders/phong.hlsl", "phong", "vertexShaderMain", "vs_4_0", "pixelShaderMain", "ps_4_0");
		m_shaders.registerShader("shaders/phong.hlsl", "phong_textured", "vertexShaderMain", "vs_4_0", "pixelShaderMain_textured", "ps_4_0");

		MeshDataf meshDataAll = MeshIOf::loadFromFile(filename, true);
		const bbox3f meshBbox = meshDataAll.computeBoundingBox();

		std::vector< std::pair <MeshDataf, Materialf > > meshDataByMaterial = meshDataAll.splitByMaterial();
		for (auto& m : meshDataByMaterial) {

			MeshDataf& meshData = m.first;
			Materialf& material = m.second;

			MLIB_ASSERT(meshData.isConsistent());
			if (!meshData.isTriMesh()) {
				if (printWarnings) std::cout << "Warning mesh " << filename << " contains non-tri faces (auto-converting)" << std::endl;
				meshData.makeTriMesh();
			}
			MLIB_ASSERT(meshData.isConsistent());
			if (meshData.m_Colors.size() == 0) meshData.m_Colors.resize(meshData.m_Vertices.size(), vec4f(1.0f, 1.0f, 1.0f, 1.0f));	//set default color if none present
			TriMeshf triMesh(meshData);
			if (!triMesh.hasNormals())	triMesh.computeNormals();

			material.m_ambient = vec4f(0.0f, 1.0f, 0.0f, 1.0f);

			std::string path = util::directoryFromPath(filename);
			if (material.m_TextureFilename_Kd != "") material.m_TextureFilename_Kd = path + material.m_TextureFilename_Kd;
			addObject(triMesh, material);
		}
		if (m_boundingBox.getExtentY() < 0.2f) m_boundingBox.setMaxY(m_boundingBox.getMinY() + 1.0f); //inflate the bbox, debugging
	}

	template<typename T>
	void parseSceneGraphNode(const rapidjson::GenericValue<T>& d, SceneGraphNode& node) {
		node.id = d["id"].GetString();
		if (d.HasMember("bbox")) json::fromJSON(d["bbox"], node.bbox);
		//if (d.HasMember("valid")) node.valid = (bool)d["valid"].GetInt();
		if (d.HasMember("valid")) node.valid = (d["valid"].GetInt() > 0) ? true : false;
		if (d.HasMember("modelId")) node.modelId = d["modelId"].GetString();
		if (d.HasMember("type")) {
			node.type = d["type"].GetString();
			if (node.type == "Room") {
				if (d.HasMember("nodeIndices")) {
					const auto& nodeIndices = d["nodeIndices"];
					node.nodeIndices.resize(nodeIndices.Size());
					for (unsigned int i = 0; i < nodeIndices.Size(); i++)
						node.nodeIndices[i] = (unsigned int)nodeIndices[i].GetInt();
				}
				if (d.HasMember("roomTypes")) {
					const auto& roomTypes = d["roomTypes"];
					node.roomTypes.resize(roomTypes.Size());
					for (unsigned int i = 0; i < roomTypes.Size(); i++)
						node.roomTypes[i] = roomTypes[i].GetString();
				}
				if (d.HasMember("hideCeiling")) node.hideCeiling = (d["hideCeiling"].GetInt() > 0) ? true : false;
				if (d.HasMember("hideFloor")) node.hideFloor = (d["hideFloor"].GetInt() > 0) ? true : false;
				if (d.HasMember("hideWalls")) node.hideWalls = (d["hideWalls"].GetInt() > 0) ? true : false;
			}
			else if (node.type == "Object") {
				if (d.HasMember("state"))		node.state = d["state"].GetInt();
				//if (d.HasMember("isMirrored"))	node.isMirrored = (bool)d["isMirrored"].GetInt();
				if (d.HasMember("isMirrored"))	node.isMirrored = (d["isMirrored"].GetInt() > 0) ? true : false;
				if (d.HasMember("transform"))	json::fromJSON(d["transform"], node.transform);
				if (d.HasMember("materials")) {
					const auto& materials = d["materials"];
					node.materials.resize(materials.Size());
					for (unsigned int i = 0; i < materials.Size(); i++)
						json::fromJSON(materials[i], node.materials[i].data);
				}
			}
			else if (node.type == "Box") {
				if (d.HasMember("dimensions")) json::fromJSON(d["dimensions"], node.dimensions);
				if (d.HasMember("transform"))	json::fromJSON(d["transform"], node.transform);
				if (d.HasMember("materials")) {
					const auto& materials = d["materials"];
					node.materials.resize(materials.Size());
					for (unsigned int i = 0; i < materials.Size(); i++)
						json::fromJSON(materials[i], node.materials[i].data);
				}
			}
			//else if (node.type == "Ground") {} //nothing special for ground
		}
	}

	void saveSceneMesh(const std::string& filename, bool bExcludeCeilings = false) const {

		std::cout << "saving to file " << filename << std::endl;
		MeshDataf md;
		std::vector<Materialf> mats;
		computeSceneMesh(md, mats, bExcludeCeilings);

		if (util::getFileExtension(filename) == "obj") {
			md.m_materialFile = util::removeExtensions(filename) + ".mtl";
			Materialf::saveToMTL(md.m_materialFile, mats);
		}
		else { //apply semantic colors
			for (auto& c : md.m_Colors) {
				if (c.w == 0) c = vec4f(0.0f, 0.0f, 0.0f, 0.0f); //no label
				else {
					RGBColor color = RGBColor::colorPalette((unsigned int)std::round(c.w));
					c = vec4f(vec3f(color), 1.0f);
				}
			}
		}

		MeshIOf::saveToFile(filename, md);
	}

	void loadFromGlobaAppState(GraphicsDevice& g, const GlobalAppState& gas) {

		const bool bIgnoreNans = true;
		const std::vector<std::string> meshFilenames = gas.s_meshFilenames;
		m_graphics = &g;

		m_cbCamera.init(g);
		m_cbMaterial.init(g);
		m_lighting.loadFromGlobaAppState(g, gas);

		//TODO factor our the shader loading; ideally every object has a shader
		m_shaders.init(g);
		m_shaders.registerShader("shaders/phong.hlsl", "phong", "vertexShaderMain", "vs_4_0", "pixelShaderMain", "ps_4_0");
		m_shaders.registerShader("shaders/phong.hlsl", "phong_textured", "vertexShaderMain", "vs_4_0", "pixelShaderMain_textured", "ps_4_0");

		for (const std::string& meshFilename : meshFilenames) {
			MeshDataf meshDataAll = MeshIOf::loadFromFile(meshFilename, bIgnoreNans);

			std::vector< std::pair <MeshDataf, Materialf > > meshDataByMaterial = meshDataAll.splitByMaterial();

			for (auto& m : meshDataByMaterial) {

				MeshDataf& meshData = m.first;
				Materialf& material = m.second;

				MLIB_ASSERT(meshData.isConsistent());
				if (!meshData.isTriMesh()) {
					std::cout << "Warning mesh " << meshFilename << " contains non-tri faces (auto-converting)" << std::endl;
					meshData.makeTriMesh();
				}


				MLIB_ASSERT(meshData.isConsistent());
				if (meshData.m_Colors.size() == 0) meshData.m_Colors.resize(meshData.m_Vertices.size(), vec4f(1.0f, 1.0f, 1.0f, 1.0f));	//set default color if none present
				TriMeshf triMesh(meshData);
				if (!triMesh.hasNormals())	triMesh.computeNormals();

				material.m_ambient = vec4f(0.1f);

				std::string path = util::directoryFromPath(meshFilename);
				if (material.m_TextureFilename_Kd != "") material.m_TextureFilename_Kd = path + material.m_TextureFilename_Kd;
				addObject(triMesh, material);
			}

		}


	}

	void addObject(const TriMeshf& triMesh, const Materialf& material, const mat4f& modelToWorld = mat4f::identity()) {
		m_objects.emplace_back(RenderObject(*m_graphics, triMesh, material, modelToWorld));
		m_boundingBox.include(m_objects.back().getBoundingBoxWorld());
	}

	void computeSceneMesh(MeshDataf& md, std::vector<Materialf>& mats = std::vector<Materialf>(), bool bExcludeCeilings = false) const {
		md.clear();
		size_t i = 0;
		for (auto &o : m_objects) {
			MeshDataf::GroupIndex gi;
			gi.start = md.m_FaceIndicesVertices.size();

			MeshDataf curr = o.getD3D11TriMesh().getTriMesh().computeMeshData();
			if (bExcludeCeilings && curr.m_Colors.front().w == 2) continue; // exclude ceilings
			if (o.getD3D11TriMesh().getTriMesh().hasTexCoords()) {
				curr.m_FaceIndicesTextureCoords = curr.m_FaceIndicesVertices;
			}
			if (!o.getD3D11TriMesh().getTriMesh().hasTexCoords()) {
				curr.m_TextureCoords.resize(curr.m_Vertices.size(), vec2f(0.0f));
				curr.m_FaceIndicesTextureCoords = curr.m_FaceIndicesVertices;
			}
			curr.applyTransform(o.getModelToWorld());
			md.merge(curr);

			gi.end = md.m_FaceIndicesVertices.size();
			std::string matName = "material_" + std::to_string(i);	//o.getMaterial().m_name
			gi.name = matName;
			md.m_indicesByGroup.push_back(gi);
			md.m_indicesByMaterial.push_back(gi);

			Materialf m = o.getMaterial();
			m.m_name = matName;
			mats.push_back(m);
			i++;
		}
	}

	bool intersectsCameraBox(const Cameraf& camera, float bboxRadius, float viewDirThresh) const;

	const std::vector<BBInfo>& getObjectBBs() const { return m_objectBBs; }
	MeshDataf computeOBBMeshVis(const std::vector<BBInfo>& obbs, const vec4f& color = vec4f(1.0f, 0.0f, 0.0f, 1.0f)) const;
	MeshDataf computeAABBMeshVis(const std::vector<BBInfo>& obbs, const vec4f& color = vec4f(1.0f, 0.0f, 0.0f, 1.0f), float radius = 0.1f) const;

	void augment(GraphicsDevice& g, const std::string& outCacheFile);

	//! renders to the currently-bound render target
	void render(const Cameraf& camera) {

		m_lighting.updateAndBind(2);

		for (const RenderObject& o : m_objects) {
			ConstantBufferCamera cbCamera;
			cbCamera.worldViewProj = camera.getProj() * camera.getView() * o.getModelToWorld();
			cbCamera.world = o.getModelToWorld();
			cbCamera.eye = vec4f(camera.getEye());
			m_cbCamera.updateAndBind(cbCamera, 0);

			const Materialf material = o.getMaterial();

			ConstantBufferMaterial cbMaterial;
			cbMaterial.ambient = material.m_ambient;
			cbMaterial.diffuse = material.m_diffuse;
			cbMaterial.specular = material.m_specular;
			cbMaterial.shiny = material.m_shiny;
			m_cbMaterial.updateAndBind(cbMaterial, 1);

			if (o.isTextured()) {
				o.getD3D11Texture2D().bind(0);
				m_shaders.bindShaders("phong_textured");
			}
			else {
				m_shaders.bindShaders("phong");
			}
			o.getD3D11TriMesh().render();

			if (o.isTextured()) {
				o.getD3D11Texture2D().unbind(0);
			}
		}

	}

	struct RenderedView {
		DepthImage32 depth;
		ColorImageR8G8B8A8 color;
		BaseImage<unsigned short> semanticInstance; 
		BaseImage<unsigned short> semanticLabel; 
		Cameraf camera;

		ColorImageR8G8B8 getColoredSemanticLabel() const {
			ColorImageR8G8B8 res(semanticLabel.getDimensions());
			for (const auto& p : semanticLabel) {
				if (p.value == 0) {
					res(p.x, p.y) = vec3uc(0, 0, 0);
				}
				else {
					RGBColor c = RGBColor::colorPalette(p.value);
					res(p.x, p.y) = vec3uc(c.r, c.g, c.b);
				}
			}
			return res;
		}
		ColorImageR8G8B8 getColoredSemanticInstance() const {
			ColorImageR8G8B8 res(semanticInstance.getDimensions());
			for (const auto& p : semanticInstance) {
				if (p.value == 0) {
					res(p.x, p.y) = vec3uc(0, 0, 0);
				}
				else {
					RGBColor c = RGBColor::colorPalette(p.value);
					res(p.x, p.y) = vec3uc(c.r, c.g, c.b);
				}
			}
			return res;
		}
	};

	//depth is already back-projected
	RenderedView renderToImage(const Cameraf& camera, unsigned int width, unsigned int height) {

		RenderedView ret;

		ret.camera = camera;

		ColorImageR32G32B32A32 color32;
		const std::vector< DXGI_FORMAT > formats = { DXGI_FORMAT_R32G32B32A32_FLOAT };
#pragma omp critical 
		{
			if (m_renderTarget.getWidth() != width || m_renderTarget.getHeight() != height) {
				m_renderTarget = D3D11RenderTarget(*m_graphics, width, height, formats);
			}

			m_renderTarget.bind();
			m_renderTarget.clear(ml::vec4f(0.0f, 0.0f, 0.0f, 0.0f));


			ret.camera.updateAspectRatio((float)m_renderTarget.getWidth() / m_renderTarget.getHeight());
			render(ret.camera);

			m_renderTarget.unbind();

			m_renderTarget.captureColorBuffer(color32);
			m_renderTarget.captureDepthBuffer(ret.depth);
		}

		//semantics are stored in alpha channel
		ret.color.allocate(width, height);
		ret.semanticLabel.allocate(width, height);
		ret.semanticInstance.allocate(width, height);
		for (auto& c : color32) {
			ret.color(c.x, c.y) = math::round(vec4f(c.value.getVec3() * 255.0f, 255.0f));
			unsigned int instanceLabel = (unsigned int)std::round(c.value.w);
			unsigned short instance = instanceLabel % 1000;
			unsigned short label = instanceLabel / 1000;
			ret.semanticLabel(c.x, c.y) = label;
			ret.semanticInstance(c.x, c.y) = instance;

			MLIB_ASSERT(label <= 2552);
			//MLIB_ASSERT(label > 0 == instance > 0);
		}

		mat4f projToCamera = ret.camera.getProj().getInverse();
		mat4f cameraToWorld = ret.camera.getView().getInverse();
		mat4f projToWorld = cameraToWorld * projToCamera;

		ret.depth.setInvalidValue(-std::numeric_limits<float>::infinity());
		for (auto &p : ret.depth) {
			if (p.value != 0.0f && p.value != 1.0f) {
				vec3f posProj = vec3f(m_graphics->castD3D11().pixelToNDC(vec2i((int)p.x, (int)p.y), ret.depth.getWidth(), ret.depth.getHeight()), p.value);
				vec3f posCamera = projToCamera * posProj;
				p.value = posCamera.z;
			}
			else {
				p.value = ret.depth.getInvalidValue();
			}
		}


		return ret;
	}

	const BoundingBox3f& getBoundingBox() const {
		return m_boundingBox;
	}

	void randomizeLighting() {
		m_lighting.randomize();
	}

	const Lighting& getLighting() const {
		return m_lighting;
	}

	void setLighting(const Lighting& l) {
		m_lighting = l;
	}

	//list of room bboxes by level
	void getRoomBboxes(std::vector<std::vector<bbox3f>>& bboxes);

	void computeInstances(VoxelGrid& grid, const TriMeshf& triMesh, const mat4f& worldToVoxel) const {
		voxelize(grid, triMesh, worldToVoxel, false, false);
	}

private:
	void clear() {
		m_objects.clear();
		m_boundingBox.reset();
		m_sceneGraph.clear();
		m_objectBBs.clear();
	}
	unsigned short computeInstance(const vec3ui& location, const VoxelGrid& labels) const;
	void voxelizeTriangle(const vec3f& v0, const vec3f& v1, const vec3f& v2, unsigned short id, VoxelGrid& grid, bool solid = false) const;
	void voxelize(VoxelGrid& grid, const TriMeshf& triMesh, const mat4f& worldToVoxel = mat4f::identity(), bool solid = false, bool verbose = true) const;

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

	float computeIntersection(const bbox3f& b0, const bbox3f& b1) const {
		bool intersects = false;
		for (const auto& p : b0.getVertices()) {
			if (b1.intersects(p)) {
				intersects = true;
				break;
			}
		}
		if (!intersects) {
			for (const auto& p : b1.getVertices()) {
				if (b0.intersects(p)) {
					intersects = true;
					break;
				}
			}
		}
		if (!intersects) {
			for (const auto& e : b0.getEdges()) {
				if (b1.intersect(Rayf(e.p0(), (e.p1() - e.p0()).getNormalized()), 0.0f, (e.p1() - e.p0()).length())){
					intersects = true;
					break;
				}
			}
			if (!intersects) {
				for (const auto& e : b1.getEdges()) {
					if (b0.intersect(Rayf(e.p0(), (e.p1() - e.p0()).getNormalized()), 0.0f, (e.p1() - e.p0()).length())){
						intersects = true;
						break;
					}
				}
			}
		}
		float xmin = std::max(b0.getMinX(), b1.getMinX());
		float ymin = std::max(b0.getMinY(), b1.getMinY());
		float zmin = std::max(b0.getMinZ(), b1.getMinZ());
		float xmax = std::min(b0.getMaxX(), b1.getMaxX());
		float ymax = std::min(b0.getMaxY(), b1.getMaxY());
		float zmax = std::min(b0.getMaxZ(), b1.getMaxZ());
		return intersects ? (xmax - xmin) * (ymax - ymin) * (zmax - zmin) : 0.0f;
	}


	GraphicsDevice* m_graphics;

	D3D11ShaderManager m_shaders;

	std::vector<RenderObject> m_objects;
	BoundingBox3f m_boundingBox;


	D3D11ConstantBuffer<ConstantBufferCamera>	m_cbCamera;
	D3D11ConstantBuffer<ConstantBufferMaterial> m_cbMaterial;
	Lighting m_lighting;

	D3D11RenderTarget m_renderTarget;

	SceneInfo m_sceneInfo;
	std::list<std::list<SceneGraphNode>> m_sceneGraph;
	std::vector<BBInfo> m_objectBBs;
	std::vector<std::vector<unsigned int>> m_objBBRenderObjectIdxes; // same size as m_objectBBs, indexes into m_objects
	std::vector<mat4f> m_curAugmentedTransforms;
	std::vector<float> m_curAugmentedAngles;
	std::vector<BinaryGrid3> m_objMasksOrig;
};