
#include "stdafx.h"
#include "Scene.h"

void Scene::loadFromJson(const std::string& filename, GraphicsDevice& g, const GlobalAppState& gas, bool printWarnings /*= false*/) {
	//TODO add to global app state
	const std::string suncgPath = gas.s_suncgPath;
	const std::string modelDir = "object";
	const std::string roomDir = "room";
	const std::string textureDir = "texture";
	const bool bIgnoreNans = true;
	clear();

	rapidjson::Document d;
	if (!json::parseRapidJSONDocument(filename, &d)) {
		std::cerr << "Parse error reading " << filename << std::endl
			<< "Error code " << d.GetParseError() << " at " << d.GetErrorOffset() << std::endl;
		return;
	}

	m_sceneInfo.id = d["id"].GetString();
	json::fromJSON(d["up"], m_sceneInfo.up);
	json::fromJSON(d["front"], m_sceneInfo.front);
	m_sceneInfo.scaleToMeters = (float)d["scaleToMeters"].GetDouble();
	const auto& json_levels = d["levels"];
	for (unsigned int l = 0; l < json_levels.Size(); l++) {
		SceneGraphNode nodeLevel;
		parseSceneGraphNode(json_levels[l], nodeLevel);
		m_sceneGraph.push_back(std::list<SceneGraphNode>());
		m_sceneGraph.back().push_back(nodeLevel);

		const auto& json_nodes = json_levels[l]["nodes"];
		for (unsigned int n = 0; n < json_nodes.Size(); n++) {
			SceneGraphNode node;
			parseSceneGraphNode(json_nodes[n], node);
			m_sceneGraph.back().push_back(node);
		}
	}

	m_graphics = &g;

	m_cbCamera.init(g);
	m_cbMaterial.init(g);
	m_lighting.loadFromGlobaAppState(g, gas);

	//TODO factor our the shader loading; ideally every object has a shader
	m_shaders.init(g);
	m_shaders.registerShader("shaders/phong.hlsl", "phong", "vertexShaderMain", "vs_4_0", "pixelShaderMain", "ps_4_0");
	m_shaders.registerShader("shaders/phong.hlsl", "phong_textured", "vertexShaderMain", "vs_4_0", "pixelShaderMain_textured", "ps_4_0");

	unsigned int objectId = 0; // instance counter
	for (auto& level : m_sceneGraph) {
		const bbox3f bbox = level.front().bbox; //sometimes this can filter out random floating crap
		for (auto& node : level) {
			if (!node.valid) continue; //ignore invalid
			if (node.type == "") {
				continue;	//if it's the level root ignore it
			}
			else if (node.type == "Room") {
				std::vector<std::string> component = { "c", "f", "w" };
				std::vector<std::string> componentModelIds = { "Ceiling", "Floor", "Wall" };

				for (unsigned int i = 0; i < component.size(); i++) {
					const std::string& c = component[i];
					const std::string meshFilename = suncgPath + "/" + roomDir + "/" + m_sceneInfo.id + "/" + node.modelId + c + ".obj";
					if (node.hide[i] || !util::fileExists(meshFilename)) continue;
					MeshDataf meshDataAll = MeshIOf::loadFromFile(meshFilename, bIgnoreNans);
					const bbox3f meshBbox = meshDataAll.computeBoundingBox();
					if (!bbox.intersects(meshBbox)) { 
						if (printWarnings) std::cout << "warning: skipping mesh (" << node.modelId << ") that falls outside of level bbox" << std::endl; 
						node.valid = false;
						continue;
					} //skip since outside bbox
					//find label
					unsigned short label = 0, nyuId = 0;
					bool bValid = LabelUtil::get().getIdForLabel(componentModelIds[i], label);
					if (!bValid) throw MLIB_EXCEPTION("no label index for " + componentModelIds[i]);
					bValid = LabelUtil::get().getNyuIdForId(label, nyuId);

					std::vector< std::pair <MeshDataf, Materialf > > meshDataByMaterial = meshDataAll.splitByMaterial();
					for (auto& m : meshDataByMaterial) {

						MeshDataf& meshData = m.first;
						Materialf& material = m.second;

						MLIB_ASSERT(meshData.isConsistent());
						if (!meshData.isTriMesh()) {
							if (printWarnings) std::cout << "Warning mesh " << meshFilename << " contains non-tri faces (auto-converting)" << std::endl;
							meshData.makeTriMesh();
						}

						MLIB_ASSERT(meshData.isConsistent());
						if (meshData.m_Colors.size() == 0) meshData.m_Colors.resize(meshData.m_Vertices.size(), vec4f(1.0f, 1.0f, 1.0f, 0.0f));	//set default color if none present
						unsigned int instanceLabel = (unsigned int)nyuId * 1000u; // no instance available
						for (auto& c : meshData.m_Colors) c.w = (float)instanceLabel;
						TriMeshf triMesh(meshData);
						if (!triMesh.hasNormals())	triMesh.computeNormals();

						material.m_ambient = vec4f(0.1f);

						std::string path = util::directoryFromPath(meshFilename);
						if (material.m_TextureFilename_Kd != "") material.m_TextureFilename_Kd = path + material.m_TextureFilename_Kd;
						addObject(triMesh, material);
					}
				}
			}
			else if (node.type == "Object") {
				const std::string meshFilename = suncgPath + "/" + modelDir + "/" + node.modelId + "/" + node.modelId + ".obj";
				MeshDataf meshDataAll = MeshIOf::loadFromFile(meshFilename, bIgnoreNans);
				meshDataAll.applyTransform(node.transform);
				const bbox3f meshBbox = meshDataAll.computeBoundingBox();
				if (!bbox.intersects(meshBbox)) { if (printWarnings) std::cout << "warning: skipping mesh (" << node.modelId << ") that falls outside of level bbox" << std::endl; continue; } //skip since outside bbox
				objectId++;
				//find label
				unsigned short label = 0, nyuId = 0;
				bool bValid = LabelUtil::get().getIdForLabel(node.modelId, label);
				if (!bValid) throw MLIB_EXCEPTION("no label index for " + node.modelId);
				bValid = LabelUtil::get().getNyuIdForId(label, nyuId);
				if (!bValid) throw MLIB_EXCEPTION("no nyu index for " + label);
				// bb
				m_objectBBs.push_back(BBInfo(OBB3f(node.bbox), meshBbox, label, objectId, meshBbox, 0.0f));
				m_objBBRenderObjectIdxes.push_back(std::vector<unsigned int>());
				auto& objBBRenderObjectIdx = m_objBBRenderObjectIdxes.back();
				m_curAugmentedTransforms.push_back(mat4f::identity());
				m_curAugmentedAngles.push_back(0.0f);

				std::vector< std::pair <MeshDataf, Materialf > > meshDataByMaterial = meshDataAll.splitByMaterial();
				for (auto& m : meshDataByMaterial) {

					MeshDataf& meshData = m.first;
					Materialf& material = m.second;

					MLIB_ASSERT(meshData.isConsistent());
					if (!meshData.isTriMesh()) {
						if (printWarnings) std::cout << "Warning mesh " << meshFilename << " contains non-tri faces (auto-converting)" << std::endl;
						meshData.makeTriMesh();
					}

					MLIB_ASSERT(meshData.isConsistent());
					if (meshData.m_Colors.size() == 0) meshData.m_Colors.resize(meshData.m_Vertices.size(), vec4f(1.0f, 1.0f, 1.0f, 1.0f));	//set default color if none present
					//for (auto& c : meshData.m_Colors) c.w = (float)label;
					if (objectId >= 1000) throw MLIB_EXCEPTION("unable to handle object id " + std::to_string(objectId) + " (max 1000 objects)");
					unsigned int instanceLabel = (unsigned int)nyuId * 1000u + (unsigned int)objectId;

					for (auto& c : meshData.m_Colors) {
						c.w = (float)instanceLabel;
					}
					TriMeshf triMesh(meshData);
					if (!triMesh.hasNormals())	triMesh.computeNormals();
					material.m_ambient = vec4f(0.1f);

					objBBRenderObjectIdx.push_back((unsigned int)m_objects.size());
					std::string path = util::directoryFromPath(meshFilename);
					if (material.m_TextureFilename_Kd != "") material.m_TextureFilename_Kd = path + material.m_TextureFilename_Kd;
					addObject(triMesh, material);
				}
			}
			else if (node.type == "Box") { //TODO HERE ANGIE
				const std::string meshFilename = suncgPath + "/" + modelDir + "/mgcube/mgcube.obj";
				MeshDataf meshDataAll = MeshIOf::loadFromFile(meshFilename, bIgnoreNans);
				meshDataAll.applyTransform(mat4f::scale(node.dimensions));
				meshDataAll.applyTransform(node.transform);
				const bbox3f meshBbox = meshDataAll.computeBoundingBox();
				if (!bbox.intersects(meshBbox)) { if (printWarnings) std::cout << "warning: skipping mesh (" << node.modelId << ") that falls outside of level bbox" << std::endl; continue; } //skip since outside bbox
				//find label
				unsigned short label = 0, nyuId = 0;
				bool bValid = LabelUtil::get().getIdForLabel(node.type, label);
				if (!bValid) throw MLIB_EXCEPTION("no label index for " + node.type);
				bValid = LabelUtil::get().getNyuIdForId(label, nyuId);

				std::vector< std::pair <MeshDataf, Materialf > > meshDataByMaterial = meshDataAll.splitByMaterial();
				for (unsigned int i = 0; i < meshDataByMaterial.size(); i++) {
					MeshDataf& meshData = meshDataByMaterial[i].first;
					Materialf& material = meshDataByMaterial[i].second;
					MLIB_ASSERT(meshData.isConsistent());
					if (!meshData.isTriMesh()) {
						if (printWarnings) std::cout << "Warning mesh " << meshFilename << " contains non-tri faces (auto-converting)" << std::endl;
						meshData.makeTriMesh();
					}
					MLIB_ASSERT(meshData.isConsistent());
					if (meshData.m_Colors.size() == 0) meshData.m_Colors.resize(meshData.m_Vertices.size(), vec4f(1.0f, 1.0f, 1.0f, 0.0f));	//set default color if none present
					unsigned int instanceLabel = (unsigned int)nyuId * 1000u;
					for (auto& c : meshData.m_Colors) c.w = (float)instanceLabel;
					TriMeshf triMesh(meshData);
					if (!triMesh.hasNormals())	triMesh.computeNormals();
					material.m_ambient = vec4f(0.1f);

					if (i < node.materials.size()) {
						const auto it = node.materials[i].data.find("texture");
						if (it != node.materials[i].data.end()) {
							if (util::fileExists(suncgPath + textureDir + "/" + it->second + ".jpg"))
								material.m_TextureFilename_Kd = suncgPath + textureDir + "/" + it->second + ".jpg";
							else if (util::fileExists(suncgPath + textureDir + "/" + it->second + ".png"))
								material.m_TextureFilename_Kd = suncgPath + textureDir + "/" + it->second + ".png";
						}
					}
					addObject(triMesh, material);
				}
			}
			else if (node.type == "Ground") {
				const std::string meshFilename = suncgPath + "/" + roomDir + "/" + m_sceneInfo.id + "/" + node.modelId + "f.obj";
				MeshDataf meshDataAll = MeshIOf::loadFromFile(meshFilename, bIgnoreNans);
				const bbox3f meshBbox = meshDataAll.computeBoundingBox();
				if (!bbox.intersects(meshBbox)) { if (printWarnings) std::cout << "warning: skipping mesh (" << node.modelId << ") that falls outside of level bbox" << std::endl; continue; } //skip since outside bbox
				//find label
				unsigned short label = 0, nyuId = 0;
				bool bValid = LabelUtil::get().getIdForLabel("Floor", label); //they seem to want this mapped to floor
				if (!bValid) throw MLIB_EXCEPTION("no label index for " + node.type);
				bValid = LabelUtil::get().getNyuIdForId(label, nyuId);

				std::vector< std::pair <MeshDataf, Materialf > > meshDataByMaterial = meshDataAll.splitByMaterial();
				for (auto& m : meshDataByMaterial) {

					MeshDataf& meshData = m.first;
					Materialf& material = m.second;

					MLIB_ASSERT(meshData.isConsistent());
					if (!meshData.isTriMesh()) {
						if (printWarnings) std::cout << "Warning mesh " << meshFilename << " contains non-tri faces (auto-converting)" << std::endl;
						meshData.makeTriMesh();
					}
					MLIB_ASSERT(meshData.isConsistent());
					if (meshData.m_Colors.size() == 0) meshData.m_Colors.resize(meshData.m_Vertices.size(), vec4f(1.0f, 1.0f, 1.0f, 0.0f));	//set default color if none present
					unsigned int instanceLabel = (unsigned int)nyuId * 1000u;
					for (auto& c : meshData.m_Colors) c.w = (float)instanceLabel;
					TriMeshf triMesh(meshData);
					if (!triMesh.hasNormals())	triMesh.computeNormals();

					material.m_ambient = vec4f(0.1f);
					std::string path = util::directoryFromPath(meshFilename);
					if (material.m_TextureFilename_Kd != "") material.m_TextureFilename_Kd = path + material.m_TextureFilename_Kd;
					addObject(triMesh, material);
				}
			}
			else {
				throw MLIB_EXCEPTION("unknown type: " + node.type);
			}
		}
	}
}

bool Scene::intersectsCameraBox(const Cameraf& camera, float bboxRadius, float viewDirThresh) const {
	const vec3f eye = camera.getEye();
	const bbox3f bbox = bbox3f(eye - bboxRadius, eye + bboxRadius);
	const OBB3f obbView = OBB3f(eye + camera.getLook()*0.5f*viewDirThresh, camera.getRight()*viewDirThresh, camera.getLook()*viewDirThresh, camera.getUp()*viewDirThresh);

	const TriMeshf bboxMesh = Shapesf::box(bbox, vec4f(1.0f, 0.0f, 0.0f, 1.0f));
	std::vector< std::vector<vec3f> > bboxTriangles;
	for (size_t i = 0; i < bboxMesh.m_indices.size(); i++) {
		vec3f p0 = bboxMesh.m_vertices[bboxMesh.m_indices[i].x].position;
		vec3f p1 = bboxMesh.m_vertices[bboxMesh.m_indices[i].y].position;
		vec3f p2 = bboxMesh.m_vertices[bboxMesh.m_indices[i].z].position;
		bboxTriangles.push_back({ p0, p1, p2 });
	}

	for (unsigned int o = 0; o < m_objects.size(); o++) {
		const auto& triMesh = m_objects[o].getD3D11TriMesh().getTriMesh();
		const bbox3f objBbox = triMesh.computeBoundingBox();
		if (OBB3f(objBbox).intersects(obbView)) {
			Rayf ray(eye, camera.getLook());
			if (objBbox.intersect(ray, 0.0f, viewDirThresh))
				return true; // too close to view
		}
		if (!objBbox.intersects(bbox)) continue;

		for (size_t i = 0; i < triMesh.m_indices.size(); i++) {
			vec3f p0 = triMesh.m_vertices[triMesh.m_indices[i].x].position;
			vec3f p1 = triMesh.m_vertices[triMesh.m_indices[i].y].position;
			vec3f p2 = triMesh.m_vertices[triMesh.m_indices[i].z].position;
			for (size_t j = 0; j < bboxTriangles.size(); j++)
				if (intersection::intersectTriangleTriangle(p0, p1, p2, bboxTriangles[j][0], bboxTriangles[j][1], bboxTriangles[j][2]))
					return true;
		}
	}
	return false;
}

void Scene::getRoomBboxes(std::vector<std::vector<bbox3f>>& bboxes) {

	bboxes.clear();
	for (auto& level : m_sceneGraph) {
		const bbox3f bbox = level.front().bbox; //sometimes this can filter out random floating crap
		bboxes.push_back(std::vector<bbox3f>());
		for (auto& node : level) {
			if (!node.valid) continue; //ignore invalid
			if (node.type == "") {
				continue;	//if it's the level root ignore it
			}
			else if (node.type == "Room") {
				if (node.bbox.isValid()) bboxes.back().push_back(node.bbox);
			}
		}
	}
}

MeshDataf Scene::computeOBBMeshVis(const std::vector<BBInfo>& obbs, const vec4f& color /*= vec4f(1.0f, 0.0f, 0.0f, 1.0f)*/) const {
	MeshDataf mesh;
	if (obbs.empty()) {
		std::cout << "warning: no obbs to visualize!" << std::endl;
		return mesh;
	}
	const float radius = 0.1f;
	for (const BBInfo& obb : obbs) {
		const auto& edges = obb.obb.getEdges();
		for (const auto& e : edges)
			mesh.merge(Shapesf::cylinder(e.p0(), e.p1(), radius, 10, 10, color).computeMeshData());
	}
	return mesh;
}

MeshDataf Scene::computeAABBMeshVis(const std::vector<BBInfo>& aabbs, const vec4f& color /*= vec4f(1.0f, 0.0f, 0.0f, 1.0f)*/, float radius /*= 0.1f*/) const {
	MeshDataf mesh;
	if (aabbs.empty()) {
		std::cout << "warning: no aabbs to visualize!" << std::endl;
		return mesh;
	}
	for (const BBInfo& aabb : aabbs) {
		const auto& edges = aabb.aabb.getEdges();
		for (const auto& e : edges)
			mesh.merge(Shapesf::cylinder(e.p0(), e.p1(), radius, 10, 10, color).computeMeshData());
	}
	return mesh;
}

void Scene::augment(GraphicsDevice& g, const std::string& outCacheFile)
{
	const auto oldAugmentedTransforms = m_curAugmentedTransforms;
	m_curAugmentedTransforms.clear();
	m_curAugmentedAngles.clear();
	m_curAugmentedTransforms.resize(m_objBBRenderObjectIdxes.size(), mat4f::identity());
	m_curAugmentedAngles.resize(m_objBBRenderObjectIdxes.size(), 0.0f);
	if (false && util::fileExists(outCacheFile)) {
		BinaryDataStreamFile ifs(outCacheFile, false);
		ifs >> m_curAugmentedTransforms;
		ifs >> m_curAugmentedAngles;
		std::vector<std::vector<unsigned int>> objBBRnderObjectIdxes;
		ifs >> objBBRnderObjectIdxes;
		ifs.close();
		bool ok = true;
		if (m_objBBRenderObjectIdxes.size() != objBBRnderObjectIdxes.size()) {
			ok = false;
		}
		else {
			for (unsigned int i = 0; i < m_objBBRenderObjectIdxes.size(); i++) {
				const auto& idxes0 = m_objBBRenderObjectIdxes[i];
				const auto& idxes1 = objBBRnderObjectIdxes[i];
				if (idxes0.size() != idxes1.size()) {
					ok = false;
					break;
				}
				else {
					for (unsigned int j = 0; j < idxes0.size(); j++) {
						if (idxes0[j] != idxes1[j]) {
							ok = false;
							break;
						}
					}
				}
				if (!ok) break;
			}
		}
		if (!ok) throw MLIB_EXCEPTION("failed to read in augment data: inconsistent objects in cache");

		for (unsigned int i = 0; i < m_objBBRenderObjectIdxes.size(); i++) {
			const std::vector<unsigned int>& renderObjIdxes = m_objBBRenderObjectIdxes[i];
			auto& obb = m_objectBBs[i];
			mat4f transform = m_curAugmentedTransforms[i];
			const mat4f& oldTransformInv = oldAugmentedTransforms[i].getInverse();
			transform = transform * oldTransformInv;

			bbox3f bbox;
			for (unsigned int renderObjIdx : renderObjIdxes) {
				auto& renderObject = m_objects[renderObjIdx];
				mat4f modelToWorld = renderObject.getModelToWorld();
				modelToWorld = transform * modelToWorld;
				renderObject.setModelToWorld(modelToWorld);
				bbox.include(renderObject.getBoundingBoxWorld());
			}
			obb.obb = transform * obb.obb;
			obb.aabb = bbox;
			obb.angleCanonical = m_curAugmentedAngles[i];
			m_curAugmentedTransforms[i] = transform;
		}
	}
	else {
		for (unsigned int i = 0; i < m_objBBRenderObjectIdxes.size(); i++) {
			const std::vector<unsigned int>& renderObjIdxes = m_objBBRenderObjectIdxes[i];
			auto& obb = m_objectBBs[i];
			// random rotation
			const vec3f center = obb.obb.getCenter();
			float horizAngle = math::randomUniform(-40.0f, 40.0f);
			mat4f transform = mat4f::translation(center) * mat4f::rotationY(horizAngle) * mat4f::translation(-center);
			const mat4f& oldTransformInv = oldAugmentedTransforms[i].getInverse();
			transform = transform * oldTransformInv;

			// check if part of no-rotate set
			unsigned short labelId = m_objects[renderObjIdxes.front()].getLabelId();
			if (LabelUtil::get().isExcludedAugmentClass(labelId))
				continue;

			bbox3f bbox;
			for (unsigned int renderObjIdx : renderObjIdxes) {
				auto& renderObject = m_objects[renderObjIdx];
				mat4f modelToWorld = renderObject.getModelToWorld();
				modelToWorld = transform * modelToWorld;
				renderObject.setModelToWorld(modelToWorld);
				bbox.include(renderObject.getBoundingBoxWorld());
			}
			obb.obb = transform * obb.obb;
			obb.aabb = bbox;
			m_curAugmentedTransforms[i] = transform;
			m_curAugmentedAngles[i] = horizAngle;
			obb.angleCanonical = horizAngle;
		}
		// cache augmentation
		{
			BinaryDataStreamFile ofs(outCacheFile, true);
			ofs << m_curAugmentedTransforms;
			ofs << m_curAugmentedAngles;
			ofs << m_objBBRenderObjectIdxes;
			ofs.close();
		}
	}
}

unsigned short Scene::computeInstance(const vec3ui& location, const VoxelGrid& labels) const {
	const Voxel& v0 = labels(location);
	unsigned short label = (((unsigned short)v0.color[1]) << 8) | v0.color[0];
	if (label > 0) return label;

	const int searchRadius = 3;
	float minDistSq = std::numeric_limits<float>::infinity();
	for (int zz = -searchRadius; zz <= searchRadius; zz++) {
		for (int yy = -searchRadius; yy <= searchRadius; yy++) {
			for (int xx = -searchRadius; xx <= searchRadius; xx++) {
				vec3i coord((int)location.x + xx, (int)location.y + yy, (int)location.z + zz);
				if (labels.isValidCoordinate(coord)) {
					const Voxel& v = labels(coord);
					unsigned short l = (((unsigned short)v.color[1]) << 8) | v.color[0];
					if (l > 0) {
						float dist = vec3f::distSq(coord, location);
						if (dist < minDistSq) {
							label = l;
							minDistSq = dist;
						}
					}
				}
			}
		}
	}
	if (label == 0) throw MLIB_EXCEPTION("no non-empty label for occupied location (" + std::to_string(location.x) + "," + std::to_string(location.y) + "," + std::to_string(location.z) + ")");
	return label;
}

void Scene::voxelizeTriangle(const vec3f& v0, const vec3f& v1, const vec3f& v2, unsigned short id, VoxelGrid& grid, bool solid /*= false*/) const
{
	const size_t depth = grid.getDimZ();
	float diagLenSq = 3.0f;
	if ((v0 - v1).lengthSq() < diagLenSq && (v0 - v2).lengthSq() < diagLenSq && (v1 - v2).lengthSq() < diagLenSq) {
		bbox3f bb(v0, v1, v2);
		vec3ui minI = math::floor(bb.getMin());
		vec3ui maxI = math::ceil(bb.getMax());
		minI = vec3ui(math::clamp(minI.x, 0u, (unsigned int)grid.getDimX()), math::clamp(minI.y, 0u, (unsigned int)grid.getDimY()), math::clamp(minI.z, 0u, (unsigned int)grid.getDimZ()));
		maxI = vec3ui(math::clamp(maxI.x, 0u, (unsigned int)grid.getDimX()), math::clamp(maxI.y, 0u, (unsigned int)grid.getDimY()), math::clamp(maxI.z, 0u, (unsigned int)grid.getDimZ()));

		//test for accurate voxel-triangle intersections
		for (unsigned int i = minI.x; i <= maxI.x; i++) {
			for (unsigned int j = minI.y; j <= maxI.y; j++) {
				for (unsigned int k = minI.z; k <= maxI.z; k++) {
					vec3f v(i, j, k);
					bbox3f voxel;
					const float eps = 0.0000f;
					voxel.include((v - 0.5f - eps));
					voxel.include((v + 0.5f + eps));
					if (voxel.intersects(v0, v1, v2)) {
						if (solid) {
							//project to xy-plane
							vec2f pv = v.getVec2();
							if (intersection::intersectTrianglePoint(v0.getVec2(), v1.getVec2(), v2.getVec2(), pv)) {
								Rayf r0(vec3f(v), vec3f(0, 0, 1));
								Rayf r1(vec3f(v), vec3f(0, 0, -1));
								float t0, t1, _u0, _u1, _v0, _v1;
								bool b0 = intersection::intersectRayTriangle(v0, v1, v2, r0, t0, _u0, _v0);
								bool b1 = intersection::intersectRayTriangle(v0, v1, v2, r1, t1, _u1, _v1);
								if ((b0 && t0 <= 0.5f) || (b1 && t1 <= 0.5f)) {
									if (i < grid.getDimX() && j < grid.getDimY() && k < grid.getDimZ()) {
										//grid.toggleVoxelAndBehindSlice(i, j, k);
										for (size_t kk = k; kk < depth; kk++) {
											Voxel& v = grid(i, j, kk);
											v.color[0] = id & 0xff; //ushort to vec2uc
											v.color[1] = (id >> 8) & 0xff;
										}
									}
								}
								//grid.setVoxel(i,j,k);
							}
						}
						else {
							if (i < grid.getDimX() && j < grid.getDimY() && k < grid.getDimZ()) {
								Voxel& v = grid(i, j, k);
								v.color[0] = id & 0xff; //ushort to vec2uc
								v.color[1] = (id >> 8) & 0xff;
							}
						}
					}
				}
			}
		}
	}
	else {
		vec3f e0 = (float)0.5f*(v0 + v1);
		vec3f e1 = (float)0.5f*(v1 + v2);
		vec3f e2 = (float)0.5f*(v2 + v0);
		voxelizeTriangle(v0, e0, e2, id, grid, solid);
		voxelizeTriangle(e0, v1, e1, id, grid, solid);
		voxelizeTriangle(e1, v2, e2, id, grid, solid);
		voxelizeTriangle(e0, e1, e2, id, grid, solid);
	}
}

void Scene::voxelize(VoxelGrid& grid, const TriMeshf& triMesh, const mat4f& worldToVoxel /*= mat4f::identity()*/, bool solid /*= false*/, bool verbose /*= true*/) const
{
	for (size_t i = 0; i < triMesh.m_indices.size(); i++) {
		vec3f p0 = worldToVoxel * triMesh.m_vertices[triMesh.m_indices[i].x].position;
		vec3f p1 = worldToVoxel * triMesh.m_vertices[triMesh.m_indices[i].y].position;
		vec3f p2 = worldToVoxel * triMesh.m_vertices[triMesh.m_indices[i].z].position;
		if (isnan(p0.x) || isnan(p1.x) || isnan(p2.x))
			continue;
		unsigned short id = (unsigned short)std::round(triMesh.m_vertices[triMesh.m_indices[i].x].color.w);

		bbox3f bb0(p0, p1, p2);
		bbox3f bb1(vec3f(0, 0, 0), vec3f((float)grid.getDimX() + 1.0f, (float)grid.getDimY() + 1.0f, (float)grid.getDimZ() + 1.0f));
		if (bb0.intersects(bb1)) {
			voxelizeTriangle(p0, p1, p2, id, grid, solid);
		}
		else if (verbose) {
			std::cerr << "out of bounds: " << p0 << "\tof: " << grid.getDimensions() << std::endl;
			std::cerr << "out of bounds: " << p1 << "\tof: " << grid.getDimensions() << std::endl;
			std::cerr << "out of bounds: " << p2 << "\tof: " << grid.getDimensions() << std::endl;
			MLIB_WARNING("triangle outside of grid - ignored");
		}
	}
}