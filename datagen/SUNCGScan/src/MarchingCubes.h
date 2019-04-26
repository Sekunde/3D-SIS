
#pragma  once

#include "mLibInclude.h"

#include "VoxelGrid.h"
#include "Tables.h"
#include "omp.h"

class MarchingCubes {

public:

	static MeshDataf doMC(const VoxelGrid& grid, const mat4f* transform = NULL) {

		float thresh = 10.0f * grid.getVoxelSize();

		int maxThreads = omp_get_max_threads();	//should be on a quad-core with hyper-threading

		std::vector<std::vector<Triangle>> results(maxThreads);
#pragma omp parallel for
		for (int z = 0; z < grid.getDimZ(); z++) {
			for (int y = 0; y < grid.getDimY(); y++) {
				for (int x = 0; x < grid.getDimX(); x++) {
					vec3f worldPos = grid.voxelToWorld(vec3i(x, y, z));

					int thread = omp_get_thread_num();
					extractIsoSurfaceAtPosition(worldPos, grid, thresh, results[thread]);
				}
			}
		}

		MeshDataf meshData;
		
		size_t numTris = 0;
		for (size_t i = 0; i < results.size(); i++) {
			numTris += results[i].size();
		}

		meshData.m_Vertices.resize(numTris * 3);
		size_t currTri = 0;
		for (size_t i = 0; i < results.size(); i++) {
			for (size_t j = 0; j < results[i].size(); j++) {
				meshData.m_Vertices[3 * currTri + 0] = results[i][j].v0.p;
				meshData.m_Vertices[3 * currTri + 1] = results[i][j].v1.p;
				meshData.m_Vertices[3 * currTri + 2] = results[i][j].v2.p;
				currTri++;
			}
		}

		//create index buffer (required for merging the triangle soup)
		meshData.m_FaceIndicesVertices.resize(meshData.m_Vertices.size());
		for (unsigned int i = 0; i < (unsigned int)meshData.m_Vertices.size() / 3; i++) {
			meshData.m_FaceIndicesVertices[i][0] = 3 * i + 0;
			meshData.m_FaceIndicesVertices[i][1] = 3 * i + 1;
			meshData.m_FaceIndicesVertices[i][2] = 3 * i + 2;

			//m_meshData.m_FaceIndicesVertices[i][0] = 3*i+2;
			//m_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
			//m_meshData.m_FaceIndicesVertices[i][2] = 3*i+0;
		}
		std::cout << "size before:\t" << meshData.m_Vertices.size() << std::endl;

		//m_meshData.removeDuplicateVertices();
		//m_meshData.mergeCloseVertices(0.00001f);
		std::cout << "merging close vertices... ";
		meshData.mergeCloseVertices(0.00001f, true);
		std::cout << "done!" << std::endl;
		std::cout << "removing duplicate faces... ";
		meshData.removeDuplicateFaces();
		std::cout << "done!" << std::endl;

		std::cout << "size after:\t" << meshData.m_Vertices.size() << std::endl;

		if (transform) {
			meshData.applyTransform(*transform);
		}

		return meshData;

	}



private:

	struct Vertex
	{
		vec3f p;
		vec3f c;
	};

	struct Triangle
	{
		Vertex v0;
		Vertex v1;
		Vertex v2;
	};


	static void extractIsoSurfaceAtPosition(const vec3f& worldPos, const VoxelGrid& grid, float thresh, std::vector<Triangle>& result) {

		float voxelSize = grid.getVoxelSize();

		//if (params.m_boxEnabled == 1) {
		//	if (!isInBoxAA(params.m_minCorner, params.m_maxCorner, worldPos)) return;
		//}

		const float isolevel = 0.0f;

		const float P = voxelSize / 2.0f;
		const float M = -P;

		vec3f p000 = worldPos + vec3f(M, M, M); float dist000; vec3uc color000; bool valid000 = grid.trilinearInterpolationSimpleFastFast(p000, dist000, color000);
		vec3f p100 = worldPos + vec3f(P, M, M); float dist100; vec3uc color100; bool valid100 = grid.trilinearInterpolationSimpleFastFast(p100, dist100, color100);
		vec3f p010 = worldPos + vec3f(M, P, M); float dist010; vec3uc color010; bool valid010 = grid.trilinearInterpolationSimpleFastFast(p010, dist010, color010);
		vec3f p001 = worldPos + vec3f(M, M, P); float dist001; vec3uc color001; bool valid001 = grid.trilinearInterpolationSimpleFastFast(p001, dist001, color001);
		vec3f p110 = worldPos + vec3f(P, P, M); float dist110; vec3uc color110; bool valid110 = grid.trilinearInterpolationSimpleFastFast(p110, dist110, color110);
		vec3f p011 = worldPos + vec3f(M, P, P); float dist011; vec3uc color011; bool valid011 = grid.trilinearInterpolationSimpleFastFast(p011, dist011, color011);
		vec3f p101 = worldPos + vec3f(P, M, P); float dist101; vec3uc color101; bool valid101 = grid.trilinearInterpolationSimpleFastFast(p101, dist101, color101);
		vec3f p111 = worldPos + vec3f(P, P, P); float dist111; vec3uc color111; bool valid111 = grid.trilinearInterpolationSimpleFastFast(p111, dist111, color111);

		if (!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;

		uint cubeindex = 0;
		if (dist010 < isolevel) cubeindex += 1;
		if (dist110 < isolevel) cubeindex += 2;
		if (dist100 < isolevel) cubeindex += 4;
		if (dist000 < isolevel) cubeindex += 8;
		if (dist011 < isolevel) cubeindex += 16;
		if (dist111 < isolevel) cubeindex += 32;
		if (dist101 < isolevel) cubeindex += 64;
		if (dist001 < isolevel) cubeindex += 128;

		const float thres = thresh;
		float distArray[] = { dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111 };
		for (uint k = 0; k < 8; k++) {
			for (uint l = 0; l < 8; l++) {
				if (distArray[k] * distArray[l] < 0.0f) {
					if (abs(distArray[k]) + abs(distArray[l]) > thres) return;
				}
				else {
					if (abs(distArray[k] - distArray[l]) > thres) return;
				}
			}
		}

		if (abs(dist000) > thresh) return;
		if (abs(dist100) > thresh) return;
		if (abs(dist010) > thresh) return;
		if (abs(dist001) > thresh) return;
		if (abs(dist110) > thresh) return;
		if (abs(dist011) > thresh) return;
		if (abs(dist101) > thresh) return;
		if (abs(dist111) > thresh) return;

		if (edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return; // added by me edgeTable[cubeindex] == 255

		Voxel v = grid.getVoxel(worldPos);

		Vertex vertlist[12];
		if (edgeTable[cubeindex] & 1)	vertlist[0] = vertexInterp(isolevel, p010, p110, dist010, dist110, v.color, v.color);
		if (edgeTable[cubeindex] & 2)	vertlist[1] = vertexInterp(isolevel, p110, p100, dist110, dist100, v.color, v.color);
		if (edgeTable[cubeindex] & 4)	vertlist[2] = vertexInterp(isolevel, p100, p000, dist100, dist000, v.color, v.color);
		if (edgeTable[cubeindex] & 8)	vertlist[3] = vertexInterp(isolevel, p000, p010, dist000, dist010, v.color, v.color);
		if (edgeTable[cubeindex] & 16)	vertlist[4] = vertexInterp(isolevel, p011, p111, dist011, dist111, v.color, v.color);
		if (edgeTable[cubeindex] & 32)	vertlist[5] = vertexInterp(isolevel, p111, p101, dist111, dist101, v.color, v.color);
		if (edgeTable[cubeindex] & 64)	vertlist[6] = vertexInterp(isolevel, p101, p001, dist101, dist001, v.color, v.color);
		if (edgeTable[cubeindex] & 128)	vertlist[7] = vertexInterp(isolevel, p001, p011, dist001, dist011, v.color, v.color);
		if (edgeTable[cubeindex] & 256)	vertlist[8] = vertexInterp(isolevel, p010, p011, dist010, dist011, v.color, v.color);
		if (edgeTable[cubeindex] & 512)	vertlist[9] = vertexInterp(isolevel, p110, p111, dist110, dist111, v.color, v.color);
		if (edgeTable[cubeindex] & 1024) vertlist[10] = vertexInterp(isolevel, p100, p101, dist100, dist101, v.color, v.color);
		if (edgeTable[cubeindex] & 2048) vertlist[11] = vertexInterp(isolevel, p000, p001, dist000, dist001, v.color, v.color);


		for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
		{
			Triangle t;
			t.v0 = vertlist[triTable[cubeindex][i + 0]];
			t.v1 = vertlist[triTable[cubeindex][i + 1]];
			t.v2 = vertlist[triTable[cubeindex][i + 2]];

			result.push_back(t);
		}
	}

	static Vertex vertexInterp(float isolevel, const vec3f& p1, const vec3f& p2, float d1, float d2, const vec3uc& c1, const vec3uc& c2)
	{
		Vertex r1; r1.p = p1; r1.c = vec3f(c1.x, c1.y, c1.z) / 255.f;
		Vertex r2; r2.p = p2; r2.c = vec3f(c2.x, c2.y, c2.z) / 255.f;

		if (abs(isolevel - d1) < 0.00001f)		return r1;
		if (abs(isolevel - d2) < 0.00001f)		return r2;
		if (abs(d1 - d2) < 0.00001f)			return r1;

		float mu = (isolevel - d1) / (d2 - d1);

		Vertex res;
		res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
		res.p.y = p1.y + mu * (p2.y - p1.y);
		res.p.z = p1.z + mu * (p2.z - p1.z);

		res.c.x = (float)(c1.x + mu * (float)(c2.x - c1.x)) / 255.f; // Color
		res.c.y = (float)(c1.y + mu * (float)(c2.y - c1.y)) / 255.f;
		res.c.z = (float)(c1.z + mu * (float)(c2.z - c1.z)) / 255.f;

		return res;
	}

};

