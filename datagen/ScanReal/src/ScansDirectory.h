
#pragma once

#include "mLibInclude.h"


struct ScanInfo {
	std::string sceneName;
	std::string meshFile;
	std::string aggregationFile;
	std::string segmentationFile;
	std::vector<std::string> sensFiles;
	std::string alnFile;

	ScanInfo(const std::string& _sceneName, const std::string& _meshFile, const std::string& _aggFile, const std::string& _segFile, const std::vector<std::string>& _sensFiles, const std::string& _alnFile = "") {
		sceneName = _sceneName;
		meshFile = _meshFile;
		aggregationFile = _aggFile;
		segmentationFile = _segFile;
		sensFiles = _sensFiles;
		alnFile = _alnFile;
	}
};

class ScansDirectory {
public:
	ScansDirectory() {}
	~ScansDirectory() {}

	void loadScanNet(std::string scanNetPath, std::string scanMeshPath, const std::string& alnPath, const std::string& sceneListFile, bool bGenerateTestOnly) {
		if (scanNetPath.back() != '/' && scanNetPath.back() != '\\') scanNetPath.push_back('/');
		if (scanMeshPath.back() != '/' && scanMeshPath.back() != '\\') scanMeshPath.push_back('/');

		const std::string meshExt = "_vh_clean_2.ply";
		const std::string segExt = "_vh_clean_2.0.010000.segs.json";
		const std::string aggExt = ".aggregation.json";
		const std::string sensExt = ".sens";
		const std::string alignmentExt = "t.aln";

		clear();
		std::ifstream s(sceneListFile);
		if (!s.is_open()) throw MLIB_EXCEPTION("failed to open " + sceneListFile);
		std::cout << "loading scan info from list..." << std::endl;
		std::string scene;
		while (std::getline(s, scene)) {
			std::cout << "\rloading " << scene;
			const std::string meshFile = scanMeshPath + scene + "/" + scene + meshExt;
			const std::string aggFile = scanMeshPath + scene + "/" + scene + aggExt;
			const std::string segFile = scanMeshPath + scene + "/" + scene + segExt;
			const std::string alnFile = util::fileExists(alnPath + "/" + scene + ".aln") ? alnPath + "/" + scene + ".aln" : "";
			const std::vector<std::string> sensFiles = { scanNetPath + scene + "/" + scene + sensExt };
			if (bGenerateTestOnly) {
				if (util::fileExists(sensFiles.front()) && util::fileExists(meshFile))
					m_scans.push_back(ScanInfo(scene, meshFile, "", "", sensFiles, ""));
			}
			else {
				if (util::fileExists(sensFiles.front()) && util::fileExists(meshFile)) // && util::fileExists(alnFile))
					m_scans.push_back(ScanInfo(scene, meshFile, aggFile, segFile, sensFiles, alnFile));
			}
		}
		std::cout << std::endl << "[" << util::fileNameFromPath(sceneListFile) << "] | found " << m_scans.size() << " scenes" << std::endl;
	}
	
	void loadMatterport(std::string scanPath, std::string scanMeshPath, const std::string& sceneListFile, unsigned int maxNumSens) {
		if (scanPath.back() != '/' && scanPath.back() != '\\') scanPath.push_back('/');
		if (scanMeshPath.back() != '/' && scanMeshPath.back() != '\\') scanMeshPath.push_back('/');

		const std::string meshExt = ".reduced.ply";
		const std::string segExt = ".vsegs.json";
		const std::string aggExt = ".semseg.json";
		const std::string sensExt = ".sens";

		const std::string dataSubPath = "region_segmentations";

		clear();
		std::ifstream s(sceneListFile);
		if (!s.is_open()) throw MLIB_EXCEPTION("failed to open " + sceneListFile);
		std::cout << "loading scan info from list..." << std::endl;
		std::string room;
		while (std::getline(s, room)) {
			const auto parts = util::split(room, "_room");
			const std::string& scene = parts[0];
			const std::string& roomId = parts[1];
			const std::string meshFile = scanMeshPath + scene + "/" + dataSubPath + "/region" + roomId + meshExt;
			const std::string aggFile = scanMeshPath + scene + "/" + dataSubPath + "/region" + roomId + aggExt;
			const std::string segFile = scanMeshPath + scene + "/" + dataSubPath + "/region" + roomId + segExt;
			
			const std::string sensPath = scanPath + scene + "/sens";
			std::vector<std::string> sensFiles = Directory(sensPath).getFilesWithSuffix(sensExt); //all must exist
			if (sensFiles.empty())
				throw MLIB_EXCEPTION("no sens files found for scene " + scene);
			if (sensFiles.size() > maxNumSens) {
				MLIB_WARNING("found " + std::to_string(sensFiles.size()) + " sens, max " + std::to_string(maxNumSens) + ", truncating");
				sensFiles.resize(maxNumSens);
			}
			for (auto& sensFile : sensFiles)
				sensFile = sensPath + "/" + sensFile;
			if (util::fileExists(meshFile))
				m_scans.push_back(ScanInfo(room, meshFile, aggFile, segFile, sensFiles));
		}
		std::cout << "[" << util::fileNameFromPath(sceneListFile) << "] | found " << m_scans.size() << " scenes" << std::endl;
	}


	const std::vector<ScanInfo>& getScans() const {
		return m_scans;
	}

	void clear() { m_scans.clear(); }
private:

	std::vector<ScanInfo> m_scans;
};