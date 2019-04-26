#pragma once

class LabelUtil
{
public:
	LabelUtil() {}
	void init(const std::string& labelMapFile, const std::string& labelName, const std::string& idName) {
		if (!util::fileExists(labelMapFile)) throw MLIB_EXCEPTION(labelMapFile + " does not exist!");
		m_maxLabel = 65535;//255;

		getLabelMappingFromFile(labelMapFile, labelName, idName);

		m_bIsInitialized = true;
	}
	void initNyu(const std::string& labelMapFile, const std::string& labelName, const std::string& idName) {
		if (!util::fileExists(labelMapFile)) throw MLIB_EXCEPTION(labelMapFile + " does not exist!");
		getNYUMappingFromFile(labelMapFile, labelName, idName);
	}

	static LabelUtil& getInstance() {
		static LabelUtil s;
		return s;
	}
	static LabelUtil& get() {
		return getInstance();
	}

	bool getIdForLabel(const std::string& label, unsigned short& id) const {
		const auto it = s_labelsToIds.find(label);
		if (it == s_labelsToIds.end()) return false;
		id = it->second;
		return true;
	}

	bool getLabelForId(unsigned short id, std::string& label) const {
		const auto it = s_idsToLabels.find(id);
		if (it == s_idsToLabels.end()) return false;
		label = it->second;
		return true;
	}

	bool getNyuIdForId(unsigned short id, unsigned short& nyuId) const {
		const auto it = s_suncgIdToNyuLabel.find(id);
		if (it == s_suncgIdToNyuLabel.end()) return false;
		const auto it2 = s_nyuLabelToNyuId.find(it->second);
		if (it2 == s_nyuLabelToNyuId.end()) return false;
		nyuId = it2->second;
		return true;
	}

	bool isExcludedClass(unsigned short id) const {
		return (s_excludedObjects.find(id) != s_excludedObjects.end());
	}

	bool isExcludedAugmentClass(unsigned short id) const {
		//return (s_excludedAugmentObjects.find(id) != s_excludedAugmentObjects.end());
		return (s_excludedAugmentObjectsNYU.find(id) != s_excludedAugmentObjectsNYU.end());
	}

private:

	void getLabelMappingFromFile(const std::string& filename, const std::string& labelName, const std::string& idName)
	{
		if (!util::fileExists(filename)) throw MLIB_EXCEPTION("label mapping files does not exist!");
		const char splitter = ',';

		s_labelsToIds.clear();
		s_idsToLabels.clear();
		s_suncgIdToNyuLabel.clear();
		const std::string catName = "coarse_grained_class";
		const std::string nyuName = "nyuv2_40class";
		const std::unordered_set<std::string> excludedClasses = { "empty", "wall", "ceiling", "floor", "box", "plant" };
		const std::unordered_set<std::string> excludedAugmentClasses = { 
			"door", "window", "shower", "bathtub", "curtain",
			"mirror", "stairs", "heater", "air_conditioner", 
			"fireplace", "picture_frame", "garage_door", "fence" 
		};

		std::ifstream s(filename); std::string line;
		//read header
		std::unordered_map<std::string, unsigned int> header;
		if (!std::getline(s, line)) throw MLIB_EXCEPTION("error reading label mapping file");
		auto parts = util::split(line, splitter);
		const unsigned int numElems = (unsigned int)parts.size();
		for (unsigned int i = 0; i < parts.size(); i++) header[parts[i]] = i;

		auto it = header.find(labelName);
		if (it == header.end()) throw MLIB_EXCEPTION("could not find value " + labelName + " in label mapping file");
		unsigned int labelIdx = it->second;
		it = header.find(idName);
		if (it == header.end()) throw MLIB_EXCEPTION("could not find value " + idName + " in label mapping file");
		unsigned int idIdx = it->second;
		it = header.find(catName);
		if (it == header.end()) throw MLIB_EXCEPTION("could not find value " + catName + " in label mapping file");
		unsigned int catIdx = it->second;
		it = header.find(nyuName);
		if (it == header.end()) throw MLIB_EXCEPTION("could not find value " + nyuName + " in label mapping file");
		unsigned int nyuIdx = it->second;

		//read elements
		while (std::getline(s, line)) {
			parts = util::split(line, splitter, true);
			if (!parts[labelIdx].empty() && !parts[idIdx].empty()) {
				unsigned int id = util::convertTo<unsigned int>(parts[idIdx]);
				if (id > m_maxLabel) //skip
					continue;
				s_idsToLabels[(unsigned short)id] = parts[labelIdx];
				s_labelsToIds[parts[labelIdx]] = (unsigned short)id;

				if (excludedClasses.find(parts[catIdx]) != excludedClasses.end())
					s_excludedObjects.insert((unsigned short)id);
				if (excludedAugmentClasses.find(parts[catIdx]) != excludedAugmentClasses.end())
					s_excludedAugmentObjects.insert((unsigned short)id);

				if (!parts[nyuIdx].empty()) {
					s_suncgIdToNyuLabel[id] = parts[nyuIdx];
				}
			}
		}
		s.close();

		std::cout << "read " << s_labelsToIds.size() << " labels, " << s_excludedObjects.size() << " excluded objects, " << s_excludedAugmentObjects.size() << " excluded augment objects" << std::endl;
	}
	void getNYUMappingFromFile(const std::string& filename, const std::string& labelName, const std::string& idName)
	{
		if (!util::fileExists(filename)) throw MLIB_EXCEPTION("label mapping files does not exist!");
		const char splitter = ',';

		s_nyuLabelToNyuId.clear();


		const std::unordered_set<std::string> excludedAugmentClasses = {
			"door", "window", "bathtub", "curtain",
			"mirror", "floor", "picture", "fence", "counter",
			"wall", "ceiling"
		};

		std::ifstream s(filename); std::string line;
		//read header
		std::unordered_map<std::string, unsigned int> header;
		if (!std::getline(s, line)) throw MLIB_EXCEPTION("error reading label mapping file");
		auto parts = util::split(line, splitter);
		const unsigned int numElems = (unsigned int)parts.size();
		for (unsigned int i = 0; i < parts.size(); i++) header[parts[i]] = i;

		auto it = header.find(labelName);
		if (it == header.end()) throw MLIB_EXCEPTION("could not find value " + labelName + " in label mapping file");
		unsigned int labelIdx = it->second;
		it = header.find(idName);
		if (it == header.end()) throw MLIB_EXCEPTION("could not find value " + idName + " in label mapping file");
		unsigned int idIdx = it->second;

		//read elements
		while (std::getline(s, line)) {
			parts = util::split(line, splitter, true);
			if (!parts[labelIdx].empty() && !parts[idIdx].empty()) {
				unsigned int id = util::convertTo<unsigned int>(parts[idIdx]);
				if (id > m_maxLabel) //skip
					continue;
				s_nyuLabelToNyuId[parts[labelIdx]] = (unsigned short)id;
				if (excludedAugmentClasses.find(parts[labelIdx]) != excludedAugmentClasses.end())
					s_excludedAugmentObjectsNYU.insert((unsigned short)id);
			}
		}
		s.close();

		std::cout << "(nyu) read " << s_nyuLabelToNyuId.size() << " labels" << std::endl;
	}

	bool			m_bIsInitialized;
	unsigned int	m_maxLabel;

	std::unordered_map<std::string, unsigned short> s_labelsToIds;
	std::unordered_map<unsigned short, std::string> s_idsToLabels;

	std::unordered_set<unsigned short> s_excludedObjects; //objects excluded from object-ness metric //e.g. empty/wall/ceil/floor/box/plant 

	std::unordered_set<unsigned short> s_excludedAugmentObjects; // objects excluded from rotation augmentatoin (door, window)
	std::unordered_set<unsigned short> s_excludedAugmentObjectsNYU; // objects excluded from rotation augmentatoin (door, window)

	std::unordered_map<unsigned short, std::string> s_suncgIdToNyuLabel;
	std::unordered_map<std::string, unsigned short> s_nyuLabelToNyuId;
};