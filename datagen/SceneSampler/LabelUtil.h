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
	
	bool getIdForLabel(const std::string& label, unsigned short& id) const {
		const auto it = m_labelsToIds.find(label);
		if (it == m_labelsToIds.end()) return false;
		id = it->second;
		return true;
	}

	bool getLabelForId(unsigned short id, std::string& label) const {
		const auto it = m_idsToLabels.find(id);
		if (it == m_idsToLabels.end()) return false;
		label = it->second;
		return true;
	}

	bool isInitialized() const { return m_bIsInitialized; }
	
private:

	void getLabelMappingFromFile(const std::string& filename, const std::string& labelName, const std::string& idName)
	{
		if (!util::fileExists(filename)) throw MLIB_EXCEPTION("label mapping files does not exist!");
		const char splitter = ',';

		m_labelsToIds.clear();
		m_idsToLabels.clear();

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
				m_idsToLabels[(unsigned short)id] = parts[labelIdx];
				m_labelsToIds[parts[labelIdx]] = (unsigned short)id;
			}
		}
		s.close();
	}

	bool			m_bIsInitialized;
	unsigned int	m_maxLabel;

	std::unordered_map<std::string, unsigned short> m_labelsToIds;
	std::unordered_map<unsigned short, std::string> m_idsToLabels;
};