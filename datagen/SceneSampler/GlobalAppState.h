#pragma once

#include "stdafx.h"

#include <vector>
#include <string>
#include <list>

//#define SUNCG
#define IMAGE_FIND_BBOX_INTERSECT_THRESH 0.8

#define X_GLOBAL_APP_STATE_FIELDS \
	X(std::string, s_sceneFileList) \
	X(std::string, s_outputPath) \
	X(std::string, s_suncgLabelMapFile) \
	X(std::string, s_suncgLabelMapFrom) \
	X(std::string, s_suncgLabelMapTo) \
	X(std::string, s_nyuLabelMapFile) \
	X(std::string, s_nyuLabelMapFrom) \
	X(std::string, s_nyuLabelMapTo) \
	X(std::string, s_scenePath) \
	X(std::string, s_framePath) \
	X(std::string, s_AABBPath) \
	X(unsigned int, s_maxNumTrajectories) \
	X(unsigned int, s_numAugment) \
	X(unsigned int, s_sampleFactor) \
	X(unsigned int, s_maxNumNearestImages) \
	X(float, s_voxelSize) \
	X(bool, s_generateTrain) \
	X(vec3i, s_chunkDim) \
	X(bool, s_bRotate90) \
	X(bool, s_bGenerateTestOnly)


#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif

#define checkSizeArray(a, d)( (((sizeof a)/(sizeof a[0])) >= d))

class GlobalAppState
{
public:

#define X(type, name) type name;
	X_GLOBAL_APP_STATE_FIELDS
#undef X

		//! sets the parameter file and reads
		void readMembers(const ParameterFile& parameterFile) {
		m_ParameterFile = parameterFile;
		readMembers();
	}

	//! reads all the members from the given parameter file (could be called for reloading)
	void readMembers() {
#define X(type, name) \
	if (!m_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
		X_GLOBAL_APP_STATE_FIELDS
#undef X


			m_bIsInitialized = true;
	}

	void print() const {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
		X_GLOBAL_APP_STATE_FIELDS
#undef X
	}

	static GlobalAppState& getInstance() {
		static GlobalAppState s;
		return s;
	}
	static GlobalAppState& get() {
		return getInstance();
	}


	//! constructor
	GlobalAppState() {
		m_bIsInitialized = false;
	}

	//! destructor
	~GlobalAppState() {
	}

	Timer	s_Timer;

private:
	bool			m_bIsInitialized;
	ParameterFile	m_ParameterFile;
};