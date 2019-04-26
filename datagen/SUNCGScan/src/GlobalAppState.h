#pragma once

#include "stdafx.h"

#include <vector>
#include <string>
#include <list>



#define X_GLOBAL_APP_STATE_FIELDS \
	X(std::vector<std::string>, s_meshFilenames) \
	X(unsigned int, s_renderWidth) \
	X(unsigned int, s_renderHeight) \
	X(unsigned int, s_BRDF) \
	X(float, s_cameraFov) \
	X(bool, s_addNoiseToDepth) \
	X(float, s_depthNoiseSigma) \
	X(bool, s_filterDepthMap) \
	X(float, s_depthSigmaD) \
	X(float, s_depthSigmaR) \
	X(float, s_voxelSize) \
	X(float, s_renderNear) \
	X(float, s_renderFar) \
	X(float, s_viewEvalMaxDepth) \
	X(std::string, s_suncgPath) \
	X(std::string, s_suncgLabelMapFile) \
	X(std::string, s_labelMapLabelName) \
	X(std::string, s_labelMapIdName) \
	X(std::string, s_viewStatsFile) \
	X(vec3f, s_gridExtents) \
	X(std::string, s_outputPath) \
	X(std::string, s_outputOBBPath) \
	X(std::string, s_outputAABBPath) \
	X(std::string, s_sceneFileList) \
	X(unsigned int, s_maxNumViewsToSample) \
	X(unsigned int, s_maxNumViews) \
	X(unsigned int, s_maxNumTrajectories) \
	X(unsigned int, s_maxNumViewPerTraj) \
	X(std::string, s_cameraPath) \
	X(bool, s_bGenerateOBBs) \
	X(bool, s_bGenerateAABBs) \
	X(unsigned int, s_numAugmentPartial) \
	X(unsigned int, s_scenePadding) \
	X(float, s_validObbOccThresh) \
	X(unsigned int, s_validObbMinNumOcc) \
	X(std::string, s_outputImagePath) \
	X(bool, s_bGenerateImages) \
	X(bool, s_bDebugOut) \
	X(bool, s_bGenerateCameras)


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
