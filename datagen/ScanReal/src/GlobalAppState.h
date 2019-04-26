#pragma once

#include "stdafx.h"

#include <vector>
#include <string>
#include <list>



#define X_GLOBAL_APP_STATE_FIELDS \
	X(std::string, s_sceneFileList) \
	X(std::string, s_scanPath) \
	X(std::string, s_scanMeshPath) \
	X(std::string, s_scanLabelFile) \
	X(std::string, s_labelIdName) \
	X(std::string, s_labelName) \
	X(std::string, s_outputPath) \
	X(std::string, s_outputAABBPath) \
	X(unsigned int, s_renderWidth) \
	X(unsigned int, s_renderHeight) \
	X(unsigned int, s_BRDF) \
	X(float, s_cameraFov) \
	X(float, s_minDepth) \
	X(float, s_maxDepth) \
	X(bool, s_addNoiseToDepth) \
	X(float, s_depthNoiseSigma) \
	X(bool, s_filterDepthMap) \
	X(float, s_depthSigmaD) \
	X(float, s_depthSigmaR) \
	X(float, s_voxelSize) \
	X(float, s_renderNear) \
	X(float, s_renderFar) \
	X(unsigned int, s_scenePadding) \
	X(unsigned int, s_numRotAug) \
	X(unsigned int, s_maxNumScenes) \
	X(unsigned int, s_maxNumSens) \
	X(std::string, s_alnPath) \
	X(bool, s_bGenerateTestOnly) \
	X(std::string, s_outputImagePath) \
	X(bool, s_bDebugOut)


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
