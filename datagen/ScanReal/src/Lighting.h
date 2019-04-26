
#pragma once

#include "mLibInclude.h"

#define MAX_NUM_LIGHTS 10

class Lighting {
public:
	Lighting() {
		m_lights.numLights = 0;
	}

	Lighting(GraphicsDevice& g, const vec4f& light) {
		m_cbLight.init(g);
		m_lights.numLights = 1;
		m_lights.lights[0] = light;
	}

	Lighting(GraphicsDevice& g, std::vector<vec4f>& lights) {
		m_cbLight.init(g);
		if (lights.size() > MAX_NUM_LIGHTS) throw MLIB_EXCEPTION("exceeds the max num light count");
		m_lights.numLights = (unsigned int)lights.size();
		for (unsigned int i = 0; i < m_lights.numLights; i++) {
			m_lights.lights[i] = lights[i];
		}
	}

	~Lighting() {

	}

	void loadFromGlobaAppState(GraphicsDevice& g, const GlobalAppState& gas) {
		m_cbLight.init(g);
		setNumLights(1);
		randomize();
	}

	void updateAndBind(UINT constantBufferIndex) {
		m_cbLight.updateAndBind(m_lights, constantBufferIndex);
	}

	void setNumLights(unsigned int numLights) {
		m_lights.numLights = numLights;
		if (m_lights.numLights > MAX_NUM_LIGHTS) {
			m_lights.numLights = MAX_NUM_LIGHTS;
			MLIB_WARNING("exceeded max num lights");
		}
	}

	void randomize() {
		//TODO differentiate between directional and point lights (last component 0.0f vs 1.0f)
		for (unsigned int i = 0; i < m_lights.numLights; i++) {
			m_lights.lights[i] = vec4f(vec3f(rndUD(), rndUD(), rndUD()).getNormalized(), 0.0f);	
			//std::cout << i << " : " << m_lights.lightDirs[i] << std::endl;
		}
	}

	void saveToFile(const std::string& filename) const {
		std::ofstream out(filename);
		saveToFile(out);
		out.close();
	}

	void saveToFile(std::ofstream& out) const {
		out << m_lights.numLights << "\n";
		for (unsigned int i = 0; i < m_lights.numLights; i++) {
			out << m_lights.lights[i] << "\n";
		}
	}

	void loadFromFile(const std::string& filename) {
		std::ifstream in(filename);
		if (!in.is_open()) throw MLIB_EXCEPTION(__FUNCTION__);
		loadFromFile(in);
		in.close();
	}

	void loadFromFile(std::ifstream& in) {
		in >> m_lights.numLights;
		for (unsigned int i = 0; i < m_lights.numLights; i++) {
			in >> m_lights.lights[i];
		}
	}

private:

	// random normal distribution and clamped
	static float rndNC(float mu, float sd, float minV = 0.0f, float maxV = 1.0f) {
		static std::default_random_engine generator;
		std::normal_distribution<float> distribution(mu, sd);

		float res = -1.0f;
		while (res < minV || res > maxV) {
			res = distribution(generator);
		}
		return res;
	}

	// random uniform distribution and clamped
	static float rndUD(float minV = -0.5f, float maxV = 0.5f) {
		static std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(minV, maxV);
		return distribution(generator);
	}


	struct ConstantBufferLight {
		vec4f lights[MAX_NUM_LIGHTS];
		unsigned int numLights;
		vec3f dummy;
	};

	ConstantBufferLight m_lights;
	D3D11ConstantBuffer<ConstantBufferLight>	m_cbLight;

};