#pragma once

namespace CameraUtil {

	inline float gaussR(float sigma, float dist)
	{
		return std::exp(-(dist*dist) / (2.0f*sigma*sigma));
	}

	inline float linearR(float sigma, float dist)
	{
		return std::max(1.0f, std::min(0.0f, 1.0f - (dist*dist) / (2.0f*sigma*sigma)));
	}

	inline float gaussD(float sigma, int x, int y)
	{
		return std::exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
	}

	inline float gaussD(float sigma, int x)
	{
		return std::exp(-((x*x) / (2.0f*sigma*sigma)));
	}

	DepthImage32 bilateralFilter(const DepthImage32& input, float sigmaD, float sigmaR)
	{
		const int kernelRadius = (int)std::ceil(2.0*sigmaD);
		DepthImage32 output(input.getWidth(), input.getHeight());
		for (unsigned int y = 0; y < input.getHeight(); y++) {
			for (unsigned int x = 0; x < input.getWidth(); x++) {

				float sum = 0.0f;
				float sumWeight = 0.0f;
				const float depthCenter = input(x, y);
				if (depthCenter != -std::numeric_limits<float>::infinity())
				{
					for (int m = (int)x - kernelRadius; m <= (int)x + kernelRadius; m++)
					{
						for (int n = (int)y - kernelRadius; n <= (int)y + kernelRadius; n++)
						{
							if (m >= 0 && n >= 0 && m < (int)input.getWidth() && n < (int)input.getHeight())
							{
								const float currentDepth = input(m, n);

								if (currentDepth != -std::numeric_limits<float>::infinity()) {
									const float weight = gaussD(sigmaD, m - x, n - y)*gaussR(sigmaR, currentDepth - depthCenter);

									sumWeight += weight;
									sum += weight*currentDepth;
								}
							}
						}
					}

					if (sumWeight > 0.0f) output(x, y) = sum / sumWeight;
					else output(x, y) = -std::numeric_limits<float>::infinity();
				}
				else output(x, y) = -std::numeric_limits<float>::infinity();
			} //x
		} //y
		return output;
	}

}  // namespace CameraUtil