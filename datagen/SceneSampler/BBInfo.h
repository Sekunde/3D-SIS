#pragma once

#include "GlobalAppState.h"

#ifdef SUNCG
typedef Grid3<unsigned short> MaskType;
struct BBInfo {
	OBB3f obb;
	bbox3f aabb;
	unsigned short labelId;
	unsigned short instanceId;

	MaskType mask;

	bbox3f aabbCanonical;
	float angleCanonical;
	MaskType maskCanonical;

	BBInfo() {}
	BBInfo(const OBB3f& _obb, const bbox3f& _aabb, unsigned short _labelId, unsigned short _instanceId,
		const bbox3f& _aabbCanonical, float _angleCanonical) {
		obb = _obb;
		aabb = _aabb;
		labelId = _labelId;
		instanceId = _instanceId;
		aabbCanonical = _aabbCanonical;
		angleCanonical = _angleCanonical;
	}
};
#else
typedef Grid3<unsigned short> MaskType;
struct BBInfo {
	OBB3f obb;
	bbox3f aabb;
	unsigned short labelId;
	unsigned short instanceId;

	MaskType mask;
	
	BBInfo() {}
	BBInfo(const OBB3f& _obb, const bbox3f& _aabb, unsigned short _labelId, unsigned short _instanceId) {
		obb = _obb;
		aabb = _aabb;
		labelId = _labelId;
		instanceId = _instanceId;
	}
};
#endif
