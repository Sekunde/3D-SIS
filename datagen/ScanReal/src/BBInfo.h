#pragma once

typedef Grid3<unsigned short> MaskType;
struct BBInfo {
	bbox3f aabb;
	unsigned short labelId;
	unsigned short instanceId;

	MaskType mask;
	
	BBInfo() {}
	BBInfo(const bbox3f& _aabb, unsigned short _labelId, unsigned short _instanceId) {
		aabb = _aabb;
		labelId = _labelId;
		instanceId = _instanceId;
	}
};
