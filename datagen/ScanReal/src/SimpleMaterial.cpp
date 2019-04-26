
#include "stdafx.h"
#include "SimpleMaterial.h"



std::ostream& operator<<(std::ostream& s, const SimpleMaterial& m)
{
	s << "ambient\t" << m.ambient << std::endl;
	s << "diffuse\t" << m.diffuse << std::endl;
	s << "specular\t" << m.specular << std::endl;
	s << "shiny\t" << m.shiny << std::endl;
	return s;
}
