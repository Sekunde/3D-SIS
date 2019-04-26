
cbuffer ConstantBufferCamera : register( b0 )
{
	matrix worldViewProj;
	matrix world;
	float4 eye;
}

cbuffer ConstantBufferMaterial : register( b1 )
{
	float4	ambient;
	float4	diffuse;
	float4	specular;
	float	shiny;		   //defined as 1/roughness
	float3	dummyMaterial;
}

#define MAX_NUM_LIGHTS 10
cbuffer ConstantBufferLight : register ( b2 )
{
	float4 lightDirs[MAX_NUM_LIGHTS];
	unsigned int numLights;
	float3 dummy123;
}



struct VertexShaderOutput
{
    float4 position : SV_POSITION;
	float4 normal : TEXCOORD0;
	float4 view : TEXCOORD1;
};

VertexShaderOutput vertexShaderMain( float4 position :	position,
									 float4 normal :	normal,
									 float4 color :		color,
									 float4 texCoord :	texCoord )
{
    VertexShaderOutput output;
    output.position = mul( position, worldViewProj );
	float4 posWorld = mul(position, world); 
	output.view = eye - posWorld;
	normal.w = 0.0f;
	output.normal = mul(normal, world);
    return output;
}

#define ONE_OVER_PI			0.31830989
#define ONE_OVER_FOUR_PI	0.07957747


float3 ward(float3 normal, float3 viewDir, float3 lightDir)
{
	normal = normalize(normal);
	lightDir = normalize(lightDir);
	viewDir = normalize(viewDir);

	float costi = saturate(dot(normal, lightDir)); 
	float costo = saturate(dot(normal, viewDir)); 
	float diff = ONE_OVER_PI*costi;

	float spec = 0.0f;
	if (shiny > 0.0f && costi > 0.0f && costo > 0.0f) {
		float3 H = normalize(lightDir + viewDir); // half angle vector
		float t = tan(acos(dot(H, normal)));
		float v = sqrt(costi/costo);
		float s2 = shiny*shiny;
		spec = exp(-t*t*s2)*ONE_OVER_FOUR_PI*s2*v;
	}

	float lightColor = 3.0f; //TODO: specify light intensity
	return (ambient.xyz + diffuse.xyz*diff + specular.xyz*spec)*lightColor;
} 

// blinn phong
float3 blinn_phong(float3 normal, float3 viewDir, float3 lightDir)
{
	normal = normalize(normal);
	lightDir = normalize(lightDir);
	viewDir = normalize(viewDir);

	float costi = saturate(dot(normal, lightDir));
	float spec = 0.0f;
	if (shiny > 0.0f) {
		float3 H = normalize(lightDir + viewDir); // half angle vector
		spec = pow(saturate(dot(H, normal)), shiny/4.0);
	}

	//return ambient.xyz + diffuse.xyz*ONE_OVER_PI*costi*3; 
	//return ambient.xyz + (diffuse.xyz*ONE_OVER_PI + specular.xyz*spec)*costi;
	return (diffuse.xyz*ONE_OVER_PI + specular.xyz*spec)*costi * 3;
}

float4 pixelShaderMain( VertexShaderOutput input ) : SV_Target
{
	float3 res = 0.0f;
	for (unsigned int i = 0; i < numLights; i++)	{
		float3 curr = ward(input.normal.xyz, input.view.xyz, lightDirs[i]);
		res += curr;
	}
	//return float4(1.0f, 0.0f, 0.0f, 1.0f);
	return float4(res, 1.0f); 
}
