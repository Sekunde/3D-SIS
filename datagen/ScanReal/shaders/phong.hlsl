
cbuffer ConstantBufferCamera : register( b0 )
{
	matrix worldViewProj;
	matrix world;
	float4 eye;
}

Texture2D modelTexture : register(t0);
SamplerState modelSampler : register(s0);

cbuffer ConstantBufferMaterial : register( b1 )
{
	float4	ambient;
	float4	diffuse;
	float4	specular;
	float	shiny;
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
	//float4 color : TEXCOORD1;
	float4 view : TEXCOORD2;
	float4 texCoord : TEXCOORD3;
	nointerpolation float4 color : TEXCOORD1;
};

VertexShaderOutput vertexShaderMain( float4 position :	position,
									 float4 normal :	normal,
									 float4 color :		color,
									 float4 texCoord :	texCoord )
{
    VertexShaderOutput output;
    output.position = mul(position, worldViewProj);
	float4 posWorld = mul(position, world); 
	output.view = normalize(eye - posWorld);
	normal.w = 0.0f;
	output.normal = mul(normal, world);
	output.color = color;
	output.texCoord = texCoord;
    return output;
}

float3 phong(float3 normal, float3 viewDir, float3 lightDir)
{
	normal = normalize(normal);
	lightDir = normalize(lightDir);
	viewDir = normalize(viewDir);

	//float4 diff = saturate(dot(normal, lightDir)); // diffuse component
	float3 diff = saturate(dot(normal, lightDir)); // diffuse component

	// R = 2 * (N.L) * N - L
	float3 reflect = normalize(2 * dot(normal, lightDir) * normal - lightDir);
	float3 spec = 0.0f;
	
	if (shiny > 0.0f) {
		spec = pow(saturate(dot(reflect, viewDir)), shiny); // R.V^n
	}

	return ambient.xyz + diffuse.xyz*diff.xyz + specular.xyz*spec.xyz;
}

float4 shade_phong( VertexShaderOutput input ) {
	float3 color = input.color.xyz;

	float3 res = 0.0f;
	for (unsigned int i = 0; i < numLights; i++)	{
		float3 curr = phong(input.normal.xyz, input.view.xyz, lightDirs[i]) * color;
		res += curr;
	}

	return float4(res, input.color.w);
}

float4 pixelShaderMain( VertexShaderOutput input ) : SV_Target
{
	return shade_phong(input);
}

float4 pixelShaderMain_textured(VertexShaderOutput input) : SV_Target
{
	float4 shadeColor = shade_phong(input);
	float4 texColor = modelTexture.Sample(modelSampler, float2(input.texCoord.x, 1.0f - input.texCoord.y));

	//return float4(texColor.xyz, 1.0f);
	//return float4(shadeColor.xyz, 1.0f);

	//return float4(shadeColor.xyz * texColor.xyz, 1.0f);
	return float4(shadeColor.xyz * texColor.xyz, input.color.w);
}
