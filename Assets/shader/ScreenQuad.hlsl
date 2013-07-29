#define SAMPLES 8

Texture2DMS<float4, SAMPLES> texture2render : register(t0);
SamplerState SampleType : register(s2);

struct PixelInput 
{
    float2 texCoord : TEXCOORD0;
    float4 position : SV_POSITION;
};

struct PixelOutput 
{
    float4 color : SV_Target0;
};

struct VertexInput {
    float3 position : POSITION0;
    float2 texCoord : TEXCOORD0;
};


PixelInput RT_VS(VertexInput input) 
{
    PixelInput op;
    op.position = float4(input.position, 1);
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput RT_PS(PixelInput input)
{
    PixelOutput op;
    op.color = float4(0,0,0,0);
    int3 dim;
    texture2render.GetDimensions(dim.x, dim.y, dim.z);
    [unroll]
    for(int i = 0; i < SAMPLES; ++i)
    {
        op.color += texture2render.Load(int2(dim.xy * input.texCoord), i);
    }
    op.color *= (1.0 / SAMPLES);
    return op;
}