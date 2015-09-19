#include "ShaderGlobals.h"

struct PixelInput 
{
    float2 texCoord : TEXCOORD0;
    float4 position : SV_POSITION;
};

struct PixelOutput 
{
    float4 color : SV_Target0;
};

struct VertexInput 
{
    float3 position : POSITION0;
    float2 texCoord : TEXCOORD0;
};

PixelInput Effect_VS(VertexInput input) 
{
    PixelInput op;
    op.position = float4(input.position, 1);
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput Luminance(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    float lumi = color.r * 0.27 + color.g * 0.67 + color.b * 0.06;
    lumi = log(0.01 + lumi);
    op.color = float4(lumi, lumi, lumi, 0);
    op.color.a = 1;
    return op;
}

PixelOutput SampleDiffuseTexture(PixelInput input)
{
    PixelOutput op;
    float4 color = g_diffuseColor.Sample(g_samplerClamp, input.texCoord);
    op.color = color;
    return op;
}

PixelOutput Sample(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    op.color = color;
    return op;
}

static const float weights[7] = {
0.00038771,  
0.01330373,
0.11098164,
0.22508352,
0.11098164,
0.01330373,
0.00038771
};

PixelOutput BlurH(PixelInput input)
{
    float4 color = float4(0, 0, 0, 0);

    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(1.0 / w, 0);

    float2 fs_in_tex = input.texCoord;
        
    float4 c0 = 0.0000000076834112 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (15.0 + 0.030303)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (15.0 + 0.030303)*texelSize));

    float4 c1 = 0.0000012703239918 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (13.0 + 0.090909)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (13.0 + 0.090909)*texelSize));

    float4 c2 = 0.0000552590936422 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (11.0 + 0.151515)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (11.0 + 0.151515)*texelSize));

    float4 c3 = 0.0009946636855602 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (9.0 + 0.212121)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (9.0 + 0.212121)*texelSize));

    float4 c4 = 0.0089796027168632 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (7.0 + 0.272727)*texelSize)
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (7.0 + 0.272727)*texelSize));

    float4 c5 = 0.0450612790882587 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (5.0 + 0.333333)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (5.0 + 0.333333)*texelSize));

    float4 c6 = 0.1334507111459971 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (3.0 + 0.393939)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (3.0 + 0.393939)*texelSize));

    float4 c7 = 0.2414822392165661 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (1.0 + 0.454545)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (1.0 + 0.454545)*texelSize));

    float4 c8 = 0.1399499340914190 * g_effectSource0.Sample(g_samplerClamp, fs_in_tex);
      
    PixelOutput op;
      op.color = float4(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8);

    return op;
}

PixelOutput BlurV(PixelInput input)
{
    float4 color = float4(0, 0, 0, 0);

    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(0, 1.0 / h);

    float2 fs_in_tex = input.texCoord;
        
    float4 c0 = 0.0000000076834112 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (15.0 + 0.030303)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (15.0 + 0.030303)*texelSize));

    float4 c1 = 0.0000012703239918 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (13.0 + 0.090909)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (13.0 + 0.090909)*texelSize));

    float4 c2 = 0.0000552590936422 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (11.0 + 0.151515)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (11.0 + 0.151515)*texelSize));

    float4 c3 = 0.0009946636855602 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (9.0 + 0.212121)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (9.0 + 0.212121)*texelSize));

    float4 c4 = 0.0089796027168632 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (7.0 + 0.272727)*texelSize)
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (7.0 + 0.272727)*texelSize));

    float4 c5 = 0.0450612790882587 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (5.0 + 0.333333)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (5.0 + 0.333333)*texelSize));

    float4 c6 = 0.1334507111459971 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (3.0 + 0.393939)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (3.0 + 0.393939)*texelSize));

    float4 c7 = 0.2414822392165661 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (1.0 + 0.454545)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (1.0 + 0.454545)*texelSize));
    
    float4 c8 = 0.1399499340914190 * g_effectSource0.Sample(g_samplerClamp, fs_in_tex);
      
    PixelOutput op;
    op.color = float4(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8);
    return op;
}
#define FILTER_SIZE 7
PixelOutput VSMBlurV(PixelInput input)
{
    PixelOutput op;
    float4 color = float4(0, 0, 0, 0);
    uint filterSize = FILTER_SIZE;
    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(1.0 / w, 0);

    float2 fs_in_tex = input.texCoord - texelSize * filterSize / 2;
    
    for(uint i = 0; i < filterSize; ++i)
    {
        color += g_effectSource0.Sample(g_samplerClamp, fs_in_tex + texelSize * i);
    }
    color /= filterSize;
    op.color = color;
    return op;
}

PixelOutput VSMBlurH(PixelInput input)
{
    PixelOutput op;
    float4 color = float4(0, 0, 0, 0);
    uint filterSize = FILTER_SIZE;
    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(1.0 / h, 0);

    float2 fs_in_tex = input.texCoord - texelSize * filterSize / 2;
    
    for(uint i = 0; i < filterSize; ++i)
    {
        color += g_effectSource0.Sample(g_samplerClamp, fs_in_tex + texelSize * i);
    }
    color /= filterSize;
    op.color = color;
    return op;
}

PixelOutput Brightness(PixelInput input)
{
    PixelOutput op;
    op.color = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
    if(!(op.color.x > 1 || op.color.y > 1 || op.color.z > 1))
    {
        discard;
    }
    return op;
}

float3x3 GetTangentSpaceMatrix3(float3 N, float3 p, float2 uv)
{
    // get edge vectors of the pixel triangle
    float3 dp1 = ddx( p );
    float3 dp2 = ddy( p );
    float2 duv1 = ddx( uv );
    float2 duv2 = ddy( uv );
 
    // solve the linear system
    float3 dp2perp = cross( dp2, N );
    float3 dp1perp = cross( N, dp1 );
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    //if(dot(cross(B, T), N) < 0)
    {
        B *= -1;
    }
 
    // construct a scale-invariant frame 
    float invmax = rsqrt( max( dot(T,T), dot(B,B) ) );
    return transpose(float3x3( T * invmax, B * invmax, N ));
}

PixelOutput SSAA(PixelInput input)
{
    PixelOutput op;

    float4 posDepth = g_worldPosDepth.Sample(g_samplerClamp, input.texCoord);

    if(posDepth.w <= 0)
    {
        op.color = float4(1,1,1,1);
        return op;
    }

    uint w, h, levels;
    g_worldPosDepth.GetDimensions(0, w, h, levels);

    const float noiseSizeSqrt = 4;
    float2 noiseScale = float2(w / noiseSizeSqrt, h / noiseSizeSqrt);

    float3 rvec = g_effectSource2.Sample(g_samplerWrap, noiseScale * input.texCoord).xyz;
    float3 N = mul(g_view, float4(g_normals.Sample(g_samplerClamp, input.texCoord).xyz, 0)).xyz;
    float3 T = normalize(rvec - N * dot(rvec, N));
    float3 B = cross(N, T);

    float3 origin = mul(g_view, float4(posDepth.xyz, 1)).xyz;

    float3x3 tbn = transpose(float3x3(normalize(T), normalize(B), normalize(N)));

    //float4 color = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
     
    float occlusion = 0.0;
    float radius = min(4.2, origin.z);

    const int kernelSize = 64;

    //op.color = color;
    [unroll]
    for (int i = 0; i < kernelSize; ++i) 
    {
       float3 sample = mul(tbn, g_effectSource1.Sample(g_samplerClamp, float2((float)i / (float)kernelSize, 0)).xyz);
       sample = sample * radius + origin;
  
       float4 offset = float4(sample, 1.0);
       offset = mul(g_projection, offset);
       offset.xy /= offset.w;
       offset.xy = offset.xy * 0.5 + 0.5;
       offset.xy = float2(offset.x, 1-offset.y);
  
       float sampleDepth = g_worldPosDepth.Sample(g_samplerClamp, offset.xy).w;
  
       float rangeCheck = abs(origin.z - sampleDepth) < radius ? 1.0 : 0.0;
       occlusion += (sampleDepth <= sample.z ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / (float)kernelSize);
    op.color = float4(occlusion, occlusion, occlusion, 1);
    return op;
}

PixelOutput SSAAAfterBlur(PixelInput input)
{
    PixelOutput op;
    float4 blur = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    float4 color = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
    op.color = color * blur * blur;
    return op;
}

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

float3 Uncharted2Tonemap(float3 x)
{
     return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

PixelOutput ToneMap(PixelInput input)
{
    PixelOutput op;
    float4 bright = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    float4 color = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
    float avgLum = exp(g_effectSource2.Sample(g_samplerClamp, input.texCoord).x);
    
    float lumi = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;

    float key = 0.75; //max(0, 1.5 - 1.5 / (avgLum*0.1 + 1)) + 0.1; //ke = 10 mehr glitzer
    float yr = key * lumi / avgLum;
    float lumiScaled = yr / (1 + yr);

    //color += bright;
    //color = color * lumiScaled;
    
    op.color = color; //float4(avgLum); //pow(abs(color), 1.0 / 2.2); gamma correction
    return op;
}

#define VLS_SAMPLES 80

PixelOutput LightScattering(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
    float4 screenSpaceLightPos = mul(g_projection, mul(g_view, float4(g_lightPos.xyz, 1))); //scale out
    screenSpaceLightPos /= screenSpaceLightPos.w;
    
    float2 nssl = 0.5f * float2(screenSpaceLightPos.x, screenSpaceLightPos.y) + 0.5f;
    nssl = float2(nssl.x, 1 - nssl.y);
    float2 deltaTexoord = input.texCoord - nssl;
    
    float density = 1;
    
    deltaTexoord *= 1.0f / VLS_SAMPLES * density;
    
    float illumDecay = 1.0f;
    float decay = 0.98f;
    float weight = 1.0f / VLS_SAMPLES;
    float2 tc = input.texCoord;
    for(int i = 0; i < VLS_SAMPLES; ++i)
    {
        tc -= deltaTexoord;
        float depth = g_effectSource1.Sample(g_samplerClamp, tc).w;
        //float4 sample = illumDecay * g_effectSource1.Sample(g_samplerClamp, tc) * weight;
        color = (color * 0.5) * sign(depth);
        illumDecay *= decay;
    }
    
    op.color = color;

    return op;
}

