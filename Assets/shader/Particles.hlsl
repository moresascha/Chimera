#include "ShaderGlobals.h"

struct PixelOutput 
{
    float4 worldPosDepth : SV_Target0;
    float4 normal : SV_Target1;
    half4 diffMaterialSpecR : SV_Target2;
    half4 ambientMaterialSpecG : SV_Target3;
    half4 diffuseColorSpecB : SV_Target4;
    //float2 specExReflectance : SV_Target5;
};
/*
struct PixelOutput 
{
    float4 color : SV_Target0;
};
*/

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
    float4 particlePosNAlive : INSTANCED_POSITION;
    float3 velo : INSTANCED_VELO;
};

struct PixelInput 
{
    float2 texCoord : TEXCOORD0;
    float4 position : SV_POSITION;
    float4 world : POSITION1;
    float4 worldView : POSITION2;
    float3 normal : NORMAL0;
    float3 projectedVelo : NORMAL1;
    float alive : POSITION3;
};

PixelInput Particle_VS(VertexInput input) 
{
    PixelInput op;

    float3 dir = normalize(input.velo.xyz); //normalize(mul(g_model, float4(input.velo.xyz, 0))).xyz;

    float phi = atan2(dir.x, dir.z);
    float theta = PI + acos(dir.y);

    /*
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    float3x3 rot = float3x3(
                    cosPhi, -sinPhi, 0,
                    sinPhi, cosPhi, 0,
                    0, 0, 1
                    ); */

    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    float3x3 rot = float3x3(
                         cosPhi, sinPhi*sinTheta, sinPhi*cosTheta,
                              0,        cosTheta,       -sinTheta,
                        -sinPhi, cosPhi*sinTheta, cosPhi*cosTheta
                        );
    
    float3 tp = float3(0.5*input.position.x, 0.25*input.position.y, 0);
    op.world = float4(mul(rot, tp), 1);
    //op.world = float4(input.position, 1);

    op.world = float4(input.particlePosNAlive.xyz, 0) + op.world;//mul(g_model, op.world);

    op.worldView = mul(g_view, op.world);
    op.position = mul(g_projection, op.worldView);
    op.texCoord = input.texCoord;
    op.normal = input.normal; //mul(g_model, float4(input.normal, 0)).xyz;
    op.alive = input.particlePosNAlive.w;
    float4 v = float4(input.velo, 0);
    //v = float4(0,1,0,0);
    dir = mul(g_view, float4(dir, 0)).xyz;
    //float4 pdir = mul(g_projection, float4(dir, 1));
    //pdir /= pdir.w;
    op.projectedVelo = dir.xyz;
    return op;
}

PixelOutput Particle_PS(PixelInput input)
{
    float2 tc = 2 * input.texCoord - 1;
    tc.y = - tc.y;

    float yd = 1000;//abs(input.projectedVelo.y);
    
    float border = -(tc.x * tc.x) * yd;

    float offset = 1;

    if(tc.y > border + offset || tc.y < -border - offset)
    {
        discard;
    }

    /*float xd = abs(input.projectedVelo.x);
    
    border = -(tc.y * tc.y) * xd;

    if(tc.x > border + offset || tc.x < -border - offset)
    {
        discard;
    }
    */
    if(input.alive < 0.5 || tc.x*tc.x + tc.y*tc.y > 1)//|| f > 0)
    {
        discard;
    }
    
    PixelOutput op;
    
    op.worldPosDepth = input.world;
    op.worldPosDepth.w = length(input.worldView.xyz);

    op.normal = float4(0,0,0,0);
    half scale = 16;
    half3 color = half3(0.9,1,0.1);
    //color.xyz = float3(1,1,1) * dot(normalize(input.projectedVelo), getViewDir());
    color *= scale;
    op.diffMaterialSpecR = half4(1,1,1,1);
    op.ambientMaterialSpecG = half4(1,1,1,1);
    op.diffuseColorSpecB = half4(color,1);
    
    return op;
}