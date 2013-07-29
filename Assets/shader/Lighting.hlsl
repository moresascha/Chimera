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

PixelInput Lighting_VS(VertexInput input) 
{
    PixelInput op;
    op.position = float4(input.position, 1);
    op.texCoord = input.texCoord;
    return op;
}

int isOut(in float3 world, uint index, out float2 tc, out float4 worldInLight)
{
    worldInLight = mul(g_lightProjection[index], mul(g_lightView, float4(world, 1)));
    worldInLight /= worldInLight.w;
    tc = 0.5 * float2(worldInLight.x, worldInLight.y) + 0.5;
    tc = float2(tc.x, 1 - tc.y);
    float nearClip = 0;//index < 2 ? 0.2 : 0;
    return tc.x < 0 || tc.x > 1 || tc.y < 0 || tc.y > 1 || worldInLight.z < nearClip || worldInLight.z > 1;
}

void computeCSMContr(out in float4 color, in float2 texCoords, in float3 normal)
{

    float3 world = g_worldPosDepth.Sample(g_samplerClamp, texCoords).xyz;
    float lightPosDepth = mul(g_lightView, float4(world, 1)).z;

    uint index = 0;
    float2 tc = float2(0,0);
    float4 worldInLight = float4(0,0,0,0);

    if(!isOut(world, 0, tc, worldInLight))
    {
        index = 0;
    }
    else if(!isOut(world, 1, tc, worldInLight))
    {
        index = 1;
    }
    else if(!isOut(world, 2, tc, worldInLight))
    {
        index = 2;
    }
    else
    {
        return;
    }

    float4 colorMask = 4 * float4(index == 0, index == 1, index == 2, 1);
    //color *= colorMask;

    float4 gs[3];
    gs[0] = g_effectSource0.Sample(g_samplerClamp, tc);
    gs[1] = g_effectSource1.Sample(g_samplerClamp, tc);
    gs[2] = g_effectSource2.Sample(g_samplerClamp, tc);

    //gs[3] = g_cascadedShadows[3].Sample(g_samplerClamp, tc).x;
    //color = float4(gs[index],0,0,1);

    float2 moments = gs[index].xy;
      
    float dist = g_distances[index];
        
    float rescaled_dist_to_light = (120 + lightPosDepth) / dist;

    float light_shadow_bias = 0;//0.1;//-0.1f;
    
    float light_vsm_epsilon = 0.00001f;
    
    rescaled_dist_to_light -= light_shadow_bias;
    
    float lit_factor = (rescaled_dist_to_light <= moments.x);
    
    // Variance shadow mapping
    float E_x2 = moments.y;
    float Ex_2 = moments.x * moments.x;
    float variance = min(max(E_x2 - Ex_2, 0.0) + light_vsm_epsilon, 1);
    float m_d = (rescaled_dist_to_light - moments.x);
    float p = variance / (variance + m_d * m_d);

    color *= max(lit_factor, max(p, 0.25));
    
    /*if(lightPosDepth > (moments.x + bias))
       {

           // if(dot(dd, float3(0,0,1)) > 0)
            {
                color *= 0.3;//float4(0,0,0,0);
            }
        } */
}

PixelOutput DebugGlobalLighting_PS(PixelInput input)
{
    PixelOutput op;
    op.color = float4(0,0,0,0);

    float3 normal = 0;
    float3 diffuse = 0;

    float4 nn = g_normals.Sample(g_samplerClamp, input.texCoord);
    normal = nn.xyz;
    diffuse = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord).xyz;

    float3 sunposition = normalize(g_lightPos.xyz);
    diffuse = float3(1,1,1);
    op.color = float4(diffuse * saturate(dot(sunposition, normalize(normal))),1);
    
    return op;
}

PixelOutput GlobalLighting_PS(PixelInput input)
{
    PixelOutput op;
    op.color = float4(0,0,0,0);

    float3 normal = 0;
    float3 ambientMat = 0;
    float3 diffuseMat = 0;
    float3 diffuse = 0;

    ambientMat = g_ambientMaterialSpecG.Sample(g_samplerClamp, input.texCoord).xyz;
    diffuseMat = g_diffuseMaterialSpecR.Sample(g_samplerClamp, input.texCoord).xyz;
    float4 nn = g_normals.Sample(g_samplerClamp, input.texCoord);
    normal = nn.xyz;
    diffuse = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord).xyz;

    float3 sunposition = normalize(g_lightPos.xyz);

    float4 worldDepth = g_worldPosDepth.Sample(g_samplerClamp, input.texCoord);
    float d = worldDepth.w;

    float4 sun = 0.9 * float4(253 / 255.0, 184 / 255.0, 0 / 255.0, 0);
    float4 sky = 0;//0.7 * float4(135 / 255.0, 206 / 255.0, 1, 0);

    float scale = 1.65f;

    if(d > 0)
    {
        //reflectance
        float ref = g_normalColor.Sample(g_samplerClamp, input.texCoord).x;
        float3 skyTex = float3(0,0,0);
        if(ref > 0)
        {
            float3 reflectVec = normalize(reflect(worldDepth.xyz - g_eyePos.xyz, normal));
            float u = 0.5 + atan2(reflectVec.z, reflectVec.x) / (2 * PI);
            float v = 0.5 - asin(reflectVec.y) / PI;
            float2 tc = float2(u,1-1.5*v);
            skyTex = g_diffuseColor.Sample(g_samplerClamp, tc).xyz;
            skyTex *= skyTex;
            float refScale = saturate(0.6 + dot(float3(0,1,0), reflectVec));
            skyTex *= refScale;
        }

        int selfShade = nn.w < 0;
        
        float3 factor = (0.5 * diffuseMat * saturate(dot(sunposition, normalize(normal))));
        diffuse = lerp(diffuse, skyTex, ref);
        //op.color = float4(normal, 1);
        op.color = float4(diffuse * (scale * factor), 1);
        
        //CSM
        /*if(selfShade) //hack to avoid peter panning, todo
        {
            op.color *= (0.65 + 0.4 * nn.w);
        }
        else */
        {
            computeCSMContr(op.color, input.texCoord, normal);
        }

        op.color += float4(ambientMat * diffuse,0);
    } 
    else
    {
        float4 ray = float4(-1.0 +  2.0 * float2(input.texCoord.x, 1 - input.texCoord.y), 1, 0);
        ray.y *= 1;
        ray.w = 0;
        ray = mul(g_invView, ray);
        ray = normalize(ray);

        float4 tex = float4(diffuse,0);

        float powa = pow(saturate(dot(ray.xyz, sunposition)), 32);
        
        float l = clamp(worldDepth.y * 0.1, 0, 1);
        op.color = clamp(scale, 0, 1) * lerp(sky, tex, l);
        //op.color += sun * powa;
    }

    return op;
}

PixelOutput PointLighting_PS(PixelInput input)
{
    PixelOutput op;

    float3 normal = 0;
    float3 ambientMat = 0;
    float3 diffuseMat = 0;
    float3 specularMat = 0;
    float3 diffuseColor = 0;
    float3 world = 0;
    float depth = 0;

    float4 wd =  g_worldPosDepth.Sample(g_samplerClamp, input.texCoord);
    world = wd.xyz; 
    depth = wd.w;

    /*if(depth == 0)
    {
        discard;
    } */

    float3 lightPos = g_pointLightPos.xyz;

    float3 d = world - lightPos;

    float radius = g_pointLightColorRadius.w;

    float distSquared = dot(d, d);

    if(distSquared > radius*radius) 
    {
        discard;
    }

    float3 lightToPos = normalize(world - lightPos);

    float shadowSample = g_pointLightShadowMap.Sample(g_samplerClamp, lightToPos).r;
    float bias = 0.15;//0.005 * tan(acos(dot(-lightToPos, normal)));//0.15
    //bias = saturate(bias);
    int shadow = shadowSample < (distSquared - bias) ? 1 : 0;

    if(shadow)
    {
        discard;
    }

    normal = g_normals.Sample(g_samplerClamp, input.texCoord).xyz;
    normal = normalize(normal);

    float4 dmsmr = g_diffuseMaterialSpecR.Sample(g_samplerClamp, input.texCoord);
    float4 amsmg = g_ambientMaterialSpecG.Sample(g_samplerClamp, input.texCoord);
    float4 dcsmb = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord);

    ambientMat = amsmg.xyz;
    diffuseMat = dmsmr.xyz;
    specularMat = float3(dmsmr.w, amsmg.w, dcsmb.w);

    diffuseColor = dcsmb.xyz;

    float3 lightColor = g_pointLightColorRadius.xyz;
    float3 posToEye = normalize(g_eyePos.xyz - world);

    float3 reflectVec = reflect(lightToPos, normal);

    //float bias = (distSquared * DepthBias) - shadowSample;

    float s = sign(dot(abs(normal), abs(normal)));

    float diffuse = s * saturate(dot(-lightToPos, normal)) + (1-s);

    float specular = s * pow(saturate(dot(reflectVec, posToEye)), 32) + (1-s);

    float intensity = max(0, 1 - distSquared / (radius * radius));

    float3 color = 0;

    color = lightColor * (specular * specularMat + diffuseMat * diffuse * diffuseColor);
    
    color *= intensity; // * fShadow;

    op.color = float4(color, 1);
    return op;
}