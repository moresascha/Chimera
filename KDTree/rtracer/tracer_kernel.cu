#pragma once
#include "../kdtree.cuh"
#include <cutil_math.h>

struct Sphere
{
    float3 pos;
    float radius;
    float4 color;
};

/*__device__ float getAxis(float4* val, int axis) 
{
    switch(axis)
    {
        case 0 : return val->x;
        case 1 : return val->y;
        case 2 : return val->z;
        case 3 : return val->w;
    }
    return -1;
}*/

__device__ int intersectP(float4* eye, float4* ray, float4* boxmin, float4* boxmax, float* tmin, float* tmax) {

    float t0 = 0; float t1 = FLT_MAX;

    float4 invRay = 1.0 / *ray;

    for(uint i = 0; i < 3; ++i) 
    {
        float tNear = (getAxis(boxmin, i) - getAxis(eye, i)) * getAxis(&invRay, i);
        float tFar = (getAxis(boxmax, i) - getAxis(eye, i)) * getAxis(&invRay, i);

        if(tNear > tFar) 
        {
            float tmp = tNear;
            tNear = tFar;
            tFar = tmp;
        }

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;

        if(t0 > t1) return 0;
    }
    *tmin = t0;
    *tmax = t1;
    return 1;
}

//seite 118 in PBRT, Compute quadratic sphere coeffficienten
__device__ float4 computeQuadraticCoefs(float4* eye, float4* ray, float radius) 
{
    float4 v;
    v.x = 1;//ray.x*ray.x + ray.y*ray.y + ray.z*ray.z;
    v.y = 2 * (ray->x*eye->x + ray->y*eye->y + ray->z*eye->z);
    v.z = eye->x*eye->x + eye->y*eye->y + eye->z*eye->z - radius*radius;
    return v;
}

//seite 118 in PVRT, Quadtratic
__device__ int quad(float A, float B, float C, float* t0, float* t1) 
{
    float d = B * B - 4.0 * A * C;
    if(d <= 0) return 0;
    float rd = sqrt(d);
    float q;
    if(B < 0) 
    {
        q = -0.5 * (B - rd);
    } else {
        q = -0.5 * (B + rd);
    }
    float x = q / A;
    float y = C / q;
    if(x > y) 
    {
        *t0 = y;
        *t1 = x;
    } else {
        *t0 = x;
        *t1 = y;
    }
    return 1;
}

//119 in PBRT, Compute intersection distance along ray
__device__ int isHit(float t0, float t1, float n, float f, float* hit) 
{
    if(t0 > f || t1 < n) return 0;
    *hit = t0;
    if(t0 < n) 
    {
        *hit = t1;
        if(*hit > f) return 0;
    }
    return 1;
}

__device__ void mul(float* m4x4l, float* m4x4r, float* result)
{
    float4 r0 = make_float4(*m4x4l, *(m4x4l+1), *(m4x4l+2), *(m4x4l+3));
    float4 r1 = make_float4(*(m4x4l+4), *(m4x4l+5), *(m4x4l+6), *(m4x4l+7));
    float4 r2 = make_float4(*(m4x4l+8), *(m4x4l+9), *(m4x4l+10), *(m4x4l+11));
    float4 r3 = make_float4(*(m4x4l+12), *(m4x4l+13), *(m4x4l+14), *(m4x4l+15));

    float4 c0 = make_float4(*m4x4r, *(m4x4r+1), *(m4x4r+2), *(m4x4r+3));
    float4 c1 = make_float4(*(m4x4r+4), *(m4x4r+5), *(m4x4r+6), *(m4x4r+7));
    float4 c2 = make_float4(*(m4x4r+8), *(m4x4l+9), *(m4x4r+10), *(m4x4r+11));
    float4 c3 = make_float4(*(m4x4r+12), *(m4x4r+13), *(m4x4r+14), *(m4x4r+15));

    *(result+0) = dot(r0, c0);
    *(result+1) = dot(r0, c1);
    *(result+2) = dot(r0, c2);
    *(result+3) = dot(r0, c3);

    *(result+4) = dot(r1, c0);
    *(result+5) = dot(r1, c1);
    *(result+6) = dot(r1, c2);
    *(result+7) = dot(r1, c3);

    *(result+8) = dot(r2, c0);
    *(result+9) = dot(r2, c1);
    *(result+10) = dot(r2, c2);
    *(result+11) = dot(r2, c3);

    *(result+12) = dot(r3, c0);
    *(result+13) = dot(r3, c1);
    *(result+14) = dot(r3, c2);
    *(result+15) = dot(r3, c3);
}

__device__ float4 transform(float* m4x4l, float4* vector)
{
    float4 r0 = make_float4(*m4x4l, *(m4x4l+4), *(m4x4l+8), *(m4x4l+12));
    float4 r1 = make_float4(*(m4x4l+1), *(m4x4l+5), *(m4x4l+9), *(m4x4l+13));
    float4 r2 = make_float4(*(m4x4l+2), *(m4x4l+6), *(m4x4l+10), *(m4x4l+14));
    float4 r3 = make_float4(*(m4x4l+3), *(m4x4l+7), *(m4x4l+11), *(m4x4l+15));

    return make_float4(dot(r0, *vector), dot(r1, *vector), dot(r2, *vector), dot(r3, *vector));
}

__device__ float4 refract(float4* i, float4* n, float eta)
{
  float cosi = dot(- *i, *n);
  float cost2 = 1.0f - eta * eta * (1.0f - cosi*cosi);
  float4 t = eta * (*i) + ((eta*cosi - sqrt(abs(cost2))) * (*n));
  return t * make_float4(cost2 > 0);
}

struct HitResult
{
    float4 normal;
    float4 worldPosition;
    float t;
    int isHit;
};

__device__ int computeHit(float3* sphere, float4* eye, float4* ray, float tmin, float tmax, float radius, HitResult* result)
{
    float4 sphereToEye = *eye - make_float4(sphere->x, sphere->y, sphere->z, 0);
    float4 cos = computeQuadraticCoefs(&sphereToEye, ray, radius);
    float t0, t1;
    if(!quad(cos.x, cos.y, cos.z, &t0, &t1)) 
    {
        return 0;
    }

    float t;
    if(!isHit(t0, t1, tmin, tmax, &t))
    {
        return 0;
    }

    if(t > result->t)
    {
        return 0;
    }

    result->worldPosition = (*eye + *ray * t);
    result->normal = normalize(result->worldPosition - make_float4(sphere->x, sphere->y, sphere->z, 0));
    result->t = t;
    result->isHit = 1;

    return 1;
}

struct ToDo 
{
    Node node;
    float min;
    float max;
    uint treePos;
    uint level;
};

__device__ void traversal0(float3* spheres, Node n, float4* eye, float4* ray, HitResult* hit, float4* min, float4* max, uint treeDepth) 
{
    float sceneMin, sceneMax;

    hit->isHit = 0;

    if(!intersectP(eye, ray, min, max, &sceneMin, &sceneMax))
    {
        return;
    }

    //if(sceneMin < 0.0f) sceneMax = 0.0f;

    hit->t = FLT_MAX;

    float tmin;
    float tmax;
    tmin = tmax = sceneMin;
    int pushDown = 1;
    uint root = 0;
    uint lastlevel = 0;

    float c = 1;

    while(tmax < sceneMax) 
    {
        uint nodeIndex = root;
        tmin = tmax;
        tmax = sceneMax;
        pushDown = 1;

        while(!n.leaf[nodeIndex])
        {

            byte axis = n.axis[nodeIndex];
            float nsplit = n.split[nodeIndex];

            float tsplit = (nsplit- getAxis(eye, axis)) / getAxis(ray, axis);

            int belowFirst = (getAxis(eye, axis) < nsplit) || ((getAxis(eye, axis) == nsplit) && (getAxis(ray, axis) >= 0));

            uint first, second;

            if(belowFirst)
            {
                first = nodeIndex + nodeIndex + 1;
                second = nodeIndex + nodeIndex + 2;
            } 
            else 
            {
                first = nodeIndex + nodeIndex + 2;
                second = nodeIndex + nodeIndex + 1;
            }

            if(tsplit >= tmax || tsplit < 0)
            {
                nodeIndex = first;
            } 
            else if(tsplit <= tmin) 
            {
                nodeIndex = second;
            }
            else
            {
                nodeIndex = first;
                tmax = tsplit;
                pushDown = 0;
            }
            if(pushDown)
            {
                root = nodeIndex;
            }
        }

        uint prims = n.contentCount[nodeIndex];
        for(uint i = 0; i < prims; ++i)
        {
            computeHit(&spheres[n.contentStartIndex[nodeIndex] + i], eye, ray, tmin, tmax, 1, hit);

            if(hit->t < tmax)
            {
                return;
            }
        }
    }
}

__device__ bool testShadow(float3* spheres, float4* light, float4* pos, Node root, float4* mini, float4* maxi, uint treeDepth, float* contri)
{
    HitResult res;
    memset(&res, 0, sizeof(HitResult));
    res.t = FLT_MAX;
    traversal0(spheres, root, pos, &normalize(*light - *pos), &res, mini, maxi, treeDepth);

    *contri = 0.5;
    return res.isHit;
}

texture<float4, cudaTextureType2D, cudaReadModeElementType> src;
texture<float4, cudaTextureType2D, cudaReadModeElementType> worldPosTexture;

__device__ float4 getRefraction(float4* ray, HitResult* res)
{
    float4 rfract = normalize(refract(&normalize(*ray * res->t), &res->normal, 1.0/1.5));

    rfract = rfract * 0.5 + 0.5;
    rfract.y = 1 - rfract.y;

    return tex2D(src, rfract.x, rfract.y);
}

extern "C" __global__ void simpleSphereTracer(
    float4* dst, 
    float3* spheres,
    AABB* aabbs,
    Node* nodes,
    uint treeDepth,
    float* view, 
    float3 _eye,
    uint w, uint h)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;

    if(idx >= w || idy >= h)
    {
        return;
    }
    
    float u = idx / (float)w;
    float v = idy / (float)h;

    float4 eye = make_float4(_eye, 0);

    float4 light = 1000* make_float4(1.0f,0.3f,-0.2f, 0);

    float4 color = tex2D(src, u, v);

    float4 ray = normalize(make_float4(2 * u - 1, 2 * (1-v) - 1, 1.0f, 0));

    ray = transform(view, &ray);
    ray.w = 0;

    float4 mini = make_float4(aabbs[0].min, 0);
    float4 maxi = make_float4(aabbs[0].max, 0);
    mini -= make_float4(1,1,1,0);
    maxi += make_float4(1,1,1,0);

    float4 sphereColor = make_float4(0, 0, 0, 0);
    
    HitResult res;
    memset(&res, 0, sizeof(HitResult));
    res.t = FLT_MAX;

    Node root = nodes[0];

    float shadowContri = 0;

    traversal0(spheres, root, &eye, &ray, &res, &mini, &maxi, treeDepth);
    
    if(!res.isHit)
    {
        /*float4 wp = tex2D(worldPosTexture, u, v);
        if(wp.w > 0)
        {
            wp.w = 0;
            if(testShadow(spheres, &light, &wp, root, &mini, &maxi, treeDepth, &shadowContri))
            {
                color *= 0.5;
            }
        }*/
        dst[id] = color;
        return;
    }

#if 0
    float4 c;
    c = color * 0;

#else 
    float4 reflectionColor = make_float4(0,0,0,0);

    HitResult refRes;
    memset(&refRes, 0, sizeof(HitResult));
    refRes.t = FLT_MAX;

    traversal0(spheres, root, &(res.worldPosition + res.normal * 0.1), &res.normal, &refRes, &mini, &maxi, treeDepth);

    float4 refractionColor = getRefraction(&ray, &res);

    if(refRes.isHit)
    {
        reflectionColor = sphereColor + 0.5 * getRefraction(&refRes.normal, &refRes);
        if(testShadow(spheres, &light, &(refRes.worldPosition + 0.1 * refRes.normal), root, &mini, &maxi, treeDepth, &shadowContri))
        {
            reflectionColor *= shadowContri;
        }
    }

    float4 lightToPos = normalize(res.worldPosition - light);

    float3 reflectVec = reflect(make_float3(lightToPos), make_float3(normalize(res.normal)));

    float diffuse = 0.8 + 0.2 * max(0.0, dot(-lightToPos, normalize(res.normal)));

    float specular = pow(max(0.0, dot(make_float4(reflectVec), normalize(eye - res.worldPosition))), 32.0);

    c = diffuse * (0.4*sphereColor + refractionColor + reflectionColor) + 2 * specular * refractionColor;

    if(res.isHit)
    {
        if(testShadow(spheres, &light, &(res.worldPosition + 0.1 * res.normal), root, &mini, &maxi, treeDepth, &shadowContri))
        {
            c *= shadowContri;
        }
    }
#endif
    dst[id] = make_float4(c.x, c.y, c.z, color.w);
}

extern "C" void runTracer(
    cudaArray_t srcPtr, cudaArray_t worldPosptr, uint width, uint height, void* linearMem, float3* kdData, AABB* kdBBox, Node* nodes, uint depth, float* view, float3 eye,
    cudaStream_t stream)
{
    

}

