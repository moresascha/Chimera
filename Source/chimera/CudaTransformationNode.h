#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "cudah.h"
#include "Material.h"
#include "Spline.h"

namespace d3d
{
    class d3d::Geometry;
}

namespace cudah
{
    cudah;
}

typedef VOID (*CudaFuncCallBack)(cudah::cuda_buffer buffer, cudah::cuda_buffer staticNormals, cudah::cuda_buffer staticPositions, cudah::cuda_buffer indices, UINT gws, UINT lws, ULONG time, cudah::cuda_stream stream);

typedef d3d::Geometry* (*GeometryCreatorCallBack)(VOID);

namespace tbd
{
    class TransformCudaHandle;
    class CudaTransformationNode : public SceneNode
    {
    protected:
        CudaFuncCallBack m_fpFunc;
        GeometryCreatorCallBack m_fpGeoCreator;
        cudah::cuda_stream m_pStream;
        tbd::Material m_material;
        std::shared_ptr<TransformCudaHandle> m_pHandle;
        std::shared_ptr<d3d::Texture2D> m_pNormalTextureHandle;
        std::shared_ptr<d3d::Texture2D> m_pDiffuseTextureHandle;
        tbd::Resource m_diffTextureRes;
        tbd::Resource m_normalTexRes;
    public:
        CudaTransformationNode(CudaFuncCallBack func, GeometryCreatorCallBack geoCreator);
        virtual VOID VOnUpdate(ULONG millis, SceneGraph* graph);
        virtual VOID VOnRestore(tbd::SceneGraph* graph);
        BOOL VIsVisible(SceneGraph* graph);
        virtual VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);
        tbd::Material& GetMaterial(VOID);
        VOID SetTexture(tbd::Resource res);
        VOID SetNormaleTexture(tbd::Resource res);
        cudah::cuda_buffer GetCudaBuffer(std::string& name);
        cudah::cudah* GetCuda(VOID);
        virtual UINT VGetRenderPaths(VOID);
        virtual ~CudaTransformationNode(VOID);
    };

    class UniformBSplineNode : public CudaTransformationNode
    {
    private:
        util::UniformBSpline m_bspline;
        UINT m_vertexStride;
        UINT m_cntpStride;
        BOOL m_useRawControlPointBuffer;
    public:
        UniformBSplineNode(GeometryCreatorCallBack geoCreator, CudaFuncCallBack func, UINT vertexStride, UINT controlPntsStride, BOOL useRawControlPointBuffer = FALSE);
        util::UniformBSpline& GetSpline(VOID);
        VOID VOnRestore(tbd::SceneGraph* graph);
        VOID VOnUpdate(ULONG millis, SceneGraph* graph);
        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);
        UINT VGetRenderPaths(VOID);
        virtual ~UniformBSplineNode(VOID) {}

        static BOOL drawCP_CP; //tmp
    };
}


