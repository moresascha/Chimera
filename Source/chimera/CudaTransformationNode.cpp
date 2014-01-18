#include "CudaTransformationNode.h"
#include "Geometry.h"
#include "GameApp.h"
#include "Mesh.h"
#include "GeometryFactory.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "Components.h"
#include "Mat4.h"
#include "Vec3.h"
#include "Vec4.h"
#include "SceneGraph.h"
#include "Frustum.h"
#include "Transformation.cuh"

namespace chimera
{
    bool UniformBSplineNode::drawCP_CP = false;

    class TransformCudaHandle : public VRamHandle
    {
        friend class CudaTransformationNode;
        friend class UniformBSplineNode;
    protected:
        chimera::Geometry* m_pGeo;
        cudah::cudah* m_pCuda;
        cudah::cuda_buffer m_d3dbuffer;
        cudah::cuda_buffer m_normals;
        cudah::cuda_buffer m_positions;
        cudah::cuda_buffer m_indices;
        GeometryCreatorCallBack m_fpCreator;
        util::AxisAlignedBB m_aabb;

    public:
        TransformCudaHandle(void) : m_pCuda(NULL), m_pGeo(NULL)
        {

        }

        virtual bool VCreate(void)
        {
            SAFE_DELETE(m_pCuda);

            SAFE_DELETE(m_pGeo);

            m_pCuda = new cudah::cudah();

            m_pGeo = m_fpCreator();

            m_d3dbuffer = m_pCuda->RegisterD3D11Buffer(std::string("vertices"), m_pGeo->GetVertexBuffer()->GetBuffer(), cudaGraphicsMapFlagsNone);

            uint elementCnt = m_pGeo->GetVertexBuffer()->GetElementCount();
        
            float3* normals = new float3[elementCnt];
            float3* pos = new float3[elementCnt];

            const VertexData* vertices = (VertexData*)m_pGeo->GetVertexBuffer()->GetRawData();

            TBD_FOR_INT(elementCnt)
            {
                uint stride = 8;
                const VertexData& d = vertices[i];
                pos[i] = d.position;
                util::Vec3 v(d.position.x, d.position.y, d.position.z);
                m_aabb.AddPoint(v);
                normals[i] = d.normal;
            }
            m_aabb.Construct();
            m_normals = m_pCuda->CreateBuffer(std::string("normals"), elementCnt * sizeof(float3), normals, sizeof(float3));
            m_positions = m_pCuda->CreateBuffer(std::string("positions"), elementCnt * sizeof(float3), pos, sizeof(float3));
            m_indices = m_pCuda->CreateBuffer(std::string("indices"), elementCnt * sizeof(int), sizeof(int));

            //computeIndices((int*)m_indices->ptr,  )

            m_pGeo->DeleteRawData();

            SAFE_DELETE(normals);
            SAFE_DELETE(pos);

            return true;
        }

        virtual void VDestroy()
        {
            SAFE_DELETE(m_pCuda);
            if(m_pGeo)
            {
                m_pGeo->VDestroy();
            }
            SAFE_DELETE(m_pGeo);
        }

        virtual uint VGetByteCount(void) const
        {
            return m_pGeo->VGetByteCount() + m_normals->VGetByteCount() + m_positions->VGetByteCount() + m_indices->VGetByteCount();
        }
    };

    class BSplinesTransformHandle : public TransformCudaHandle
    {
        friend class UniformBSplineNode;
    private:
        chimera::Geometry* m_pControlGeo;
        cudah::cuda_buffer m_controlPoints;
        float* m_pControlPoints;
        uint m_controlPointsCnt;
        bool m_useRawControlPointBuffer;

        BSplinesTransformHandle(bool useRawControlPointBuffer = false) : m_pControlPoints(NULL), m_controlPointsCnt(0), m_pControlGeo(NULL), m_useRawControlPointBuffer(useRawControlPointBuffer)
        {

        }

        bool VCreate(void)
        {
            TransformCudaHandle::VCreate();

            SAFE_DELETE(m_pControlGeo);

            //m_pControlGeo = new d3d::Geometry(TRUE);

            m_pControlGeo = GeometryFactory::CreateSphere(16, 8);
            m_pControlGeo->SetInstanceBuffer(m_pControlPoints, m_controlPointsCnt, sizeof(float3));
            m_pControlGeo->VCreate();

            if(m_useRawControlPointBuffer)
            {
                m_controlPoints = m_pCuda->CreateBuffer(std::string("controlPoints"), m_controlPointsCnt * sizeof(float3), m_pControlPoints, sizeof(float3));
            }
            else
            {
                m_controlPoints = m_pCuda->RegisterD3D11Buffer(std::string("controlPoints"), m_pControlGeo->GetInstanceBuffer()->GetBuffer(), cudaGraphicsMapFlagsNone);
            }

            SAFE_ARRAY_DELETE(m_pControlPoints);

            return true;
        }

        uint VGetByteCount(void) const
        {
            return TransformCudaHandle::VGetByteCount() + m_pControlGeo->VGetByteCount();
        }

        void VDestroy(void)
        {
            TransformCudaHandle::VDestroy();
            if(m_pControlGeo)
            {
                m_pControlGeo->VDestroy();
            }
            SAFE_DELETE(m_pControlGeo);
        }
    };

    CudaTransformationNode::CudaTransformationNode(CudaFuncCallBack func, GeometryCreatorCallBack creator) :
    m_fpFunc(func), m_fpGeoCreator(creator), m_diffTextureRes("7992-D.jpg"), m_normalTexRes("normal/rocktutn.jpg"), m_pNormalTextureHandle(NULL)
    {
        m_material.m_ambient = util::Vec4(0.5,0.5,0.5,0);
        m_material.m_diffuse = util::Vec4(1,1,1,0);
        m_material.m_specular = util::Vec4(1,1,1,0);
        m_material.m_texScale = 16;
        m_material.m_reflectance = 1;
    }

    void CudaTransformationNode::VOnUpdate(ulong millis, SceneGraph* graph)
    {
        if(VIsVisible(graph) && m_pHandle->VIsReady())
        {
            m_pHandle->m_pCuda->MapGraphicsResource(m_pHandle->m_d3dbuffer);
            m_fpFunc(m_pHandle->m_d3dbuffer, m_pHandle->m_normals, m_pHandle->m_positions, m_pHandle->m_indices, 
                m_pHandle->m_pGeo->GetVertexBuffer()->GetElementCount(), 128, chimera::g_pApp->GetUpdateTimer()->GetTime(), m_pStream);
            m_pHandle->m_pCuda->UnmapGraphicsResource(m_pHandle->m_d3dbuffer);
        }        
    }

    cudah::cuda_buffer CudaTransformationNode::GetCudaBuffer(std::string& name)
    {
        return m_pHandle->m_pCuda->GetBuffer(name);
    }

    cudah::cudah* CudaTransformationNode::GetCuda(void)
    {
        return m_pHandle->m_pCuda;
    }

    bool CudaTransformationNode::VIsVisible(chimera::SceneGraph* graph)
    {
        util::Vec3 middle = util::Mat4::Transform(*GetTransformation(), m_aabb.GetMiddle());
        bool in = graph->GetFrustum()->IsInside(middle, m_aabb.GetRadius());
        return in;
    }

    uint CudaTransformationNode::VGetRenderPaths(void)
    {
        return eRenderPath_DrawToAlbedo | eRenderPath_DrawToShadowMap | eRenderPath_DrawBounding;
    }

    void CudaTransformationNode::_VRender(chimera::SceneGraph* graph, chimera::RenderPath& path)
    {
        if(m_pHandle->VIsReady())
        {
            m_pHandle->Update();
            m_pDiffuseTextureHandle->Update();
            if(m_pNormalTextureHandle)
            {
                m_pNormalTextureHandle->Update();
            }
            switch(path)
            {
            case eRenderPath_DrawToShadowMap: 
                {
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                    m_pHandle->m_pGeo->Bind();
                    m_pHandle->m_pGeo->Draw();
                } break;
            case eRenderPath_DrawToAlbedo :
                {
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                    chimera::g_pApp->GetRenderer()->VPushMaterial(m_material);
                    chimera::g_pApp->GetHumanView()->GetRenderer()->SetDiffuseSampler(m_pDiffuseTextureHandle->GetShaderResourceView());

                    if(m_pNormalTextureHandle)
                    {
                        chimera::g_pApp->GetHumanView()->GetRenderer()->SetSampler(chimera::eNormalColorSampler, m_pNormalTextureHandle->GetShaderResourceView());
                        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(true);
                    }
                    else
                    {
                        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(false);
                    }
                    m_pHandle->m_pGeo->Bind();
                    m_pHandle->m_pGeo->Draw();
                } break;
            case eRenderPath_DrawBounding :
                {
                DrawSphere(GetTransformation(), m_aabb);
                } break;
            case eRenderPath_DrawPicking :
                {
                    //DrawPicking(m_actor, m_transformation->GetTransformation(), m_mesh, m_geo);
                } break;
            case eRenderPath_DrawDebugInfo : 
                {
                    chimera::DrawActorInfos(m_actor, GetTransformation(), graph->GetCamera());
                } break;
            }
        }
        else
        {
            VOnRestore(graph);
        }
    }

    void CudaTransformationNode::VOnRestore(chimera::SceneGraph* graph)
    {
        std::stringstream ss;
        ss << "CudaTransformationNode";
        ss << m_actorId;
        m_pHandle = std::shared_ptr<TransformCudaHandle>(new TransformCudaHandle());
        m_pHandle->m_fpCreator = m_fpGeoCreator;
        chimera::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pHandle);

        m_pStream = m_pHandle->m_pCuda->CreateStream();

        /*m_pNormalTextureHandle = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle("normal/rocktutn.jpg"));

        m_pDiffuseTextureHandle = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle("7992-D.jpg"));
        */

        SetTexture(m_diffTextureRes);

        SetNormaleTexture(m_normalTexRes);

        util::AxisAlignedBB aabb;
        m_aabb = m_pHandle->m_aabb;
        /*m_aabb.AddPoint(util::Vec3(-2,-2,-2));
        m_aabb.AddPoint(util::Vec3(2,2,2));
        m_aabb.Construct(); */
    }

    chimera::Material& CudaTransformationNode::GetMaterial(void)
    {
        return m_material;
    }

    void CudaTransformationNode::SetTexture(chimera::CMResource res)
    {
        m_diffTextureRes = res;
        m_pDiffuseTextureHandle = std::static_pointer_cast<chimera::D3DTexture2D>(chimera::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_diffTextureRes));
    }

    void CudaTransformationNode::SetNormaleTexture(chimera::CMResource res)
    {
        m_normalTexRes = res;
        m_pNormalTextureHandle = std::static_pointer_cast<chimera::D3DTexture2D>(chimera::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_normalTexRes));
    }

    CudaTransformationNode::~CudaTransformationNode(void)
    {

    }

    //bsplines

    UniformBSplineNode::UniformBSplineNode(GeometryCreatorCallBack geoCreator, CudaFuncCallBack func, uint vertexStride, uint controlPntsStride, bool b) : CudaTransformationNode(func, geoCreator), 
        m_vertexStride(vertexStride), m_cntpStride(controlPntsStride), m_useRawControlPointBuffer(b)
    {

    }

    util::UniformBSpline& UniformBSplineNode::GetSpline(void)
    {
        return m_bspline;
    }

    void UniformBSplineNode::VOnRestore(chimera::SceneGraph* graph)
    {
        std::stringstream ss;
        ss << "UniformBSplineNode";
        ss << m_actorId;
        BSplinesTransformHandle* handle = new BSplinesTransformHandle(m_useRawControlPointBuffer);
        m_pHandle = std::shared_ptr<TransformCudaHandle>(handle);
        m_pHandle->m_fpCreator = m_fpGeoCreator;

        SetTexture(m_diffTextureRes);

        SetNormaleTexture(m_normalTexRes);

        uint index = 0;
        float* controlPoints = new float[3 * m_bspline.GetControlPoints().size()];
        TBD_FOR(m_bspline.GetControlPoints())
        {
            controlPoints[index++] = it->x;
            controlPoints[index++] = it->y;
            controlPoints[index++] = it->z;
        }
        handle->m_pControlPoints = controlPoints;
        handle->m_controlPointsCnt = (uint)m_bspline.GetControlPoints().size();

        chimera::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pHandle);

        m_pStream = m_pHandle->m_pCuda->CreateStream();

        util::AxisAlignedBB aabb;
        m_aabb = m_pHandle->m_aabb;
    }

    void UniformBSplineNode::VOnUpdate(ulong millis, SceneGraph* graph)
    {
        if(VIsVisible(graph) && m_pHandle->VIsReady())
        {
            BSplinesTransformHandle* handle = (BSplinesTransformHandle*)m_pHandle.get();

            m_pHandle->m_pCuda->MapGraphicsResource(m_pHandle->m_d3dbuffer);

            if(!m_useRawControlPointBuffer)
            {
                m_pHandle->m_pCuda->MapGraphicsResource(handle->m_controlPoints);
            }

            m_fpFunc(m_pHandle->m_d3dbuffer, m_pHandle->m_normals, handle->m_controlPoints, m_pHandle->m_indices, 
                m_pHandle->m_pGeo->GetVertexBuffer()->GetElementCount(), 128, chimera::g_pApp->GetUpdateTimer()->GetTime(), m_pStream);

            LOG_CRITICAL_ERROR_A("%s\n", "TODO");
            /*bspline(gws, 128, 
                (VertexData*)handle->m_d3dbuffer->ptr, (float3*)handle->m_controlPoints->ptr,
                m_pHandle->m_pGeo->GetVertexBuffer()->GetElementCount(), (UINT)(m_bspline.GetControlPoints().size()), m_cntpStride, m_vertexStride, m_pStream->GetPtr()); */

            if(!m_useRawControlPointBuffer)
            {
                m_pHandle->m_pCuda->UnmapGraphicsResource(handle->m_controlPoints);
            }

            m_pHandle->m_pCuda->UnmapGraphicsResource(m_pHandle->m_d3dbuffer);
        }   
    }

    uint UniformBSplineNode::VGetRenderPaths(void)
    {
        return eRenderPath_DrawToAlbedoInstanced | CudaTransformationNode::VGetRenderPaths();
    }

    void UniformBSplineNode::_VRender(chimera::SceneGraph* graph, chimera::RenderPath& path)
    {
        CudaTransformationNode::_VRender(graph, path);
        if(drawCP_CP)
        switch(path)
        {
        case eRenderPath_DrawToAlbedoInstanced:
        {
            util::Mat4 mat = *GetTransformation();
            mat.Scale(0.05f);
            mat.Translate(1,2,1);
            chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(mat);
            chimera::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();
            ((BSplinesTransformHandle*)m_pHandle.get())->m_pControlGeo->Bind();
            ((BSplinesTransformHandle*)m_pHandle.get())->m_pControlGeo->Draw();
        } break;

        }
    }
}