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

namespace tbd
{
    BOOL UniformBSplineNode::drawCP_CP = FALSE;

    class TransformCudaHandle : public VRamHandle
    {
        friend class CudaTransformationNode;
        friend class UniformBSplineNode;
    protected:
        d3d::Geometry* m_pGeo;
        cudah::cudah* m_pCuda;
        cudah::cuda_buffer m_d3dbuffer;
        cudah::cuda_buffer m_normals;
        cudah::cuda_buffer m_positions;
        cudah::cuda_buffer m_indices;
        GeometryCreatorCallBack m_fpCreator;
        util::AxisAlignedBB m_aabb;

    public:
        TransformCudaHandle(VOID) : m_pCuda(NULL), m_pGeo(NULL)
        {

        }

        virtual BOOL VCreate(VOID)
        {
            SAFE_DELETE(m_pCuda);

            SAFE_DELETE(m_pGeo);

            m_pCuda = new cudah::cudah();

            m_pGeo = m_fpCreator();

            m_d3dbuffer = m_pCuda->RegisterD3D11Buffer(std::string("vertices"), m_pGeo->GetVertexBuffer()->GetBuffer(), cudaGraphicsMapFlagsNone);

            UINT elementCnt = m_pGeo->GetVertexBuffer()->GetElementCount();
        
            float3* normals = new float3[elementCnt];
            float3* pos = new float3[elementCnt];

            CONST VertexData* vertices = (VertexData*)m_pGeo->GetVertexBuffer()->GetRawData();

            TBD_FOR_INT(elementCnt)
            {
                UINT stride = 8;
                CONST VertexData& d = vertices[i];
                pos[i] = d.position;
                util::Vec3 v(d.position.x, d.position.y, d.position.z);
                m_aabb.AddPoint(v);
                normals[i] = d.normal;
            }
            m_aabb.Construct();
            m_normals = m_pCuda->CreateBuffer(std::string("normals"), elementCnt * sizeof(float3), normals, sizeof(float3));
            m_positions = m_pCuda->CreateBuffer(std::string("positions"), elementCnt * sizeof(float3), pos, sizeof(float3));
            m_indices = m_pCuda->CreateBuffer(std::string("indices"), elementCnt * sizeof(INT), sizeof(INT));

            //computeIndices((int*)m_indices->ptr,  )

            m_pGeo->DeleteRawData();

            SAFE_DELETE(normals);
            SAFE_DELETE(pos);

            return TRUE;
        }

        virtual VOID VDestroy()
        {
            SAFE_DELETE(m_pCuda);
            if(m_pGeo)
            {
                m_pGeo->VDestroy();
            }
            SAFE_DELETE(m_pGeo);
        }

        virtual UINT VGetByteCount(VOID) CONST
        {
            return m_pGeo->VGetByteCount() + m_normals->VGetByteCount() + m_positions->VGetByteCount() + m_indices->VGetByteCount();
        }
    };

    class BSplinesTransformHandle : public TransformCudaHandle
    {
        friend class UniformBSplineNode;
    private:
        d3d::Geometry* m_pControlGeo;
        cudah::cuda_buffer m_controlPoints;
        FLOAT* m_pControlPoints;
        UINT m_controlPointsCnt;
        BOOL m_useRawControlPointBuffer;

        BSplinesTransformHandle(BOOL useRawControlPointBuffer = FALSE) : m_pControlPoints(NULL), m_controlPointsCnt(0), m_pControlGeo(NULL), m_useRawControlPointBuffer(useRawControlPointBuffer)
        {

        }

        BOOL VCreate(VOID)
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

            return TRUE;
        }

        UINT VGetByteCount(VOID) CONST
        {
            return TransformCudaHandle::VGetByteCount() + m_pControlGeo->VGetByteCount();
        }

        VOID VDestroy(VOID)
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

    VOID CudaTransformationNode::VOnUpdate(ULONG millis, SceneGraph* graph)
    {
        if(VIsVisible(graph) && m_pHandle->IsReady())
        {
            m_pHandle->m_pCuda->MapGraphicsResource(m_pHandle->m_d3dbuffer);
            m_fpFunc(m_pHandle->m_d3dbuffer, m_pHandle->m_normals, m_pHandle->m_positions, m_pHandle->m_indices, 
                m_pHandle->m_pGeo->GetVertexBuffer()->GetElementCount(), 128, app::g_pApp->GetUpdateTimer()->GetTime(), m_pStream);
            m_pHandle->m_pCuda->UnmapGraphicsResource(m_pHandle->m_d3dbuffer);
        }        
    }

    cudah::cuda_buffer CudaTransformationNode::GetCudaBuffer(std::string& name)
    {
        return m_pHandle->m_pCuda->GetBuffer(name);
    }

    cudah::cudah* CudaTransformationNode::GetCuda(VOID)
    {
        return m_pHandle->m_pCuda;
    }

    BOOL CudaTransformationNode::VIsVisible(tbd::SceneGraph* graph)
    {
        util::Vec3 middle = util::Mat4::Transform(*GetTransformation(), m_aabb.GetMiddle());
        BOOL in = graph->GetFrustum()->IsInside(middle, m_aabb.GetRadius());
        return in;
    }

    UINT CudaTransformationNode::VGetRenderPaths(VOID)
    {
        return eDRAW_TO_ALBEDO | eDRAW_TO_SHADOW_MAP | eDRAW_BOUNDING_DEBUG;
    }

    VOID CudaTransformationNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        if(m_pHandle->IsReady())
        {
            m_pHandle->Update();
            m_pDiffuseTextureHandle->Update();
            if(m_pNormalTextureHandle)
            {
                m_pNormalTextureHandle->Update();
            }
            switch(path)
            {
            case eDRAW_TO_SHADOW_MAP: 
                {
                    app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                    m_pHandle->m_pGeo->Bind();
                    m_pHandle->m_pGeo->Draw();
                } break;
            case eDRAW_TO_ALBEDO :
                {
                    app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                    app::g_pApp->GetRenderer()->VPushMaterial(m_material);
                    app::g_pApp->GetHumanView()->GetRenderer()->SetDiffuseSampler(m_pDiffuseTextureHandle->GetShaderResourceView());

                    if(m_pNormalTextureHandle)
                    {
                        app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eNormalColorSampler, m_pNormalTextureHandle->GetShaderResourceView());
                        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(TRUE);
                    }
                    else
                    {
                        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                    }
                    m_pHandle->m_pGeo->Bind();
                    m_pHandle->m_pGeo->Draw();
                } break;
            case eDRAW_BOUNDING_DEBUG :
                {
                DrawSphere(GetTransformation(), m_aabb);
                } break;
            case eDRAW_PICKING :
                {
                    //DrawPicking(m_actor, m_transformation->GetTransformation(), m_mesh, m_geo);
                } break;
            case eDRAW_DEBUG_INFOS : 
                {
                    tbd::DrawActorInfos(m_actor, GetTransformation(), graph->GetCamera());
                } break;
            }
        }
        else
        {
            VOnRestore(graph);
        }
    }

    VOID CudaTransformationNode::VOnRestore(tbd::SceneGraph* graph)
    {
        std::stringstream ss;
        ss << "CudaTransformationNode";
        ss << m_actorId;
        m_pHandle = std::shared_ptr<TransformCudaHandle>(new TransformCudaHandle());
        m_pHandle->m_fpCreator = m_fpGeoCreator;
        app::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pHandle);

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

    tbd::Material& CudaTransformationNode::GetMaterial(VOID)
    {
        return m_material;
    }

    VOID CudaTransformationNode::SetTexture(tbd::Resource res)
    {
        m_diffTextureRes = res;
        m_pDiffuseTextureHandle = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_diffTextureRes));
    }

    VOID CudaTransformationNode::SetNormaleTexture(tbd::Resource res)
    {
        m_normalTexRes = res;
        m_pNormalTextureHandle = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_normalTexRes));
    }

    CudaTransformationNode::~CudaTransformationNode(VOID)
    {

    }

    //bsplines

    UniformBSplineNode::UniformBSplineNode(GeometryCreatorCallBack geoCreator, CudaFuncCallBack func, UINT vertexStride, UINT controlPntsStride, BOOL b) : CudaTransformationNode(func, geoCreator), 
        m_vertexStride(vertexStride), m_cntpStride(controlPntsStride), m_useRawControlPointBuffer(b)
    {

    }

    util::UniformBSpline& UniformBSplineNode::GetSpline(VOID)
    {
        return m_bspline;
    }

    VOID UniformBSplineNode::VOnRestore(tbd::SceneGraph* graph)
    {
        std::stringstream ss;
        ss << "UniformBSplineNode";
        ss << m_actorId;
        BSplinesTransformHandle* handle = new BSplinesTransformHandle(m_useRawControlPointBuffer);
        m_pHandle = std::shared_ptr<TransformCudaHandle>(handle);
        m_pHandle->m_fpCreator = m_fpGeoCreator;

        SetTexture(m_diffTextureRes);

        SetNormaleTexture(m_normalTexRes);

        UINT index = 0;
        FLOAT* controlPoints = new FLOAT[3 * m_bspline.GetControlPoints().size()];
        TBD_FOR(m_bspline.GetControlPoints())
        {
            controlPoints[index++] = it->x;
            controlPoints[index++] = it->y;
            controlPoints[index++] = it->z;
        }
        handle->m_pControlPoints = controlPoints;
        handle->m_controlPointsCnt = (UINT)m_bspline.GetControlPoints().size();

        app::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pHandle);

        m_pStream = m_pHandle->m_pCuda->CreateStream();

        util::AxisAlignedBB aabb;
        m_aabb = m_pHandle->m_aabb;
    }

    VOID UniformBSplineNode::VOnUpdate(ULONG millis, SceneGraph* graph)
    {
        if(VIsVisible(graph) && m_pHandle->IsReady())
        {
            BSplinesTransformHandle* handle = (BSplinesTransformHandle*)m_pHandle.get();

            m_pHandle->m_pCuda->MapGraphicsResource(m_pHandle->m_d3dbuffer);

            if(!m_useRawControlPointBuffer)
            {
                m_pHandle->m_pCuda->MapGraphicsResource(handle->m_controlPoints);
            }

            m_fpFunc(m_pHandle->m_d3dbuffer, m_pHandle->m_normals, handle->m_controlPoints, m_pHandle->m_indices, 
                m_pHandle->m_pGeo->GetVertexBuffer()->GetElementCount(), 128, app::g_pApp->GetUpdateTimer()->GetTime(), m_pStream);

            UINT gws = cudah::cudah::GetThreadCount(m_pHandle->m_pGeo->GetVertexBuffer()->GetElementCount(), 128);
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

    UINT UniformBSplineNode::VGetRenderPaths(VOID)
    {
        return eDRAW_TO_ALBEDO_INSTANCED | CudaTransformationNode::VGetRenderPaths();
    }

    VOID UniformBSplineNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        CudaTransformationNode::_VRender(graph, path);
        if(drawCP_CP)
        switch(path)
        {
        case eDRAW_TO_ALBEDO_INSTANCED:
        {
            util::Mat4 mat = *GetTransformation();
            mat.Scale(0.05f);
            mat.Translate(1,2,1);
            app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(mat);
            app::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();
            ((BSplinesTransformHandle*)m_pHandle.get())->m_pControlGeo->Bind();
            ((BSplinesTransformHandle*)m_pHandle.get())->m_pControlGeo->Draw();
        } break;

        }
    }
}