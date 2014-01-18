#include "PointLightNode.h"

namespace chimera 
{
    std::shared_ptr<IVRamHandle> PointlightNode::m_cubeMapHandle = NULL;

    IShaderProgram* PointlightNode::m_drawShadow = NULL;

    IShaderProgram* PointlightNode::m_drawShadowInstanced = NULL;

    IShaderProgram* PointlightNode::m_drawLighting = NULL;

    class CubeMapHandle : public VRamHandle
    {
    public:
        IRenderTarget* m_cubeMap;

        CubeMapHandle(void) : m_cubeMap(NULL)
        {
            VSetResource(CMResource("PointLightShadowMap"));
        }

        bool VCreate(void) 
        {
            if(!m_cubeMap)
            {
                m_cubeMap = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget().release();
            }

            uint shadowMapSize = CmGetApp()->VGetConfig()->VGetInteger("iPointLightSMSize");

            if(!m_cubeMap->VOnRestore(shadowMapSize, shadowMapSize, eFormat_R32_FLOAT, true, true, 6))
            {
                LOG_CRITICAL_ERROR("Failed to create render target");
                return false;
            }

            return true;
        }

        void VDestroy(void)
        {
            SAFE_DELETE(m_cubeMap);
        }

        uint VGetByteCount(void) const
        {
             uint size = CmGetApp()->VGetConfig()->VGetInteger("iPointLightSMSize");
             return size * size * sizeof(float);
        }
    };

    PointlightNode::PointlightNode(ActorId actorid) : SceneNode(actorid), m_alphaBlendingState(NULL), m_depthState(NULL)
    {
        //m_cubeMapRenderTarget = NULL;
        IActor* actor = CmGetApp()->VGetLogic()->VFindActor(actorid);
        m_actor->VQueryComponent(CM_CMP_LIGHT, (IActorComponent**)&m_lightComponent);

        if(!m_drawShadow)
        {
            CMShaderProgramDescription desc;
            ZeroMemory(&desc, sizeof(CMShaderProgramDescription));

            desc.fs.file = L"PointLightShadowMap.hlsl";
            desc.fs.function = "RenderCubeMap_PS";

            desc.vs.file = L"PointLightShadowMap.hlsl";
            desc.vs.function = "RenderCubeMap_VS";
            desc.vs.layoutCount = 3;
            
            desc.gs.file = L"PointLightShadowMap.hlsl";
            desc.gs.function = "RenderCubeMap_GS";

            desc.vs.inputLayout[0].instanced = false;
            desc.vs.inputLayout[0].name = "POSITION";
            desc.vs.inputLayout[0].position = 0;
            desc.vs.inputLayout[0].slot = 0;
            desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

            desc.vs.inputLayout[1].instanced = false;
            desc.vs.inputLayout[1].name = "NORMAL";
            desc.vs.inputLayout[1].position = 1;
            desc.vs.inputLayout[1].slot = 0;
            desc.vs.inputLayout[1].format = eFormat_R32G32B32_FLOAT;

            desc.vs.inputLayout[2].instanced = false;
            desc.vs.inputLayout[2].name = "TEXCOORD";
            desc.vs.inputLayout[2].position = 2;
            desc.vs.inputLayout[2].slot = 0;
            desc.vs.inputLayout[2].format = eFormat_R32G32_FLOAT;
            CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("PointLightShadowMap", &desc);

            ZeroMemory(&desc, sizeof(CMShaderProgramDescription));
            desc.fs.file = L"PointLightShadowMap.hlsl";
            desc.fs.function = "RenderCubeMap_PS";

            desc.vs.file = L"PointLightShadowMap.hlsl";
            desc.vs.function = "RenderCubeMapInstanced_VS";
            desc.vs.layoutCount = 4;

            desc.gs.file = L"PointLightShadowMap.hlsl";
            desc.gs.function = "RenderCubeMap_GS";

            desc.vs.inputLayout[0].instanced = false;
            desc.vs.inputLayout[0].name = "POSITION";
            desc.vs.inputLayout[0].position = 0;
            desc.vs.inputLayout[0].slot = 0;
            desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

            desc.vs.inputLayout[1].instanced = false;
            desc.vs.inputLayout[1].name = "NORMAL";
            desc.vs.inputLayout[1].position = 1;
            desc.vs.inputLayout[1].slot = 0;
            desc.vs.inputLayout[1].format = eFormat_R32G32B32_FLOAT;

            desc.vs.inputLayout[2].instanced = false;
            desc.vs.inputLayout[2].name = "TEXCOORD";
            desc.vs.inputLayout[2].position = 2;
            desc.vs.inputLayout[2].slot = 0;
            desc.vs.inputLayout[2].format = eFormat_R32G32_FLOAT;

            desc.vs.inputLayout[3].instanced = true;
            desc.vs.inputLayout[3].name = "TANGENT";
            desc.vs.inputLayout[3].position = 3;
            desc.vs.inputLayout[3].slot = 1;
            desc.vs.inputLayout[3].format = eFormat_R32G32B32_FLOAT;
            CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("PointLightShadowMapInstanced", &desc);

            ZeroMemory(&desc, sizeof(CMShaderProgramDescription));
            desc.fs.file = L"Lighting.hlsl";
            desc.fs.function = "PointLighting_PS";

            desc.vs.file = L"Lighting.hlsl";
            desc.vs.function = "Lighting_VS";
            desc.vs.layoutCount = 2;

            desc.vs.inputLayout[0].instanced = false;
            desc.vs.inputLayout[0].name = "POSITION";
            desc.vs.inputLayout[0].position = 0;
            desc.vs.inputLayout[0].slot = 0;
            desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

            desc.vs.inputLayout[1].instanced = false;
            desc.vs.inputLayout[1].name = "TEXCOORD";
            desc.vs.inputLayout[1].position = 1;
            desc.vs.inputLayout[1].slot = 0;
            desc.vs.inputLayout[1].format = eFormat_R32G32_FLOAT;

            CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("PointLight", &desc);
        }

        m_drawShadow = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VGetShaderProgram("PointLightShadowMap");
        m_drawShadowInstanced = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VGetShaderProgram("PointLightShadowMapInstanced");
        m_drawLighting = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VGetShaderProgram("PointLight");

        VSetRenderPaths(CM_RENDERPATH_LIGHTING);
    }

    void PointlightNode::VOnActorMoved(void)
    {
        SceneNode::VOnActorMoved();
        XMMATRIX mat = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.01f, VGetTransformation()->GetScale().x);
        XMStoreFloat4x4(&m_projection.m_m, mat);
    }

    void PointlightNode::VOnRestore(ISceneGraph* graph)
    {
        util::Vec3& lightPos = util::Vec3();
        m_mats[0] = util::Mat4::createLookAtLH(lightPos, util::Vec3(1,0,0), util::Vec3(0,1,0)); //right

        m_mats[1] = util::Mat4::createLookAtLH(lightPos, util::Vec3(-1,0,0), util::Vec3(0,1,0)); //left

        m_mats[2] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,1,0), util::Vec3(0,0,-1)); //up

        m_mats[3] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,-1,0), util::Vec3(0,0,1)); //down

        m_mats[4] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,0,1), util::Vec3(0,1,0)); //front

        m_mats[5] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,0,-1), util::Vec3(0,1,0)); //back

        XMMATRIX mat = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.01f, VGetTransformation()->GetScale().x);
        XMStoreFloat4x4(&m_projection.m_m, mat);

        std::unique_ptr<IGraphicsStateFactroy> factroy = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateStateFactory();
        SAFE_DELETE(m_alphaBlendingState);
        BlendStateDesc blendDesc;
        ZeroMemory(&blendDesc, sizeof(BlendStateDesc));

        for(int i = 0; i < 8; ++i)
        {
            blendDesc.RenderTarget[i].BlendEnable = true;
            blendDesc.RenderTarget[i].RenderTargetWriteMask = eColorWriteAll;

            blendDesc.RenderTarget[i].BlendOp = eBlendOP_Add;
            blendDesc.RenderTarget[i].BlendOpAlpha = eBlendOP_Add;

            blendDesc.RenderTarget[i].DestBlend = eBlend_One;
            blendDesc.RenderTarget[i].DestBlendAlpha = eBlend_One;

            blendDesc.RenderTarget[i].SrcBlend = eBlend_SrcAlpha;
            blendDesc.RenderTarget[i].SrcBlendAlpha = eBlend_One;  
        }

        m_alphaBlendingState = factroy->VCreateBlendState(&blendDesc);

        SAFE_DELETE(m_depthState);
        DepthStencilStateDesc depthDesc;
        ZeroMemory(&depthDesc, sizeof(DepthStencilStateDesc));
        depthDesc.DepthEnable = false;
        depthDesc.StencilEnable = false;

        m_depthState = factroy->VCreateDepthStencilState(&depthDesc);

        //m_projection = util::Mat4::CreatePerspectiveLH(XM_PIDIV2, 1.0f, 0.01f, m_lightComponent->m_radius);
        /*
        SAFE_DELETE(m_cubeMapRenderTarget);

        m_cubeMapRenderTarget = new d3d::RenderTarget();

        UINT shadowMapSize = 2048; //TODO

        if(!m_cubeMapRenderTarget->Create(shadowMapSize, shadowMapSize, DXGI_FORMAT_R32_FLOAT, TRUE, TRUE, 6))
        {
            LOG_ERROR("Failed to create render target");
        } */
    }

    bool PointlightNode::VIsVisible(ISceneGraph* graph)
    {
        return graph->VGetFrustum()->IsInside(VGetTransformation()->GetTranslation(), VGetTransformation()->GetScale().x);
    }

    void PointlightNode::_VRender(ISceneGraph* graph, RenderPath& path)
    {
        switch(path)
        {
        case CM_RENDERPATH_LIGHTING: 
            {
                if(!m_lightComponent->m_activated)
                {
                    return;
                }

                IRenderTarget* cubeMapRT;
                if(!m_cubeMapHandle || !m_cubeMapHandle->VIsReady())
                {
                    m_cubeMapHandle = std::shared_ptr<CubeMapHandle>(new CubeMapHandle());
                    CmGetApp()->VGetHumanView()->VGetVRamManager()->VAppendAndCreateHandle(m_cubeMapHandle);
                }

                m_cubeMapHandle->VUpdate();

                cubeMapRT = ((CubeMapHandle*)m_cubeMapHandle.get())->m_cubeMap;

                m_drawShadow->VBind();
                IRenderer* renderer = CmGetApp()->VGetHumanView()->VGetRenderer();
                renderer->VPushProjectionTransform(m_projection, VGetTransformation()->GetScale().x);

                IConstShaderBuffer* lb = renderer->VGetConstShaderBuffer(eCubeMapViewsBuffer);
                XMFLOAT4X4* buffer = (XMFLOAT4X4*)lb->VMap();
                for(int i = 0; i < 6; ++i)
                {
                    buffer[i] = m_mats[i].m_m;
                }
                lb->VUnmap();

                renderer->VSetLightSettings(m_lightComponent->m_color, VGetTransformation()->GetTranslation(), util::Vec3(), VGetTransformation()->GetScale().x, 0, m_lightComponent->m_intensity, 1);

                renderer->VSetTexture(ePointLightShadowCubeMapSampler, NULL);

                cubeMapRT->VBind();
                cubeMapRT->VClear();
                m_frustum.SetParams(VGetTransformation()->GetScale().x, VGetTransformation()->GetTranslation());
                graph->VPushFrustum(&m_frustum);
               
                graph->VOnRender(CM_RENDERPATH_SHADOWMAP);

                m_drawShadowInstanced->VBind();

                graph->VOnRender(CM_RENDERPATH_SHADOWMAP_INSTANCED);

                graph->VPopFrustum();
                renderer->VPopProjectionTransform();

                //renderer->VActivateCurrentRendertarget();
                renderer->VGetCurrentRenderTarget()->VBind();

                //renderer->VSetPointLightShadowCubeMapSampler(cubeMapRT->VGetTexture());

                renderer->VSetTexture(ePointLightShadowCubeMapSampler, cubeMapRT->VGetTexture());
            
                m_drawLighting->VBind();

                renderer->VPushBlendState(m_alphaBlendingState);
                renderer->VPushDepthStencilState(m_depthState);
                renderer->VDrawScreenQuad();
                renderer->VPopBlendState();
                renderer->VPopDepthStencilState();

            } break;
/*
        case eRenderPath_DrawBounding :
            {
                chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eBoundingGeoBuffer);
                XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
                f->x = 1;//m_transformation->GetTransformation()->GetScale().x;
                f->y = 0;
                f->z = 0;
                f->w = 0;
                buffer->Unmap();
                GeometryFactory::GetSphere()->Bind();
                GeometryFactory::GetSphere()->Draw();
            } break;

        case eRenderPath_DrawPicking : 
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eRenderPath_DrawEditMode :
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawAnchorSphere(m_actor, &m, 1);
            } break;

        case eRenderPath_DrawDebugInfo : 
            {            
                std::stringstream ss;
                ss << "PointLight_";
                ss << m_actorId;
                DrawInfoTextOnScreen(graph->GetCamera().get(), GetTransformation(), ss.str());
                break;
            }*/
        }
    }

    PointlightNode::~PointlightNode(void)
    {
        SAFE_DELETE(m_alphaBlendingState);
        SAFE_DELETE(m_depthState);
    }


    //////////////////////////////////////////////////////////////////////////




};