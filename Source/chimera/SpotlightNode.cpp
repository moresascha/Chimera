#include "SpotlightNode.h"
#include "SceneGraph.h"
#include "Components.h"
#include "Camera.h"

namespace chimera
{
    std::shared_ptr<IVRamHandle> SpotlightNode::m_pShadowRenderTargetHandle = NULL;

    IShaderProgram* SpotlightNode::m_drawShadow = NULL;

    IShaderProgram* SpotlightNode::m_drawShadowInstanced = NULL;

    IShaderProgram* SpotlightNode::m_drawLighting = NULL;


    class ShadowMapHandle : public VRamHandle
    {
    public:
        IRenderTarget* m_texture;

        ShadowMapHandle(void) : m_texture(NULL)
        {
            VSetResource(CMResource("SpotLightShadowMap"));
        }

        bool VCreate(void) 
        {
            if(!m_texture)
            {
                m_texture = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget().release();
            }

            uint shadowMapSize = CmGetApp()->VGetConfig()->VGetInteger("iSpotLightSMSize");

            if(!m_texture->VOnRestore(shadowMapSize, shadowMapSize, eFormat_R32_FLOAT, true))
            {
                LOG_CRITICAL_ERROR("Failed to create render target");
                return false;
            }

            return true;
        }

        void VDestroy(void)
        {
            SAFE_DELETE(m_texture);
        }

        uint VGetByteCount(void) const
        {
            uint size = CmGetApp()->VGetConfig()->VGetInteger("iSpotLightSMSize");
            return size * size * sizeof(float);
        }
    };

    SpotlightNode::SpotlightNode(ActorId actorid) : SceneNode(actorid), m_distance(0), m_depthState(NULL), m_alphaBlendingState(NULL)
    {
        IActor* actor = CmGetApp()->VGetLogic()->VFindActor(actorid);
        m_actor->VQueryComponent(CM_CMP_LIGHT, (IActorComponent**)&m_lightComponent);

        if(!m_drawShadow)
        {
            CMShaderProgramDescription desc;
            ZeroMemory(&desc, sizeof(CMShaderProgramDescription));

            desc.fs.file = L"SpotLightShadowMap.hlsl";
            desc.fs.function = "SpotLightShadow_PS";

            desc.vs.file = L"SpotLightShadowMap.hlsl";
            desc.vs.function = "SpotLightShadow_VS";
            desc.vs.layoutCount = 3;

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
            CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("SpotLightShadowMap", &desc);

            ZeroMemory(&desc, sizeof(CMShaderProgramDescription));
            desc.fs.file = L"SpotLightShadowMap.hlsl";
            desc.fs.function = "SpotLightShadow_PS";

            desc.vs.file = L"SpotLightShadowMap.hlsl";
            desc.vs.function = "SpotLightShadowInstanced_VS";
            desc.vs.layoutCount = 4;

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
            CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("SpotLightShadowMapInstanced", &desc);

            ZeroMemory(&desc, sizeof(CMShaderProgramDescription));
            desc.fs.file = L"Lighting.hlsl";
            desc.fs.function = "SpotLighting_PS";

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

            CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("SpotLight", &desc);
        }

        m_drawShadow = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VGetShaderProgram("SpotLightShadowMap");
        m_drawShadowInstanced = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VGetShaderProgram("SpotLightShadowMapInstanced");
        m_drawLighting = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VGetShaderProgram("SpotLight");

        uint wh = CmGetApp()->VGetConfig()->VGetInteger("iSpotLightSMSize");

        m_pCamera = new util::Camera(wh, wh, 1e-2f, m_lightComponent->m_radius);
        
        VOnActorMoved();

        VSetRenderPaths(CM_RENDERPATH_LIGHTING | CM_RENDERPATH_EDITOR);
    }

    void SpotlightNode::VOnActorMoved(void)
    {
        SceneNode::VOnActorMoved();
        m_pCamera->SetPerspectiveProjection(1.0f, DEGREE_TO_RAD(m_lightComponent->m_angle), 0.01f, m_lightComponent->m_radius);
        
        util::Vec4 up(0,1,0,0);
        util::Vec4 dir(0,0,1,0);
        up = util::Mat4::Transform(*VGetTransformation(), up);
        dir = util::Mat4::Transform(*VGetTransformation(), dir);
        up.Normalize();
        dir.Normalize();
        m_pCamera->FromViewUp(util::Vec3(dir.x,dir.y,dir.z), util::Vec3(up.x,up.y,up.z));

        m_pCamera->SetEyePos(VGetTransformation()->GetTranslation());

        //m_pCamera->SetRotation(GetTransformation()->GetPYR().y, GetTransformation()->GetPYR().x);
        
        m_middle = VGetTransformation()->GetTranslation();
        float c = cos(m_pCamera->GetFoV() / 2.0f); //Todo: need a tighter bb here
        float h = m_lightComponent->m_radius / c;
        m_distance = h / 2.0f;
        m_middle = m_middle + (m_pCamera->GetViewDir() * m_distance);
    }

    void SpotlightNode::VOnRestore(ISceneGraph* graph)
    {
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

        SceneNode::VOnRestore(graph);
    }

    bool SpotlightNode::VIsVisible(ISceneGraph* graph)
    {
        return graph->VGetFrustum()->IsInside(m_middle, m_distance);
    }

    void SpotlightNode::_VRender(ISceneGraph* graph, RenderPath& path)
    {
        switch(path)
        {
        case CM_RENDERPATH_LIGHTING: 
            {
                if(!m_lightComponent->m_activated)
                {
                    return;
                }

                IRenderer* renderer = CmGetApp()->VGetHumanView()->VGetRenderer();
                renderer->VPushViewTransform(m_pCamera->GetView(), m_pCamera->GetIView(), m_pCamera->GetEyePos());
                renderer->VPushProjectionTransform(m_pCamera->GetProjection(), m_lightComponent->m_radius);

                renderer->VSetLightSettings(m_lightComponent->m_color, VGetTransformation()->GetTranslation(), 
                    m_pCamera->GetViewDir(),
                    m_lightComponent->m_radius, DEGREE_TO_RAD(m_lightComponent->m_angle), m_lightComponent->m_intensity, m_lightComponent->m_castShadow);

                if(m_lightComponent->m_castShadow)
                {
                    m_drawShadow->VBind();

                    IRenderTarget* shadowRT;

                    if(!m_pShadowRenderTargetHandle || !m_pShadowRenderTargetHandle->VIsReady())
                    {
                        m_pShadowRenderTargetHandle = std::shared_ptr<ShadowMapHandle>(new ShadowMapHandle());
                        CmGetApp()->VGetHumanView()->VGetVRamManager()->VAppendAndCreateHandle(m_pShadowRenderTargetHandle);
                    }

                    m_pShadowRenderTargetHandle->VUpdate();

                    shadowRT = ((ShadowMapHandle*)m_pShadowRenderTargetHandle.get())->m_texture;

                    shadowRT->VBind();
                    shadowRT->VClear();

                    graph->VPushFrustum(&m_pCamera->GetFrustum());

                    graph->VOnRender(CM_RENDERPATH_SHADOWMAP);

                    m_drawShadowInstanced->VBind();

                    graph->VOnRender(CM_RENDERPATH_SHADOWMAP_INSTANCED);

                    graph->VPopFrustum();

                    renderer->VGetCurrentRenderTarget()->VBind();

                    renderer->VSetTexture(chimera::eDiffuseColorSampler, shadowRT->VGetTexture());
                }

                if(m_lightComponent->m_projTexture.c_str())
                {
                    if(!m_projectedTextureHandle || !m_projectedTextureHandle->VIsReady())
                    {
                        m_projectedTextureHandle = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_lightComponent->m_projTexture));
                    }
                    renderer->VSetTexture(eNormalColorSampler, m_projectedTextureHandle.get());
                }

                m_drawLighting->VBind();

                renderer->VPushBlendState(m_alphaBlendingState);
                renderer->VPushDepthStencilState(m_depthState);
                renderer->VDrawScreenQuad();
                renderer->VPopBlendState();
                renderer->VPopDepthStencilState();

                renderer->VPopViewTransform();
                renderer->VPopProjectionTransform();

                renderer->VSetTexture(chimera::eDiffuseColorSampler, NULL); //todo

            } break;
            /*
        case eRenderPath_DrawBounding :
            {
                util::Mat4 t;
                t.SetTranslate(m_middle.x, m_middle.y, m_middle.z);
                DrawSphere(&t, m_distance);
            } break;

        case eRenderPath_DrawPicking : 
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eRenderPath_DrawEditMode :
            {
                if(!HasParent())
                {
                    util::Mat4 m = *GetTransformation();
                    m.SetScale(1);
                    DrawAnchorSphere(m_actor, &m, 1); 
                   // DrawFrustum(m_pCamera->GetFrustum());
                }
            } break;

        case eRenderPath_DrawDebugInfo : 
            {            
                std::stringstream ss;
                ss << "SpotLight_";
                ss << m_actorId;
                DrawInfoTextOnScreen(graph->GetCamera().get(), GetTransformation(), ss.str());
                break;
            }*/
        }
    }

    SpotlightNode::~SpotlightNode(void)
    {
        SAFE_DELETE(m_pCamera);
        SAFE_DELETE(m_alphaBlendingState);
        SAFE_DELETE(m_depthState);
    }
}