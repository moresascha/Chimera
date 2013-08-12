#include "GraphicsSettings.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "GeometryFactory.h"
#include "GameLogic.h"
#include "PointLightNode.h"
#include "d3d.h"
#include "vector_types.h"
#include "Effect.h"
#include "SceneGraph.h"
#include "CascadedShadowMapper.h"
#include "Picker.h"
#include "tbdFont.h"
#include "ShaderProgram.h"
#include "Profiling.h"
#include "GuiComponent.h"
#include "util.h"

namespace tbd
{  
    VOID DrawAlbedo(d3d::ShaderProgram* prog, d3d::ShaderProgram* instancedProg)
    {
        instancedProg->Bind();
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_TO_ALBEDO_INSTANCED);

        prog->Bind();
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_TO_ALBEDO);
    }

    class LuminanceParameter : public d3d::IEffectParmaters
    {
    private:
        FLOAT m_scale;
    public:
        LuminanceParameter(FLOAT scale) : m_scale(scale) {}

        VOID VApply(VOID)
        {
            d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eGuiColorBuffer);
            float4* f4 = (float4*)buffer->Map();
            f4->w = 1;
            f4->x = f4->y = f4->z = m_scale;
            buffer->Unmap();
        }
    };

    ShaderPathSetting::ShaderPathSetting(UINT path, LPCSTR progName, LPCSTR settingName) : IGraphicSetting(settingName), m_renderPath(path), m_progName(progName), m_pProgram(NULL)
    {

    }

    BOOL ShaderPathSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pProgram)
        {
            m_pProgram = d3d::ShaderProgram::GetProgram(m_progName.c_str()).get();
        }
        return TRUE;
    }

    VOID ShaderPathSetting::VRender(VOID)
    {
        m_pProgram->Bind();
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(m_renderPath);
    }

    GloablLightingSetting::GloablLightingSetting(VOID) : IGraphicSetting("GlobalLighting"), m_pGlobalLightProgram(NULL)
    {

    }

    BOOL GloablLightingSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pGlobalLightProgram) //if this is 0 everything else is 0
        {
            m_pGlobalLightProgram = d3d::ShaderProgram::GetProgram("GlobalLighting").get();
        }
        return TRUE;
    }

    VOID GloablLightingSetting::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->GetCurrentrenderTarget()->Clear();
        app::g_pApp->GetHumanView()->GetRenderer()->GetCurrentrenderTarget()->Bind();

        app::g_pApp->GetHumanView()->GetRenderer()->VPreRender();

        m_pGlobalLightProgram->Bind();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
        GeometryFactory::GetGlobalScreenQuad()->Bind();
        GeometryFactory::GetGlobalScreenQuad()->Draw();
    }

    BOOL CSMSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pCSM)
        {
            m_pCSM = new d3d::CascadedShadowMapper(3);
        }
        m_pCSM->OnRestore();
        return TRUE;
    }

    VOID CSMSetting::VRender(VOID)
    {
        m_pCSM->Render(app::g_pApp->GetHumanView()->GetSceneGraph());
    }

    CSMSetting::~CSMSetting(VOID)
    {
        SAFE_DELETE(m_pCSM);
    }

    VOID LightingSetting::VRender(VOID)
    {
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_LIGHTING);
    }
    
    VOID PostFXSetting::SetPreResult(d3d::RenderTarget* pPreResult)
    {
        m_pPreResult = pPreResult;
    }

    VOID PostFXSetting::SetScene(d3d::RenderTarget* pScene)
    {
        m_pScene = pScene;
    }

    BOOL PostFXSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pEffectChain)
        {
            m_pEffectChain = new d3d::EffectChain(m_pPreResult, w, h);

            d3d::Effect* lumi = m_pEffectChain->CreateEffect("Luminance");
            lumi->SetSource(m_pPreResult);

            //lumi->SetParameters(std::shared_ptr<LuminanceParameter>(new LuminanceParameter(1.0f)));

            d3d::Effect* e = NULL;

            for(INT i = w / 2; i > 1; i = i >> 1)
            {
                FLOAT s = (FLOAT)i / (FLOAT)w;
                if(e == NULL)
                {
                    d3d::Effect* ds = m_pEffectChain->CreateEffect("Sample", s, s);
                    ds->AddRequirement(lumi);
                    e = ds;
                }
                else
                {
                    d3d::Effect* ds = m_pEffectChain->CreateEffect("Sample", s, s);
                    ds->AddRequirement(e);
                    e = ds;
                }
            }

            d3d::Effect* ds = m_pEffectChain->CreateEffect("Sample", 1.0f / w, 1.0f / h);
            ds->AddRequirement(e);
            e = ds;

            FLOAT brightPathSize = 0.25f;

            d3d::Effect* bright = m_pEffectChain->CreateEffect("Brightness", brightPathSize, brightPathSize);
            bright->SetSource(m_pPreResult);

            d3d::Effect* e0 = m_pEffectChain->CreateEffect("BlurH", brightPathSize, brightPathSize);
            e0->AddRequirement(bright);

            d3d::Effect* e1 = m_pEffectChain->CreateEffect("BlurV", brightPathSize, brightPathSize);
            e1->AddRequirement(e0);

            d3d::Effect* e2 = m_pEffectChain->CreateEffect("ToneMap");
            e2->SetSource(m_pPreResult);
            e2->AddRequirement(e1);
            e2->AddRequirement(e);
            e2->SetTarget(m_pScene);
        }
        else
        {
            m_pEffectChain->OnRestore(w, h);
        }
        return TRUE;
    }

    VOID PostFXSetting::VRender(VOID)
    {
        m_pEffectChain->Process();
    }

    PostFXSetting::~PostFXSetting(VOID)
    {
        SAFE_DELETE(m_pEffectChain);
    }

    BoundingGeoSetting::BoundingGeoSetting(VOID) : ShaderPathSetting(eDRAW_BOUNDING_DEBUG, "BoundingGeo", "bgeo")
    {

    }

    VOID BoundingGeoSetting::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);
        ShaderPathSetting::VRender();
        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    }

    ProfileSetting::ProfileSetting(IGraphicSetting* setting) : IGraphicSetting("profile"), m_pSetting(setting)
    {

    }

    ProfileSetting& operator<<(ProfileSetting& settings, tbd::Query* q)
    {
        settings.m_pQuerys.push_back(q);
        return settings;
    }

    LPCSTR ProfileSetting::GetText()
    {
        m_resultAsString = m_pSetting->GetName();
        m_resultAsString += "\n";
        TBD_FOR(m_pQuerys)
        {
            std::string t;
            (*it)->GetResultAsString(t);
            m_resultAsString += (*it)->GetInfo();
            m_resultAsString += ": ";
            m_resultAsString += t;
            m_resultAsString += "\n";
        }
        return m_resultAsString.c_str();
    }

    BOOL ProfileSetting::VOnRestore(UINT w, UINT h)
    {
        return m_pSetting->VOnRestore(w, h);
    }

    VOID ProfileSetting::VRender(VOID)
    {
        TBD_FOR(m_pQuerys)
        {
            (*it)->VStart();
        }
        m_pSetting->VRender();
        TBD_FOR(m_pQuerys)
        {
            (*it)->VEnd();
        }
    }

    ProfileSetting::~ProfileSetting(VOID)
    {
        SAFE_DELETE(m_pSetting);
        TBD_FOR(m_pQuerys)
        {
            SAFE_DELETE(*it);
        }
    }

    EditModeSetting::EditModeSetting(VOID) : IGraphicSetting("edit")
    {

    }

    VOID EditModeSetting::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_EDIT_MODE);
    }
    
    //Settings...

    IGraphicsSettings::IGraphicsSettings(VOID)
    {

    }

    GraphicsSettings::GraphicsSettings(VOID) : m_pPostFX(NULL), m_pScene(NULL), m_pPreResult(NULL), m_lastW(0), m_lastH(0)
    {

    }

    VOID GraphicsSettings::AddSetting(IGraphicSetting* setting, SettingType type)
    {
        if(type == eAlbedo)
        {
            m_albedoSettings.push_back(setting);
        }
        else if(type == eLighting)
        {
            m_lightSettings.push_back(setting);
        }
    }

    VOID GraphicsSettings::SetPostFX(IGraphicSetting* setting)
    {
        m_pPostFX = setting;

        if(!m_pPreResult)
        {
            m_pPreResult = new d3d::RenderTarget();
        }
    }

    GraphicsSettings::~GraphicsSettings(VOID)
    {
        SAFE_DELETE(m_pPostFX);

        TBD_FOR(m_lightSettings)
        {
            SAFE_DELETE(*it);
        }

        TBD_FOR(m_albedoSettings)
        {
            SAFE_DELETE(*it);
        }

        SAFE_DELETE(m_pScene);
        SAFE_DELETE(m_pPreResult);
    }

    d3d::RenderTarget* GraphicsSettings::VGetResult(VOID)
    {
        return m_pScene;
    }

    BOOL GraphicsSettings::VOnRestore(UINT w, UINT h)
    {
        m_lastW = w;
        m_lastH = h;
        if(!m_pScene)
        {
            m_pScene = new d3d::RenderTarget();
        }

        if(m_pPreResult)
        {
            m_pPreResult->OnRestore(w, h, DXGI_FORMAT_R32G32B32A32_FLOAT);
        }

        if(m_pPostFX)
        {
            PostFXSetting* fx = dynamic_cast<PostFXSetting*>(m_pPostFX);
            if(!fx)
            {
                ProfileSetting* ps = dynamic_cast<ProfileSetting*>(m_pPostFX);
                if(!ps)
                {
                    LOG_CRITICAL_ERROR("Error while casting!");
                }
                fx = dynamic_cast<PostFXSetting*>(ps->m_pSetting);
                if(!fx)
                {
                    LOG_CRITICAL_ERROR("Error while casting!");
                }
            }
            fx->SetPreResult(m_pPreResult);
            fx->SetScene(m_pScene);
            m_pPostFX->VOnRestore(w, h);
        }

        TBD_FOR(m_albedoSettings)
        {
            (*it)->VOnRestore(w, h);
        }

        TBD_FOR(m_lightSettings)
        {
            (*it)->VOnRestore(w, h);
        }

        m_pScene->OnRestore(w, h, DXGI_FORMAT_R32G32B32A32_FLOAT);

        return TRUE;
    }

    VOID GraphicsSettings::VOnActivate(VOID)
    {
        if(m_pPreResult)
        {
            app::g_pApp->GetHumanView()->GetRenderer()->SetCurrentRendertarget(m_pPreResult);
        }
        else
        {
            app::g_pApp->GetHumanView()->GetRenderer()->SetCurrentRendertarget(m_pScene);
        }
    }

    VOID GraphicsSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();

        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

        TBD_FOR(m_albedoSettings)
        {
            (*it)->VRender();
        }

        //todo choose first
        /*if(m_pPostFX)
        {
            m_pPreResult->Clear();
            m_pPreResult->Bind();
        }
        else 
        {
            m_pScene->Clear();
            m_pScene->Bind();
        }*/

        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);

        TBD_FOR(m_lightSettings)
        {
            (*it)->VRender();
        }

        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
        app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();

        if(m_pPostFX) //TODO
        {
            m_pPostFX->VRender();
        }
        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    }

    IGraphicSetting* GraphicsSettings::GetSetting(LPCSTR name)
    {
        TBD_FOR(m_lightSettings)
        {
            if((*it)->GetName() == name)
            {
                return *it;
            }
        }
        TBD_FOR(m_albedoSettings)
        {
            if((*it)->GetName() == name)
            {
                return *it;
            }
        }
        if(m_pPostFX->GetName() == name)
        {
            return m_pPostFX;
        }
        return NULL;
    }

    DefaultGraphicsSettings::DefaultGraphicsSettings(VOID)
    {
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO, "DefShader", "albedo"), tbd::eAlbedo);
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "albedoInstanced"), tbd::eAlbedo);
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_PARTICLE_EFFECTS, "Particles", "particles"), tbd::eAlbedo);
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_SKY, "Sky", "sky"), tbd::eAlbedo);

        AddSetting(new tbd::CSMSetting(), tbd::eLighting);
        AddSetting(new tbd::GloablLightingSetting(), tbd::eLighting);
        AddSetting(new tbd::LightingSetting(), tbd::eLighting);

        SetPostFX(new tbd::PostFXSetting());
    }

    ProfileGraphicsSettings::ProfileGraphicsSettings(VOID)
    {
        ProfileSetting* ps = new ProfileSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO, "DefShader", "Albedo"));
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eAlbedo);

        ps = new ProfileSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "AlbedoInstanced"));
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eAlbedo);

        ps = new ProfileSetting(new tbd::ShaderPathSetting(tbd::eDRAW_PARTICLE_EFFECTS, "Particles", "Particles"));
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eAlbedo);

        ps = new ProfileSetting(new tbd::ShaderPathSetting(tbd::eDRAW_SKY, "Sky", "Sky"));
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eAlbedo);

        ps = new ProfileSetting(new tbd::CSMSetting());
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eLighting);

        ps = new ProfileSetting(new tbd::GloablLightingSetting());
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eLighting);

        ps = new ProfileSetting(new tbd::LightingSetting());
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();
        AddSetting(ps, tbd::eLighting);

        ps = new ProfileSetting(new tbd::PostFXSetting());
        *ps << new tbd::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new tbd::D3DOcclusionQuery();

        SetPostFX(ps);

        m_pGui = new tbd::gui::D3D_GUI();
        m_pText = new tbd::gui::GuiTextComponent();
        Dimension dim;
        dim.x = 0;//(INT)(0.4 * d3d::g_width);
        dim.y = 200;
        dim.w = 200;
        dim.h = 400;
        m_pText->VSetDimension(dim);
        m_pText->VSetAlpha(0.5f);
        m_pGui->VOnRestore();
        m_pGui->AddComponent("text", m_pText);
    }

    VOID ProfileGraphicsSettings::VRender(VOID)
    {
        GraphicsSettings::VRender();
        m_pText->ClearText();
        TBD_FOR(m_albedoSettings)
        {
            ProfileSetting* pps = dynamic_cast<ProfileSetting*>(*it);
            if(pps)
            {
                std::string t = pps->GetText();
                std::vector<std::string> s;
                util::split(t, '\n', s);
                TBD_FOR(s)
                {
                    m_pText->AppendText(*it);
                }
            }
        }
        TBD_FOR(m_lightSettings)
        {
            ProfileSetting* pps = dynamic_cast<ProfileSetting*>(*it);
            if(pps)
            {
                std::string t = pps->GetText();
                std::vector<std::string> s;
                util::split(t, '\n', s);
                TBD_FOR(s)
                {
                    m_pText->AppendText(*it);
                }
            }
        }

        if(m_pPostFX)
        {
            ProfileSetting* pps = dynamic_cast<ProfileSetting*>(m_pPostFX);
            if(pps)
            {
                std::string t = pps->GetText();
                std::vector<std::string> s;
                util::split(t, '\n', s);
                TBD_FOR(s)
                {
                    m_pText->AppendText(*it);
                }
            }
        }

        m_pGui->VDraw();
    }

    ProfileGraphicsSettings::~ProfileGraphicsSettings(VOID)
    {
        SAFE_DELETE(m_pGui);
    }

    EditorGraphicsSettings::EditorGraphicsSettings(VOID) : DefaultGraphicsSettings()
    {
        AddSetting(new tbd::EditModeSetting(), eAlbedo);
    }

    VOID EditorGraphicsSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);
        app::g_pApp->GetHumanView()->GetPicker()->VRender();
        app::g_pApp->GetHumanView()->GetPicker()->VPostRender();
        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();

        DefaultGraphicsSettings::VRender();

        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_DEBUG_INFOS);

        //clear current actor
        app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(INVALID_ACTOR_ID);
        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eSelectedActorIdBuffer);
        UINT* b = (UINT*)buffer->Map();
        b[0] = INVALID_ACTOR_ID;
        buffer->Unmap();
        //DEBUG_OUT("1");
    }

    AlbedoSettings::AlbedoSettings(VOID)
    {
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO, "DefShader", "albedo"), tbd::eAlbedo);
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "albedoInstanced"), tbd::eAlbedo);
    }

    VOID AlbedoSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();

        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

        TBD_FOR(m_albedoSettings)
        {
            (*it)->VRender();
        }
        app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();
    }

    BoundingGeoDebugSettings::BoundingGeoDebugSettings(VOID)
    {
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO, "DefShader", "albedo"), tbd::eAlbedo);
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "albedoInstanced"), tbd::eAlbedo);
        AddSetting(new tbd::ShaderPathSetting(tbd::eDRAW_PARTICLE_EFFECTS, "Particles", "particles"), tbd::eAlbedo);
        AddSetting(new tbd::BoundingGeoSetting(), tbd::eAlbedo);

        AddSetting(new tbd::GloablLightingSetting(), tbd::eLighting);
        AddSetting(new tbd::LightingSetting(), tbd::eLighting);
    }

    /*
    DebugGraphicsSettings::DebugGraphicsSettings(VOID) : m_pDeferredProgram(NULL), m_pGlobalLight(NULL), m_pParticlesProgram(NULL)
    {

    }

    BOOL DebugGraphicsSettings::VOnRestore(VOID)
    {
        if(!m_pDeferredProgram)
        {
            m_pDeferredProgram = d3d::ShaderProgram::GetProgram("DefShader").get();
            
            std::shared_ptr<d3d::ShaderProgram> globalLighting = d3d::ShaderProgram::CreateProgram(
                "DebugGlobalLighting", L"files/shader/Lighting.hlsl", "Lighting_VS", "DebugGlobalLighting_PS", NULL);
            globalLighting->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
            globalLighting->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
            globalLighting->GenerateLayout();

            m_pGlobalLight = globalLighting.get();

            m_pParticlesProgram = d3d::ShaderProgram::GetProgram("Particles").get();

            m_pScene = new d3d::RenderTarget();
        }

        m_pScene->OnRestore(m_dim.w, m_dim.h, DXGI_FORMAT_R32G32B32A32_FLOAT);

        return TRUE;
    }

    VOID DebugGraphicsSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();

        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

        DrawAlbedo(m_pDeferredProgram, d3d::ShaderProgram::GetProgram("DefShaderInstanced").get());

        //bounding geo start
        d3d::ShaderProgram::GetProgram("BoundingGeo")->Bind();

        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_BOUNDING_DEBUG);
        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
        //bounding geo end

        m_pScene->Clear();
        m_pScene->Bind();

        app::g_pApp->GetHumanView()->GetRenderer()->VPreRender();
        m_pGlobalLight->Bind();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);
        GeometryFactory::GetGlobalScreenQuad()->Bind();
        GeometryFactory::GetGlobalScreenQuad()->Draw();

        app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    }

    BOOL EditorGraphicsSettings::VOnRestore(VOID)
    {
        return m_settings->VOnRestore();
    }

    d3d::RenderTarget* EditorGraphicsSettings::VGetResult(VOID)
    {
        return m_settings->m_pScene;
    }

    VOID EditorGraphicsSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);
        app::g_pApp->GetHumanView()->GetPicker()->VRender();
        app::g_pApp->GetHumanView()->GetPicker()->VPostRender();
        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();

        app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

        DrawAlbedo(m_settings->m_pDeferredProgram, m_settings->m_pDeferredProgramInstanced);

        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_EDIT_MODE);

        m_settings->m_pParticlesProgram->Bind();

        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_PARTICLE_EFFECTS);

        d3d::ShaderProgram::GetProgram("Sky")->Bind();
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_SKY);

        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);

        m_settings->GetCSM()->Render(app::g_pApp->GetHumanView()->GetSceneGraph());

        //global lighting
        //d3d::SetDefaultViewPort();
        //d3d::SetDefaultRendertarget();
        m_settings->m_pPreResult->Clear();
        m_settings->m_pPreResult->Bind();

        app::g_pApp->GetHumanView()->GetRenderer()->VPreRender();
        m_settings->m_pGlobalLightProgram->Bind();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
        GeometryFactory::GetGlobalScreenQuad()->Bind();
        GeometryFactory::GetGlobalScreenQuad()->Draw();

        //blend point lights
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_LIGHTING);

        app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();

        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);

        m_settings->m_pEffectChain->Process();

        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_DEBUG_INFOS);

        //clear current actor
        app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(INVALID_ACTOR_ID);
        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eSelectedActorIdBuffer);
        UINT* b = (UINT*)buffer->Map();
        b[0] = INVALID_ACTOR_ID;
        buffer->Unmap();
        //DEBUG_OUT("1");

        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    }

    d3d::RenderTarget* SimpleSettings::VGetResult(VOID)
    {
        return app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->GetTarget(d3d::Diff_DiffuseColorSpecBTarget);
    }

    VOID SimpleSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

        if(m_wireFrame)
        {
            app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);
            DrawAlbedo(d3d::ShaderProgram::GetProgram("DefShader").get(), d3d::ShaderProgram::GetProgram("DefShaderInstanced").get());
            app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();

            d3d::ShaderProgram::GetProgram("Particles")->Bind();

            app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_PARTICLE_EFFECTS);
        }
        else
        {
            ID3D11Query* q = d3d::StartQuery(D3D11_QUERY_OCCLUSION);
            DrawAlbedo(d3d::ShaderProgram::GetProgram("DefShader").get(), d3d::ShaderProgram::GetProgram("DefShaderInstanced").get());
            d3d::ShaderProgram::GetProgram("Particles")->Bind();

            app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_PARTICLE_EFFECTS);
            UINT64 occs;
            d3d::EndQuery(q, &occs, sizeof(UINT64));
            DEBUG_OUT_A("%u\n", occs);
        }
    }

    BOOL SimpleSettings::VOnRestore(VOID)
    {
        return TRUE;
    }

    BOOL WireFrameFilledSettings::VOnRestore(VOID)
    {
        return TRUE;
    }

    d3d::RenderTarget* WireFrameFilledSettings::VGetResult(VOID)
    {
        return m_settings->VGetResult();
    }

    VOID WireFrameFilledSettings::VRender(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();
        
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

        DrawAlbedo(m_settings->m_pDeferredProgram, m_settings->m_pDeferredProgramInstanced);

        app::g_pApp->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);

        DrawAlbedo(d3d::ShaderProgram::GetProgram("DefShaderWire").get(), d3d::ShaderProgram::GetProgram("DefShaderWireInstanced").get());

        app::g_pApp->GetRenderer()->PopRasterizerState();

        m_settings->m_pParticlesProgram->Bind();
        
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_PARTICLE_EFFECTS);

        d3d::ShaderProgram::GetProgram("Sky")->Bind();
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_SKY);

        m_settings->m_pCSM->Render(app::g_pApp->GetHumanView()->GetSceneGraph());

        m_settings->m_pPreResult->Clear();
        m_settings->m_pPreResult->Bind();

        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);

        app::g_pApp->GetHumanView()->GetRenderer()->VPreRender();
        m_settings->m_pGlobalLightProgram->Bind();
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
        GeometryFactory::GetGlobalScreenQuad()->Bind();
        GeometryFactory::GetGlobalScreenQuad()->Draw();

        //blend point lights
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);
        app::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(tbd::eDRAW_LIGHTING);
        d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);

        app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();

        m_settings->m_pEffectChain->Process();

        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    } */
}

