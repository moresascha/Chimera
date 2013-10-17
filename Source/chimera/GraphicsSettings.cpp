#include "GraphicsSettings.h"
#include "CascadedShadowMapper.h"

namespace chimera
{  
    VOID DrawAlbedo(IShaderProgram* prog, IShaderProgram* instancedProg)
    {
        instancedProg->VBind();
        CmGetApp()->VGetHumanView()->VGetSceneGraph()->VOnRender(CM_RENDERPATH_ALBEDO_INSTANCED);

        prog->VBind();
        CmGetApp()->VGetHumanView()->VGetSceneGraph()->VOnRender(CM_RENDERPATH_ALBEDO);
    }

    /*class LuminanceParameter : public chimera::IEffectParmaters
    {
    private:
        FLOAT m_scale;
    public:
        LuminanceParameter(FLOAT scale) : m_scale(scale) {}

        VOID VApply(VOID)
        {
            chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eGuiColorBuffer);
            float4* f4 = (float4*)buffer->Map();
            f4->w = 1;
            f4->x = f4->y = f4->z = m_scale;
            buffer->Unmap();
        }
    };*/

    ShaderPathSetting::ShaderPathSetting(RenderPath path, LPCSTR programName, LPCSTR settingName) 
        : IGraphicSetting(settingName), m_programName(programName), m_renderPath(path), m_pProgram(NULL)
    {

    }

    BOOL ShaderPathSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pProgram)
        {
            m_pProgram = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram(m_programName.c_str(), &m_desc);

        }
        return TRUE;
    }

    VOID ShaderPathSetting::VRender(VOID)
    {
        m_pProgram->VBind();
        CmGetApp()->VGetHumanView()->VGetSceneGraph()->VOnRender((RenderPath)m_renderPath);
    }

    AlbedoSetting::AlbedoSetting(VOID) : ShaderPathSetting(CM_RENDERPATH_ALBEDO, DEFERRED_SHADER_NAME, DEFERRED_SHADER_NAME)
    {

    }

    BOOL AlbedoSetting::VOnRestore(UINT w, UINT h)
    {
        VGetProgramDescription()->vs.file = DEFERRED_SHADER_FILE;
        VGetProgramDescription()->vs.function = DEFERRED_SHADER_VS_FUNCTION;

        VGetProgramDescription()->vs.layoutCount = 3;

        VGetProgramDescription()->vs.inputLayout[0].instanced = FALSE;
        VGetProgramDescription()->vs.inputLayout[0].name = "POSITION";
        VGetProgramDescription()->vs.inputLayout[0].position = 0;
        VGetProgramDescription()->vs.inputLayout[0].slot = 0;
        VGetProgramDescription()->vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

        VGetProgramDescription()->vs.inputLayout[1].instanced = FALSE;
        VGetProgramDescription()->vs.inputLayout[1].name = "NORMAL";
        VGetProgramDescription()->vs.inputLayout[1].position = 1;
        VGetProgramDescription()->vs.inputLayout[1].slot = 0;
        VGetProgramDescription()->vs.inputLayout[1].format = eFormat_R32G32B32_FLOAT;

        VGetProgramDescription()->vs.inputLayout[2].instanced = FALSE;
        VGetProgramDescription()->vs.inputLayout[2].name = "TEXCOORD";
        VGetProgramDescription()->vs.inputLayout[2].position = 2;
        VGetProgramDescription()->vs.inputLayout[2].slot = 0;
        VGetProgramDescription()->vs.inputLayout[2].format = eFormat_R32G32_FLOAT;

        VGetProgramDescription()->fs.file = DEFERRED_SHADER_FILE;
        VGetProgramDescription()->fs.function = DEFERRED_SHADER_FS_FUNCTION;

        return ShaderPathSetting::VOnRestore(w, h);
    }

    GloablLightingSetting::GloablLightingSetting(VOID) : ShaderPathSetting(CM_RENDERPATH_LIGHTING, GLOBAL_LIGHTING_SHADER_NAME, GLOBAL_LIGHTING_SHADER_NAME)
    {

    }

    BOOL GloablLightingSetting::VOnRestore(UINT w, UINT h)
    {
        m_desc.fs.file = GLOBAL_LIGHTING_SHADER_FILE;
        m_desc.fs.function = GLOBAL_LIGHTING_SHADER_FS_FUNCTION;

        m_desc.vs.layoutCount = 2;

        m_desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;
        m_desc.vs.inputLayout[0].instanced = FALSE;
        m_desc.vs.inputLayout[0].name = "POSITION";
        m_desc.vs.inputLayout[0].position = 0;
        m_desc.vs.inputLayout[0].slot = 0;

        m_desc.vs.inputLayout[1].format = eFormat_R32G32_FLOAT;
        m_desc.vs.inputLayout[1].instanced = FALSE;
        m_desc.vs.inputLayout[1].name = "TEXCOORD";
        m_desc.vs.inputLayout[1].position = 1;
        m_desc.vs.inputLayout[1].slot = 0;

        m_desc.vs.file = GLOBAL_LIGHTING_SHADER_FILE;
        m_desc.vs.function = GLOBAL_LIGHTING_SHADER_VS_FUNCTION;

        return ShaderPathSetting::VOnRestore(w, h);
    }

    VOID GloablLightingSetting::VRender(VOID)
    {
		/*CmGetApp()->VGetHumanView()->VGetRenderer()->VGetCurrentRenderTarget()->VClear();
        CmGetApp()->VGetHumanView()->VGetRenderer()->VGetCurrentRenderTarget()->VBind();*/

        /*CmGetApp()->VGetHumanView()->VGetRenderer()->VPreRender();*/

        //CmGetApp()->VGetHumanView()->VGetRenderer()->VClearAndBindBackBuffer();

        m_pProgram->VBind();
        //chimera::GetContext()->OMSetDepthStencilState(chimera::m_pNoDepthNoStencilState, 0);
		CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad();
    }

    BOOL CSMSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pCSM)
        {
            m_pCSM = new CascadedShadowMapper(3);
        }
        m_pCSM->VOnRestore();
        return TRUE;
    }

    VOID CSMSetting::VRender(VOID)
    {
        m_pCSM->VRender(CmGetApp()->VGetHumanView()->VGetSceneGraph());
    }

    CSMSetting::~CSMSetting(VOID)
    {
        SAFE_DELETE(m_pCSM);
    }

    VOID LightingSetting::VRender(VOID)
    {
        //d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);
        CmGetApp()->VGetHumanView()->VGetSceneGraph()->VOnRender(CM_RENDERPATH_LIGHTING);
    }
    
    VOID PostFXSetting::VSetTarget(IRenderTarget* target)
    {
        m_pTarget = target;
    }

    VOID PostFXSetting::VSetSource(IRenderTarget* src)
    {
        m_pSource = src;
	}

    BOOL PostFXSetting::VOnRestore(UINT w, UINT h)
    {
        if(!m_pEffectChain)
        {
			IEffectFactory* eff = CmGetApp()->VGetHumanView()->VGetEffectFactory();
			m_pEffectChain = eff->VCreateEffectChain();

			m_pEffectChain->VSetTarget(NULL);

			CMShaderDescription desc;
			desc.file = L"Effects.hlsl";
			desc.function = "Luminance";

	        IEffect* lumi = m_pEffectChain->VCreateEffect(desc);
            lumi->VSetSource(m_pSource);

            //lumi->SetParameters(std::shared_ptr<LuminanceParameter>(new LuminanceParameter(1.0f)));
            
            IEffect* e = NULL;
			
            for(INT i = w / 2; i > 1; i = i >> 1)
            {
                FLOAT s = (FLOAT)i / (FLOAT)w;
                if(e == NULL)
                {
					desc.function = "Sample";
                    IEffect* ds = m_pEffectChain->VCreateEffect(desc, s, s);
                    ds->VAddRequirement(lumi);
                    e = ds;
                }
                else
                {
					desc.function = "Sample";
                    IEffect* ds = m_pEffectChain->VCreateEffect(desc, s, s);
                    ds->VAddRequirement(e);
                    e = ds;
                }
            }
          
			desc.function = "Sample";
            IEffect* ds = m_pEffectChain->VCreateEffect(desc, 1.0f / w, 1.0f / h);
            ds->VAddRequirement(e);
            e = ds;

            FLOAT brightPathSize = 0.25f;

			desc.function = "Brightness";

            IEffect* bright = m_pEffectChain->VCreateEffect(desc, brightPathSize, brightPathSize);
            bright->VSetSource(m_pSource);

			desc.function = "BlurH";
            IEffect* e0 = m_pEffectChain->VCreateEffect(desc, brightPathSize, brightPathSize);
            e0->VAddRequirement(bright);

			desc.function = "BlurV";
            IEffect* e1 = m_pEffectChain->VCreateEffect(desc, brightPathSize, brightPathSize);
            e1->VAddRequirement(e0);

			desc.function = "ToneMap";
            IEffect* e2 = m_pEffectChain->VCreateEffect(desc);
            e2->VSetSource(m_pSource);
            e2->VAddRequirement(e1);

			m_pEffectChain->VOnRestore(w, h);
        }
        else
        {
            m_pEffectChain->VOnRestore(w, h);
        }
        return TRUE;
    }

    VOID PostFXSetting::VRender(VOID)
    {
        m_pEffectChain->VProcess();
    }

    PostFXSetting::~PostFXSetting(VOID)
    {
        SAFE_DELETE(m_pEffectChain);
    }

    BoundingGeoSetting::BoundingGeoSetting(VOID) : ShaderPathSetting(CM_RENDERPATH_BOUNDING, "BoundingGeo", "bgeo")
    {
         //CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateStateFactory()->VCreateRasterState()
    }

    VOID BoundingGeoSetting::VRender(VOID)
    {
        //CmGetApp()->VGetRenderer()->VPushRasterState(chimera::g_pRasterizerStateWrireframe);
        ShaderPathSetting::VRender();
        //CmGetApp()->VGetHumanView()->VGetRenderer()->VPopRasterState();
    }
    /*
    ProfileSetting::ProfileSetting(IGraphicSetting* setting) : IGraphicSetting("profile"), m_pSetting(setting)
    {

    }

    ProfileSetting& operator<<(ProfileSetting& settings, chimera::Query* q)
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
    }*/

    EditModeSetting::EditModeSetting(VOID) : IGraphicSetting("edit")
    {

    }

    VOID EditModeSetting::VRender(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetSceneGraph()->VOnRender(CM_RENDERPATH_EDITOR);
    }
    
    //Settings...

    GraphicsSettings::GraphicsSettings(VOID) : m_lastW(0), m_lastH(0)
    {

    }

    VOID GraphicsSettings::VAddSetting(std::unique_ptr<IGraphicSetting> setting, GraphicsSettingType type)
    {
        if(type == eGraphicsSetting_Albedo)
        {
            m_albedoSettings.push_back(std::move(setting));
        }
        else if(type == eGraphicsSetting_Lighting)
        {
            m_lightSettings.push_back(std::move(setting));
        }
    }

    VOID GraphicsSettings::VSetPostFX(std::unique_ptr<IPostFXSetting> setting)
    {
        m_pPostFX = std::move(setting);

        /*if(!m_pPreResult)
        {
            m_pPreResult = std::move(CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget());
        } */
    }

    GraphicsSettings::~GraphicsSettings(VOID)
    {
		/*
        m_pPostFX.reset();
	
        TBD_FOR(m_lightSettings)
        {
            it->reset();
        }

        TBD_FOR(m_albedoSettings)
        {
            it->reset();
        }*/
    }

    IRenderTarget* GraphicsSettings::VGetResult(VOID)
    {
        return m_pScene.get();
    }

    BOOL GraphicsSettings::VOnRestore(UINT w, UINT h)
    {
        m_lastW = w;
        m_lastH = h;
        if(!m_pScene)
        {
            m_pScene = std::move(CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget());
        }

        /*if(m_pPreResult)
        {
            m_pPreResult->VOnRestore(w, h, eFormat_R32G32B32A32_FLOAT);
        }*/
        
        if(m_pPostFX)
        {
			m_pPostFX->VSetSource(m_pScene.get());
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
        
        m_pScene->VOnRestore(w, h, eFormat_R32G32B32A32_FLOAT);

        std::unique_ptr<IGraphicsStateFactroy> factory = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateStateFactory();
        
        RasterStateDesc rasterDesc;
        ZeroMemory(&rasterDesc, sizeof(RasterStateDesc));
        rasterDesc.CullMode = eCullMode_Back;
        rasterDesc.FillMode = eFillMode_Solid;
        rasterDesc.DepthClipEnable = TRUE;
        rasterDesc.FrontCounterClockwise = TRUE;
        rasterDesc.MultisampleEnable = FALSE;
        rasterDesc.AntialiasedLineEnable = FALSE;

        m_pRasterizerStateFrontFaceSolid.reset();
        m_pRasterizerStateFrontFaceSolid = std::unique_ptr<IRasterState>(factory->VCreateRasterState(&rasterDesc));

        DepthStencilStateDesc dDesc;
        ZeroMemory(&dDesc, sizeof(DepthStencilStateDesc));
        dDesc.DepthEnable = TRUE;
        dDesc.DepthWriteMask = eDepthWriteMask_All;
        dDesc.DepthFunc = eCompareFunc_Less_Equal;
        dDesc.StencilEnable = FALSE;

        m_pDepthNoStencilState.reset();
        m_pDepthNoStencilState = std::unique_ptr<IDepthStencilState>(factory->VCreateDepthStencilState(&dDesc));

        DepthStencilStateDesc dsDesc;
        ZeroMemory(&dsDesc, sizeof(DepthStencilStateDesc));

        dsDesc.DepthEnable = TRUE;
        dsDesc.DepthWriteMask = eDepthWriteMask_All;
        dsDesc.DepthFunc = eCompareFunc_Less_Equal;
        dsDesc.StencilEnable = FALSE;

        dsDesc.StencilEnable = TRUE;
        dsDesc.StencilWriteMask = 0xFF;
        dsDesc.StencilReadMask = 0xFF;

        dsDesc.FrontFace.StencilFailOp = eStencilOP_Keep;
        dsDesc.FrontFace.StencilDepthFailOp = eStencilOP_Keep;
        dsDesc.FrontFace.StencilPassOp = eStencilOP_Incr;
        dsDesc.FrontFace.StencilFunc = eCompareFunc_Always;

        dsDesc.BackFace.StencilFailOp = eStencilOP_Keep;
        dsDesc.BackFace.StencilDepthFailOp = eStencilOP_Keep;
        dsDesc.BackFace.StencilPassOp = eStencilOP_Incr; 
        dsDesc.BackFace.StencilFunc = eCompareFunc_Always;

        m_pDepthWriteStencilState.reset();
        m_pDepthWriteStencilState = std::unique_ptr<IDepthStencilState>(factory->VCreateDepthStencilState(&dsDesc));

        return TRUE;
    }

    VOID GraphicsSettings::VOnActivate(VOID)
    {
        /*if(m_pPreResult)
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VPushCurrentRenderTarget(m_pPreResult.get());
        }
        else*/
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VPushCurrentRenderTarget(m_pScene.get());
        }
    }

    VOID GraphicsSettings::VRender(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushDepthStencilState(m_pDepthWriteStencilState.get(), -1);

        CmGetApp()->VGetRenderer()->VPostRender();

        TBD_FOR(m_albedoSettings)
        {
            (*it)->VRender();
        }

        CmGetApp()->VGetRenderer()->VPreRender();

        //todo choose first
        if(m_pPostFX)
        {
            m_pScene->VClear();
            m_pScene->VBind();
        }
        else
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VClearAndBindBackBuffer();
        }

        CmGetApp()->VGetHumanView()->VGetRenderer()->VPopDepthStencilState();

        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushDepthStencilState(m_pDepthNoStencilState.get());

        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushRasterState(m_pRasterizerStateFrontFaceSolid.get());

        TBD_FOR(m_lightSettings)
        {
            (*it)->VRender();
        }

        if(m_pPostFX) //TODO
        {
            m_pPostFX->VRender();
        }
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPopDepthStencilState();

        CmGetApp()->VGetRenderer()->VPopRasterState();
    }

    /*
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
    } */

    DefaultGraphicsSettings::DefaultGraphicsSettings(VOID)
    {
        VAddSetting(std::unique_ptr<IGraphicSetting>(new AlbedoSetting()), eGraphicsSetting_Albedo);
        /*VAddSetting(std::unique_ptr<IGraphicSetting>(new ShaderPathSetting(eRenderPath_DrawToAlbedoInstanced, "DefShaderInstanced", "albedoInstanced")), eGraphicsSetting_Albedo);
        VAddSetting(std::unique_ptr<IGraphicSetting>(new ShaderPathSetting(eRenderPath_DrawParticles, "Particles", "particles")), eGraphicsSetting_Albedo);*/

        ShaderPathSetting* skySettings = new ShaderPathSetting(CM_RENDERPATH_SKY, "Sky", "sky");

        skySettings->VGetProgramDescription()->vs.file = L"Sky.hlsl";
        skySettings->VGetProgramDescription()->vs.function = "Sky_VS";

        skySettings->VGetProgramDescription()->vs.layoutCount = 2;
        skySettings->VGetProgramDescription()->vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;
        skySettings->VGetProgramDescription()->vs.inputLayout[0].instanced = FALSE;
        skySettings->VGetProgramDescription()->vs.inputLayout[0].name = "POSITION";
        skySettings->VGetProgramDescription()->vs.inputLayout[0].position = 0;
        skySettings->VGetProgramDescription()->vs.inputLayout[0].slot = 0;

        skySettings->VGetProgramDescription()->vs.inputLayout[1].format = eFormat_R32G32_FLOAT;
        skySettings->VGetProgramDescription()->vs.inputLayout[1].instanced = FALSE;
        skySettings->VGetProgramDescription()->vs.inputLayout[1].name = "TEXCOORD";
        skySettings->VGetProgramDescription()->vs.inputLayout[1].position = 1;
        skySettings->VGetProgramDescription()->vs.inputLayout[1].slot = 0;

        skySettings->VGetProgramDescription()->fs.file = L"Sky.hlsl";
        skySettings->VGetProgramDescription()->fs.function = "Sky_PS";

        VAddSetting(std::unique_ptr<IGraphicSetting>(skySettings), eGraphicsSetting_Albedo);

        VAddSetting(std::unique_ptr<IGraphicSetting>(new CSMSetting()), eGraphicsSetting_Lighting);
        VAddSetting(std::unique_ptr<IGraphicSetting>(new GloablLightingSetting()), eGraphicsSetting_Lighting);
        //VAddSetting(std::unique_ptr<IGraphicSetting>(new LightingSetting()), chimera::eLighting);

        VSetPostFX(std::unique_ptr<IPostFXSetting>(new PostFXSetting()));
    }

    /*
    ProfileGraphicsSettings::ProfileGraphicsSettings(VOID)
    {
        ProfileSetting* ps = new ProfileSetting(new chimera::ShaderPathSetting(chimera::eDRAW_TO_ALBEDO, "DefShader", "Albedo"));
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eAlbedo);

        ps = new ProfileSetting(new chimera::ShaderPathSetting(chimera::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "AlbedoInstanced"));
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eAlbedo);

        ps = new ProfileSetting(new chimera::ShaderPathSetting(chimera::eDRAW_PARTICLE_EFFECTS, "Particles", "Particles"));
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eAlbedo);

        ps = new ProfileSetting(new chimera::ShaderPathSetting(chimera::eDRAW_SKY, "Sky", "Sky"));
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eAlbedo);

        ps = new ProfileSetting(new chimera::CSMSetting());
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eLighting);

        ps = new ProfileSetting(new chimera::GloablLightingSetting());
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eLighting);

        ps = new ProfileSetting(new chimera::LightingSetting());
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();
        AddSetting(ps, chimera::eLighting);

        ps = new ProfileSetting(new chimera::PostFXSetting());
        *ps << new chimera::D3DTimeDeltaQuery();
        //*ps << new tbd::D3DPipelineStatisticsQuery();
        *ps << new chimera::D3DOcclusionQuery();

        SetPostFX(ps);

        m_pGui = new chimera::gui::D3D_GUI();
        m_pText = new chimera::gui::GuiTextComponent();
        CMDimension dim;
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
        AddSetting(new chimera::EditModeSetting(), eAlbedo);
    }

    VOID EditorGraphicsSettings::VRender(VOID)
    {
        DefaultGraphicsSettings::VRender();

        chimera::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(chimera::eDRAW_DEBUG_INFOS);

        //clear current actor
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetActorId(CM_INVALID_ACTOR_ID);
        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eSelectedActorIdBuffer);
        UINT* b = (UINT*)buffer->Map();
        b[0] = CM_INVALID_ACTOR_ID;
        buffer->Unmap();
        //DEBUG_OUT("1");
    }

    AlbedoSettings::AlbedoSettings(VOID)
    {
        AddSetting(new chimera::ShaderPathSetting(chimera::eDRAW_TO_ALBEDO, "DefShader", "albedo"), chimera::eAlbedo);
        AddSetting(new chimera::ShaderPathSetting(chimera::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "albedoInstanced"), chimera::eAlbedo);
        AddSetting(new chimera::ShaderPathSetting(chimera::eDRAW_PARTICLE_EFFECTS, "Particles", "particles"), chimera::eAlbedo);
        //AddSetting(new tbd::BoundingGeoSetting(), tbd::eAlbedo);
    }

    VOID AlbedoSettings::VRender(VOID)
    {
        chimera::g_pApp->GetHumanView()->GetRenderer()->GetDeferredShader()->ClearAndBindRenderTargets();

        chimera::GetContext()->OMSetDepthStencilState(chimera::m_pDepthNoStencilState, 0);

        TBD_FOR(m_albedoSettings)
        {
            (*it)->VRender();
        }
        chimera::g_pApp->GetHumanView()->GetRenderer()->VPostRender();
    }

    BoundingGeoDebugSettings::BoundingGeoDebugSettings(VOID)
    {
        AddSetting(new chimera::ShaderPathSetting(chimera::eDRAW_TO_ALBEDO, "DefShader", "albedo"), chimera::eAlbedo);
        AddSetting(new chimera::ShaderPathSetting(chimera::eDRAW_TO_ALBEDO_INSTANCED, "DefShaderInstanced", "albedoInstanced"), chimera::eAlbedo);
        AddSetting(new chimera::ShaderPathSetting(chimera::eDRAW_PARTICLE_EFFECTS, "Particles", "particles"), chimera::eAlbedo);
        AddSetting(new chimera::BoundingGeoSetting(), chimera::eAlbedo);

        AddSetting(new chimera::GloablLightingSetting(), chimera::eLighting);
        AddSetting(new chimera::LightingSetting(), chimera::eLighting);
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

