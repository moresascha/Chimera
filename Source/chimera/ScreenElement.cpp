#include "ScreenElement.h"
#include "GraphicsSettings.h"
#include "D3DRenderer.h"
#include "GameView.h"
#include "GameApp.h"
#include "d3d.h"
#include "Effect.h"
namespace tbd
{
    ScreenElement::ScreenElement(VOID) : m_color(util::Color(1,1,1,1)), m_isActive(TRUE), m_isEnable(TRUE), m_name("unknown")
    {
    }

    Dimension ScreenElement::VGetDimension(VOID)
    {
        return m_dimension;
    }

    LPCSTR ScreenElement::VGetName(VOID) CONST
    {
        return m_name.c_str();
    }

    VOID ScreenElement::VSetName(LPCSTR name)
    {
        m_name = name;
    }

    BOOL ScreenElement::VIsEnable(VOID) CONST
    {
        return m_isEnable;
    }

    VOID ScreenElement::VSetEnable(BOOL enable)
    {
        m_isEnable = enable;
    }

    BOOL ScreenElement::VIsActive(VOID) CONST
    {
        return m_isActive;
    }

    VOID ScreenElement::VSetActive(BOOL active)
    {
        m_isActive = active;
    }

    VOID ScreenElement::VSetDimension(CONST Dimension& dim)
    {
        m_dimensionPercent.x = dim.x / (FLOAT)app::g_pApp->GetWindowWidth();
        m_dimensionPercent.y = dim.y / (FLOAT)app::g_pApp->GetWindowHeight();
        m_dimensionPercent.w = dim.w / (FLOAT)app::g_pApp->GetWindowWidth();
        m_dimensionPercent.h = dim.h / (FLOAT)app::g_pApp->GetWindowHeight();
    }

    UINT ScreenElement::VGetHeight(VOID) CONST
    {
        return m_dimension.h;
    }

    UINT ScreenElement::VGetWidth(VOID) CONST
    {
        return m_dimension.w;
    }

    UINT ScreenElement::VGetPosX(VOID) CONST
    {
        return m_dimension.x;
    }

    UINT ScreenElement::VGetPosY(VOID) CONST
    {
        return m_dimension.y;
    }

    FLOAT ScreenElement::VGetAlpha(VOID) CONST
    {
        return m_color.a;
    }

    VOID ScreenElement::VSetAlpha(FLOAT alpha)
    {
        m_color.a = alpha;
    }

    VOID ScreenElement::VSetBackgroundColor(FLOAT r, FLOAT g, FLOAT b)
    {
        m_color.r = r;
        m_color.g = g;
        m_color.b = b;
    }

    CONST util::Color& ScreenElement::VGetBackgroundColor(VOID) CONST
    {
        return m_color;
    }

    VOID ScreenElement::VUpdate(ULONG millis)
    {

    }

    BOOL ScreenElement::VOnRestore(VOID)
    {
        m_dimension.x = (UINT)(m_dimensionPercent.x * d3d::GetWindowWidth());
        m_dimension.y = (UINT)(m_dimensionPercent.y * d3d::GetWindowHeight());
        m_dimension.w = (UINT)(m_dimensionPercent.w * d3d::GetWindowWidth());
        m_dimension.h = (UINT)(m_dimensionPercent.h * d3d::GetWindowHeight());
        return TRUE;
    }

    VOID ScreenElementContainer::VSetEnable(BOOL enable)
    {
        TBD_FOR(m_components)
        {
            it->second->VSetEnable(enable);
        }
    }

    VOID ScreenElementContainer::AddComponent(LPCSTR name, IScreenElement* cmp)
    {
        if(m_components.find(name) != m_components.end())
        {
            LOG_CRITICAL_ERROR("component with this name already exists!");
            return;
        }
        m_components[name] = cmp;
        cmp->VOnRestore();
    }

    IScreenElement* ScreenElementContainer::GetComponent(LPCSTR name)
    {
        auto it = m_components.find(name);
        if(it == m_components.end())
        {
            LOG_CRITICAL_ERROR("unknown component");
        }
        return it->second;
    }
    
    VOID ScreenElementContainer::VSetBackgroundColor(FLOAT r, FLOAT g, FLOAT b)
    {
        TBD_FOR(m_components)
        {
            it->second->VSetBackgroundColor(r, g, b);
        }
    }
    
    BOOL ScreenElementContainer::VOnRestore(VOID)
    {
        TBD_FOR(m_components)
        {
            it->second->VOnRestore();
        }
        return TRUE;
    }

    VOID ScreenElementContainer::VUpdate(ULONG millis)
    {
        if(VIsActive())
        {
            TBD_FOR(m_components)
            {
                if(it->second->VIsActive())
                {
                    it->second->VUpdate(millis);
                }
            }
        }
    }
    
    VOID ScreenElementContainer::VDraw(VOID)
    {
        if(VIsActive())
        {
            TBD_FOR(m_components)
            {
                if(it->second->VIsActive())
                {
                    (it->second)->VDraw();
                }               
            }            
        }        
    }

    ScreenElementContainer::~ScreenElementContainer(VOID)
    {
        TBD_FOR(m_components)
        {
            SAFE_DELETE(it->second);
        }
    }

    RenderScreen::RenderScreen(std::shared_ptr<IGraphicsSettings> settings) : m_pSettings(settings)
    {

    }

    VOID RenderScreen::VDraw(VOID)
    {
        m_pSettings->VRender();

        d3d::BindBackbuffer();

        app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eDiffuseColorSampler, m_pSettings->VGetResult()->GetShaderRessourceView());

        assert(((m_dimension.w > 0) && (m_dimension.h > 0)));
        
        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);

        d3d::DrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);

        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    }

    BOOL RenderScreen::VOnRestore(VOID)
    {
        ScreenElement::VOnRestore();
        return m_pSettings->VOnRestore(m_dimension.w, m_dimension.h);
    }

    RenderScreen::~RenderScreen(VOID)
    {

    }

    RendertargetScreen::RendertargetScreen(d3d::RenderTarget* target) : m_pTarget(target)
    {
    }

    VOID RendertargetScreen::VDraw(VOID)
    {
        d3d::BindBackbuffer();

        app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eDiffuseColorSampler, m_pTarget->GetShaderRessourceView());

        assert(((m_dimension.w > 0) && (m_dimension.h > 0)));

        d3d::DrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);

        app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eDiffuseColorSampler, NULL);
    }

    RendertargetScreen::~RendertargetScreen(VOID)
    {

    }

    //---
    DefShaderRenderScreen::DefShaderRenderScreen(UINT target) : RenderScreen(NULL), m_target(target)
    {
        m_pSettings = std::shared_ptr<tbd::AlbedoSettings>(new tbd::AlbedoSettings());
    }

    VOID DefShaderRenderScreen::VDraw(VOID)
    {
        m_pSettings->VRender();

        d3d::BindBackbuffer();

        app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eDiffuseColorSampler, 
            app::g_pApp->GetRenderer()->GetDeferredShader()->GetTarget((d3d::Diff_RenderTargets)m_target)->GetShaderRessourceView());

        assert(((m_dimension.w > 0) && (m_dimension.h > 0)));

        d3d::DrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);

        app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eDiffuseColorSampler, NULL);
    }

    DefShaderRenderScreen::~DefShaderRenderScreen(VOID)
    {

    }

    DefShaderRenderScreenContainer::DefShaderRenderScreenContainer(std::shared_ptr<IGraphicsSettings> settings) : m_settings(settings)
    {

    }

    BOOL DefShaderRenderScreenContainer::VOnRestore(VOID)
    {
        LOG_CRITICAL_ERROR("todo");
        m_settings->VOnRestore(d3d::GetWindowWidth(), d3d::GetWindowHeight());
        TBD_FOR(m_screens)
        {
            (*it)->VOnRestore();
        }
        return TRUE;
    }

    VOID DefShaderRenderScreenContainer::AddComponent(DefShaderRenderScreen* screen)
    {
        m_screens.push_back(screen);
    }

    VOID DefShaderRenderScreenContainer::VDraw(VOID)
    {
        m_settings->VRender();
        TBD_FOR(m_screens)
        {
            (*it)->VDraw();
        }
    }

    DefShaderRenderScreenContainer::~DefShaderRenderScreenContainer(VOID)
    {
        TBD_FOR(m_screens)
        {
            SAFE_DELETE(*it);
        }
    }
}
