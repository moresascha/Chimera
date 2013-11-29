#include "ScreenElement.h"
#include "stdafx.h"
#include "Vec4.h"

namespace chimera
{
    ScreenElement::ScreenElement(VOID) : m_color(Color(0,0,0,1)), m_isActive(FALSE), m_isEnable(TRUE), m_name("unknown")
    {

    }

    CONST CMDimension& ScreenElement::VGetDimension(VOID)
    {
        return m_dimension;
    }

    LPCSTR ScreenElement::VGetName(VOID) CONST
    {
        return m_name.c_str();
    }

    BOOL ScreenElement::VIsIn(UINT x, UINT y)
    {
        return !(x < VGetPosX() || x > VGetPosX() + VGetWidth() || y < VGetPosY() || y > VGetPosY() + VGetHeight());
    }

    VOID ScreenElement::VSetName(LPCSTR name)
    {
        m_name = std::string(name);
    }

    BOOL ScreenElement::VIsActive(VOID) CONST
    {
        return m_isActive;
    }

    VOID ScreenElement::VSetActive(BOOL active)
    {
        m_isActive = active;
    }

    VOID ScreenElement::VSetDimension(CONST CMDimension& dim)
    {
        m_dimensionPercent.x = dim.x / (FLOAT)CmGetApp()->VGetWindowWidth();
        m_dimensionPercent.y = dim.y / (FLOAT)CmGetApp()->VGetWindowHeight();
        m_dimensionPercent.w = dim.w / (FLOAT)CmGetApp()->VGetWindowWidth();
        m_dimensionPercent.h = dim.h / (FLOAT)CmGetApp()->VGetWindowHeight();
        ScreenElement::VOnRestore();
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

    CONST Color& ScreenElement::VGetBackgroundColor(VOID) CONST
    {
        return m_color;
    }

    VOID ScreenElement::VUpdate(ULONG millis)
    {

    }

    BOOL ScreenElement::VOnRestore(VOID)
    {
        m_dimension.x = (UINT)(m_dimensionPercent.x * CmGetApp()->VGetWindowWidth());
        m_dimension.y = (UINT)(m_dimensionPercent.y * CmGetApp()->VGetWindowHeight());
        m_dimension.w = (UINT)(m_dimensionPercent.w * CmGetApp()->VGetWindowWidth());
        m_dimension.h = (UINT)(m_dimensionPercent.h * CmGetApp()->VGetWindowHeight());
        return TRUE;
    }

    VOID ScreenElementContainer::VAddComponent(IScreenElement* cmp)
    {
        if(m_components.find(cmp->VGetName()) != m_components.end())
        {
            LOG_CRITICAL_ERROR("component with this name already exists!");
            return;
        }
        m_components[cmp->VGetName()] = cmp;
        cmp->VOnRestore();
    }

    IScreenElement* ScreenElementContainer::VGetComponent(LPCSTR name)
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
        ScreenElement::VSetBackgroundColor(r, g, b);
        TBD_FOR(m_components)
        {
            it->second->VSetBackgroundColor(r, g, b);
        }
    }

    BOOL ScreenElementContainer::VOnRestore(VOID)
    {
        ScreenElement::VOnRestore();
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
            ScreenElement::VUpdate(millis);
            TBD_FOR(m_components)
            {
                if(it->second->VIsActive())
                {
                    it->second->VUpdate(millis);
                }
            }
        }
    }

    VOID ScreenElementContainer::VSetActive(BOOL active)
    {
        ScreenElement::VSetActive(active);
        TBD_FOR(m_components)
        {
            it->second->VSetActive(active);
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

    RenderScreen::RenderScreen(std::unique_ptr<IGraphicsSettings> settings) : m_pSettings(std::move(settings)), m_pFDrawMethod(NULL)
    {

    }

    VOID RenderScreen::_DrawFulscreenQuad(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad();
    }

    VOID RenderScreen::_DrawQuad(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);
    }

    VOID RenderScreen::VSetDimension(CONST CMDimension& dim)
    {
        ScreenElement::VSetDimension(dim);
    }

    VOID RenderScreen::VDraw(VOID)
    {
        m_pSettings->VRender();
    }

    BOOL RenderScreen::VOnRestore(VOID)
    {
        ScreenElement::VOnRestore();

        if(m_dimension.x > 0 || m_dimension.y > 0 || m_dimension.w != CmGetApp()->VGetWindowWidth() || m_dimension.h != CmGetApp()->VGetWindowHeight())
        {
            m_pFDrawMethod = fastdelegate::MakeDelegate(this, &RenderScreen::_DrawQuad);
        }
        else
        {
            m_pFDrawMethod = fastdelegate::MakeDelegate(this, &RenderScreen::_DrawFulscreenQuad);
        }

        return m_pSettings->VOnRestore(m_dimension.w, m_dimension.h);
    }

    RenderScreen::~RenderScreen(VOID)
    {

    }

    RenderTargetScreen::RenderTargetScreen(IRenderTarget* target) : m_pTarget(target)
    {
    }

    VOID RenderTargetScreen::VDraw(VOID)
    {
        IRenderer* renderer = CmGetApp()->VGetHumanView()->VGetRenderer();

        renderer->VBindBackBuffer();

        renderer->VSetTexture(eDiffuseColorSampler, m_pTarget->VGetTexture());

        assert(((m_dimension.w > 0) && (m_dimension.h > 0)));

        renderer->VDrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);

        renderer->VSetTexture(eDiffuseColorSampler, NULL);
    }

    RenderTargetScreen::~RenderTargetScreen(VOID)
    {

    }
     /*
    //---
    TextureSlotScreen::TextureSlotScreen(UINT target) : RenderScreen(NULL), m_target(target)
    {
       // m_pSettings = std::shared_ptr<chimera::AlbedoSettings>(new chimera::AlbedoSettings());
    }

    VOID TextureSlotScreen::VDraw(VOID)
    {

        m_pSettings->VRender();

        chimera::BindBackbuffer();

        chimera::g_pApp->GetHumanView()->GetRenderer()->SetSampler(chimera::eDiffuseColorSampler, 
            chimera::g_pApp->GetRenderer()->GetDeferredShader()->GetTarget((chimera::Diff_RenderTargets)m_target)->GetShaderRessourceView());

        assert(((m_dimension.w > 0) && (m_dimension.h > 0)));

        chimera::DrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);

        chimera::g_pApp->GetHumanView()->GetRenderer()->SetSampler(chimera::eDiffuseColorSampler, NULL);
    }

    TextureSlotScreen::~TextureSlotScreen(VOID)
    {

    }

    DefShaderRenderScreenContainer::DefShaderRenderScreenContainer(std::unique_ptr<IGraphicsSettings> settings) : m_settings(std::move(settings))
    {

    }

    BOOL DefShaderRenderScreenContainer::VOnRestore(VOID)
    {
        LOG_CRITICAL_ERROR("todo");
       // m_settings->VOnRestore(chimera::GetWindowWidth(), chimera::GetWindowHeight());
        TBD_FOR(m_screens)
        {
            (*it)->VOnRestore();
        }
        return TRUE;
    }

    VOID DefShaderRenderScreenContainer::AddComponent(std::unique_ptr<TextureSlotScreen> screen)
    {
        m_screens.push_back(std::move(screen));
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

    }*/
}