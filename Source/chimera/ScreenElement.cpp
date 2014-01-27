#include "ScreenElement.h"
#include "stdafx.h"
#include "Vec4.h"

namespace chimera
{
    ScreenElement::ScreenElement(void) : m_color(Color(1,1,1,1)), m_isActive(false), m_isEnable(true), m_name("unknown")
    {

    }

    const CMDimension& ScreenElement::VGetDimension(void)
    {
        return m_dimension;
    }

    LPCSTR ScreenElement::VGetName(void) const
    {
        return m_name.c_str();
    }

    bool ScreenElement::VIsIn(uint x, uint y)
    {
        return !(x < VGetPosX() || x > VGetPosX() + VGetWidth() || y < VGetPosY() || y > VGetPosY() + VGetHeight());
    }

    void ScreenElement::VSetName(LPCSTR name)
    {
        m_name = std::string(name);
    }

    bool ScreenElement::VIsActive(void) const
    {
        return m_isActive;
    }

    void ScreenElement::VSetActive(bool active)
    {
        m_isActive = active;
    }

    void ScreenElement::VSetDimension(const CMDimension& dim)
    {
        m_dimensionPercent.x = dim.x / (float)CmGetApp()->VGetWindowWidth();
        m_dimensionPercent.y = dim.y / (float)CmGetApp()->VGetWindowHeight();
        m_dimensionPercent.w = dim.w / (float)CmGetApp()->VGetWindowWidth();
        m_dimensionPercent.h = dim.h / (float)CmGetApp()->VGetWindowHeight();
        ScreenElement::VOnRestore();
    }

    uint ScreenElement::VGetHeight(void) const
    {
        return m_dimension.h;
    }

    uint ScreenElement::VGetWidth(void) const
    {
        return m_dimension.w;
    }

    uint ScreenElement::VGetPosX(void) const
    {
        return m_dimension.x;
    }

    uint ScreenElement::VGetPosY(void) const
    {
        return m_dimension.y;
    }

    float ScreenElement::VGetAlpha(void) const
    {
        return m_color.a;
    }

    void ScreenElement::VSetAlpha(float alpha)
    {
        m_color.a = alpha;
    }

    void ScreenElement::VSetBackgroundColor(float r, float g, float b)
    {
        m_color.r = r;
        m_color.g = g;
        m_color.b = b;
    }

    const Color& ScreenElement::VGetBackgroundColor(void) const
    {
        return m_color;
    }

    void ScreenElement::VUpdate(ulong millis)
    {

    }

    bool ScreenElement::VOnRestore(void)
    {
        m_dimension.x = (uint)(m_dimensionPercent.x * CmGetApp()->VGetWindowWidth());
        m_dimension.y = (uint)(m_dimensionPercent.y * CmGetApp()->VGetWindowHeight());
        m_dimension.w = (uint)(m_dimensionPercent.w * CmGetApp()->VGetWindowWidth());
        m_dimension.h = (uint)(m_dimensionPercent.h * CmGetApp()->VGetWindowHeight());
        return true;
    }

    void ScreenElementContainer::VAddComponent(IScreenElement* cmp)
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

    void ScreenElementContainer::VSetBackgroundColor(float r, float g, float b)
    {
        ScreenElement::VSetBackgroundColor(r, g, b);
        TBD_FOR(m_components)
        {
            it->second->VSetBackgroundColor(r, g, b);
        }
    }

    bool ScreenElementContainer::VOnRestore(void)
    {
        ScreenElement::VOnRestore();
        TBD_FOR(m_components)
        {
            it->second->VOnRestore();
        }
        return true;
    }

    void ScreenElementContainer::VUpdate(ulong millis)
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

    void ScreenElementContainer::VSetActive(bool active)
    {
        ScreenElement::VSetActive(active);
        TBD_FOR(m_components)
        {
            it->second->VSetActive(active);
        }
    }

    void ScreenElementContainer::VDraw(void)
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

    ScreenElementContainer::~ScreenElementContainer(void)
    {
        TBD_FOR(m_components)
        {
            SAFE_DELETE(it->second);
        }
    }

    RenderScreen::RenderScreen(std::unique_ptr<IGraphicsSettings> settings) : m_pSettings(std::move(settings)), m_pFDrawMethod(NULL)
    {

    }

    void RenderScreen::_DrawFulscreenQuad(void)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad();
    }

    void RenderScreen::_DrawQuad(void)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);
    }

    void RenderScreen::VSetDimension(const CMDimension& dim)
    {
        ScreenElement::VSetDimension(dim);
    }

    void RenderScreen::VDraw(void)
    {
        m_pSettings->VRender();
    }

    bool RenderScreen::VOnRestore(void)
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

    RenderScreen::~RenderScreen(void)
    {

    }

    RenderTargetScreen::RenderTargetScreen(IRenderTarget* target) : m_pTarget(target)
    {
    }

    void RenderTargetScreen::VDraw(void)
    {
        IRenderer* renderer = CmGetApp()->VGetHumanView()->VGetRenderer();

        renderer->VBindBackBuffer();

        renderer->VSetTexture(eDiffuseColorSampler, m_pTarget->VGetTexture());

        assert(((m_dimension.w > 0) && (m_dimension.h > 0)));

        renderer->VDrawScreenQuad(m_dimension.x, m_dimension.y, m_dimension.w, m_dimension.h);

        renderer->VSetTexture(eDiffuseColorSampler, NULL);
    }

    RenderTargetScreen::~RenderTargetScreen(void)
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