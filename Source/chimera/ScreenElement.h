#pragma once
#include "stdafx.h"
#include "Vec4.h"
namespace chimera
{
    struct _Percentages
    {
        FLOAT x;
        FLOAT y;
        FLOAT w;
        FLOAT h;

        _Percentages(VOID)
        {
            x = y = 0;
            w = h = 1;
        }
    };

    class ScreenElement : public IScreenElement
    {
    protected:
        _Percentages m_dimensionPercent;
        CMDimension m_dimension;
        Color m_color;
        BOOL m_isEnable;
        BOOL m_isActive;
        std::string m_name;

    public:
        ScreenElement(VOID);

        LPCSTR VGetName(VOID) CONST;

        VOID VSetName(LPCSTR name);

        virtual VOID VSetDimension(CONST CMDimension& dim);

        CONST CMDimension& VGetDimension(VOID);

        BOOL VIsEnable(VOID) CONST;

        virtual VOID VSetEnable(BOOL enable);

        BOOL VIsActive(VOID) CONST;

        virtual VOID VSetActive(BOOL active);

        UINT VGetPosX(VOID) CONST;

        UINT VGetPosY(VOID) CONST;

        UINT VGetWidth(VOID) CONST;

        UINT VGetHeight(VOID) CONST;

        FLOAT VGetAlpha(VOID) CONST;

        CONST chimera::Color& VGetBackgroundColor(VOID) CONST;

        VOID VSetAlpha(FLOAT alpha);

        VOID VSetBackgroundColor(FLOAT r, FLOAT g, FLOAT b);

        virtual VOID VUpdate(ULONG millis);

        virtual BOOL VOnRestore(VOID);

        virtual VOID VDraw(VOID) = 0;

        virtual ~ScreenElement(VOID) { }
    };

    class ScreenElementContainer : public ScreenElement
    {
    protected:
        std::map<std::string, IScreenElement*> m_components;

    public:
        VOID AddComponent(LPCSTR name, IScreenElement* cmp);

        IScreenElement* GetComponent(LPCSTR name);

        virtual VOID VSetEnable(BOOL enable);

        virtual VOID VDraw(VOID);

        virtual VOID VSetBackgroundColor(FLOAT r, FLOAT g, FLOAT b);

        virtual VOID VUpdate(ULONG millis);

        virtual BOOL VOnRestore(VOID);

        virtual ~ScreenElementContainer(VOID);
    };

    typedef fastdelegate::FastDelegate0<VOID> DrawMethod;
    class RenderScreen : public ScreenElement, public IRenderScreen
    {
    protected:
        std::unique_ptr<IGraphicsSettings> m_pSettings;
        DrawMethod m_pFDrawMethod;

        VOID _DrawFulscreenQuad(VOID);
        VOID _DrawQuad(VOID);

    public:
        RenderScreen(std::unique_ptr<IGraphicsSettings> settings);

        virtual BOOL VOnRestore(VOID);

        IGraphicsSettings* VGetSettings(VOID) { return m_pSettings.get(); }

        virtual VOID VDraw(VOID);

        VOID VSetName(LPCSTR name)
        {
            ScreenElement::VSetName(name);
        }

        CONST CMDimension& VGetDimension(VOID)
        {
            return ScreenElement::VGetDimension();
        }

        VOID VSetDimension(CONST CMDimension& dim);

        LPCSTR VGetName(VOID) CONST
        {
            return ScreenElement::VGetName();
        }

        ~RenderScreen(VOID);
    };

    class RenderTargetScreen : public ScreenElement// : public IRendertargetScreen
    {
    private:
        IRenderTarget* m_pTarget;
    public:
        RenderTargetScreen(IRenderTarget* target);

        VOID VDraw(VOID);

        ~RenderTargetScreen(VOID);
    };

    class TextureSlotScreen : public RenderScreen
    {
    private:
        UINT m_target;
    public:
        TextureSlotScreen(UINT slot);

        VOID VDraw(VOID);

        ~TextureSlotScreen(VOID);
    };

    class DefShaderRenderScreenContainer : public ScreenElement
    {
    private:
        std::vector<std::unique_ptr<TextureSlotScreen>> m_screens;
        std::unique_ptr<IGraphicsSettings> m_settings;

    public:
        DefShaderRenderScreenContainer(std::unique_ptr<IGraphicsSettings> settings);

        VOID AddComponent(std::unique_ptr<TextureSlotScreen> screen);

        VOID VDraw(VOID);

        BOOL VOnRestore(VOID);

        ~DefShaderRenderScreenContainer(VOID);
    };
}