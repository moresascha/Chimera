#pragma once
#include "stdafx.h"
#include "Vec4.h"
namespace chimera
{
    struct _Percentages
    {
        float x;
        float y;
        float w;
        float h;

        _Percentages(void)
        {
            x = y = 0;
            w = h = 1;
        }
    };

    class ScreenElement : virtual public IScreenElement
    {
    protected:
        _Percentages m_dimensionPercent;
        CMDimension m_dimension;
        Color m_color;
        bool m_isEnable;
        bool m_isActive;
        std::string m_name;

    public:
        ScreenElement(void);

        LPCSTR VGetName(void) const;

        void VSetName(LPCSTR name);

        virtual void VSetDimension(const CMDimension& dim);

        virtual bool VIsIn(uint x, uint y);

        const CMDimension& VGetDimension(void);

        bool VIsActive(void) const;

        virtual void VSetActive(bool active);

        uint VGetPosX(void) const;

        uint VGetPosY(void) const;

        uint VGetWidth(void) const;

        uint VGetHeight(void) const;

        float VGetAlpha(void) const;

        const chimera::Color& VGetBackgroundColor(void) const;

        void VSetAlpha(float alpha);

        void VSetBackgroundColor(float r, float g, float b);

        virtual void VUpdate(ulong millis);

        virtual bool VOnRestore(void);

        virtual void VDraw(void) = 0;

        virtual ~ScreenElement(void) { }
    };

    class ScreenElementContainer : public virtual IScreenElementContainer, public ScreenElement
    {
    protected:
        std::map<std::string, IScreenElement*> m_components;

    public:
        void VAddComponent(IScreenElement* cmp);

        IScreenElement* VGetComponent(LPCSTR name);

        virtual void VSetActive(bool active);

        virtual void VDraw(void);

        virtual void VSetBackgroundColor(float r, float g, float b);

        virtual void VUpdate(ulong millis);

        virtual bool VOnRestore(void);

        virtual ~ScreenElementContainer(void);
    };

    typedef fastdelegate::FastDelegate0<void> DrawMethod;
    class RenderScreen : public ScreenElement, public IRenderScreen
    {
    protected:
        std::unique_ptr<IGraphicsSettings> m_pSettings;
        DrawMethod m_pFDrawMethod;

        void _DrawFulscreenQuad(void);
        void _DrawQuad(void);

    public:
        RenderScreen(std::unique_ptr<IGraphicsSettings> settings);

        virtual bool VOnRestore(void);

        IGraphicsSettings* VGetSettings(void) { return m_pSettings.get(); }

        virtual void VDraw(void);

        void VSetDimension(const CMDimension& dim);

        ~RenderScreen(void);
    };

   
    class RenderTargetScreen : public ScreenElement
    {
    private:
        IRenderTarget* m_pTarget;
    public:
        RenderTargetScreen(IRenderTarget* target);

        void VDraw(void);

        ~RenderTargetScreen(void);
    };
     /*
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
    };*/
}