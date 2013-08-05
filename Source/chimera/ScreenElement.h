#pragma once
#include "stdafx.h"
#include "Vec4.h"

namespace d3d
{
    class RenderTarget;
}

namespace tbd
{
    struct Dimension
    {
        UINT x;
        UINT y;
        UINT w;
        UINT h;

        Dimension(VOID)
        {
            x = 0;
            y = 0;
            w = 0;
            h = 0;
        }

        Dimension(CONST Dimension& dim)
        {
            x = dim.x;
            y = dim.y;
            w = dim.w;
            h = dim.h;
        }
    };

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

    class IGraphicsSettings;

    class IScreenElement
    {
    public:

        IScreenElement(VOID) {}

        virtual Dimension VGetDimension(VOID) = 0;

        virtual VOID VSetDimension(CONST Dimension& dim) = 0;

        virtual UINT VGetPosX(VOID) CONST = 0;

        virtual UINT VGetPosY(VOID) CONST = 0;

        virtual UINT VGetWidth(VOID) CONST = 0;

        virtual UINT VGetHeight(VOID) CONST = 0;

        virtual LPCSTR VGetName(VOID) CONST = 0;

        virtual VOID VSetName(LPCSTR name) = 0;

        virtual VOID VDraw(VOID) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual BOOL VIsEnable(VOID) CONST = 0;

        virtual VOID VSetEnable(BOOL enable) = 0;

        virtual BOOL VIsActive(VOID) CONST = 0;

        virtual VOID VSetActive(BOOL active) = 0;

        virtual VOID VUpdate(ULONG millis) = 0;

        virtual VOID VSetAlpha(FLOAT alpha) = 0;

        virtual VOID VSetBackgroundColor(FLOAT r, FLOAT g, FLOAT b) = 0;

        virtual CONST util::Color& VGetBackgroundColor(VOID) CONST = 0;

        virtual FLOAT VGetAlpha(VOID) CONST = 0;

        virtual ~IScreenElement(VOID) {}
    };

    class ScreenElement : public IScreenElement
    {
    protected:
        _Percentages m_dimensionPercent;
        Dimension m_dimension;
        util::Color m_color;
        BOOL m_isEnable;
        BOOL m_isActive;
        std::string m_name;

    public:
        ScreenElement(VOID);

        LPCSTR VGetName(VOID) CONST;

        VOID VSetName(LPCSTR name);

        virtual VOID VSetDimension(CONST Dimension& dim);

        Dimension VGetDimension(VOID);

        BOOL VIsEnable(VOID) CONST;

        virtual VOID VSetEnable(BOOL enable);

        BOOL VIsActive(VOID) CONST;

        virtual VOID VSetActive(BOOL active);

        UINT VGetPosX(VOID) CONST;

        UINT VGetPosY(VOID) CONST;

        UINT VGetWidth(VOID) CONST;

        UINT VGetHeight(VOID) CONST;

        FLOAT VGetAlpha(VOID) CONST;

        CONST util::Color& VGetBackgroundColor(VOID) CONST;

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

    class RenderScreen : public ScreenElement
    {
    protected:
        std::shared_ptr<IGraphicsSettings> m_pSettings;

    public:
        RenderScreen(std::shared_ptr<IGraphicsSettings> settings);

        virtual BOOL VOnRestore(VOID);

        std::shared_ptr<IGraphicsSettings> GetSettings(VOID) { return m_pSettings; }

        virtual VOID VDraw(VOID);

        ~RenderScreen(VOID);
    };

    class RendertargetScreen : public ScreenElement
    {
    private:
        d3d::RenderTarget* m_pTarget;
    public:
        RendertargetScreen(d3d::RenderTarget* target);

        VOID VDraw(VOID);

        ~RendertargetScreen(VOID);
    };

    class DefShaderRenderScreen : public RenderScreen
    {
    private:
        UINT m_target;
    public:
        DefShaderRenderScreen(UINT target);

        VOID VDraw(VOID);

        ~DefShaderRenderScreen(VOID);
    };

    class DefShaderRenderScreenContainer : public ScreenElement
    {
    private:
        std::list<DefShaderRenderScreen*> m_screens;
        std::shared_ptr<IGraphicsSettings> m_settings;

    public:
        DefShaderRenderScreenContainer(std::shared_ptr<IGraphicsSettings> settings);

        VOID AddComponent(DefShaderRenderScreen* screen);

        VOID VDraw(VOID);

        BOOL VOnRestore(VOID);

        ~DefShaderRenderScreenContainer(VOID);
    };
}
