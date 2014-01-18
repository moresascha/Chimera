#pragma once
#include "CMTypes.h"
#include "CMCommon.h"

namespace chimera
{
    class IScreenElement
    {
    public:
        virtual const CMDimension& VGetDimension(void) = 0;

        virtual void VSetDimension(const CMDimension& dim) = 0;

        virtual uint VGetPosX(void) const = 0;

        virtual uint VGetPosY(void) const = 0;

        virtual uint VGetWidth(void) const = 0;

        virtual uint VGetHeight(void) const = 0;

        virtual LPCSTR VGetName(void) const = 0;

        virtual void VSetName(LPCSTR name) = 0;

        virtual void VDraw(void) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual bool VIsIn(uint x, uint y) = 0;

        virtual bool VIsActive(void) const = 0;

        virtual void VSetActive(bool active) = 0;

        virtual void VUpdate(ulong millis) = 0;

        virtual void VSetAlpha(float alpha) = 0;

        virtual void VSetBackgroundColor(float r, float g, float b) = 0;

        virtual const Color& VGetBackgroundColor(void) const = 0;

        virtual float VGetAlpha(void) const = 0;

        virtual ~IScreenElement(void) {}
    };

    class IScreenElementContainer : virtual public IScreenElement
    {
    public:
        virtual void VAddComponent(IScreenElement* cmp) = 0;

        virtual IScreenElement* VGetComponent(LPCSTR name) = 0;
    };

    class IRenderScreen : virtual public IScreenElement
    {
    public:
        virtual bool VOnRestore(void) = 0;

        virtual IGraphicsSettings* VGetSettings(void) = 0;

        virtual void VDraw(void) = 0;

        virtual ~IRenderScreen(void) {}
    };

    /*class IScreenElementFactroy
    {
    public:
        virtual IRenderScreen* VCreateRenderScreen(VOID) = 0;

        virtual IScreenElementContainer* VCreateScreenElementContainer(VOID) = 0;
    };*/
}