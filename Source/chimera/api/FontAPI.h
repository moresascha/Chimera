#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IFont
    {
    public:
        virtual bool VCreate(const std::string& file) = 0;

        virtual bool VIsBold(void) const = 0;

        virtual bool VIsItalic(void) const = 0;

        virtual void VActivate(void) = 0;

        virtual uint VGetLineHeight(void) const = 0;

        virtual uint VGetSize(void) const = 0;

        virtual uint VGetCharCount (void) const = 0;

        virtual LPCSTR VGetFileName(void) const = 0;

        virtual uint VGetTextureWidth(void) const = 0;

        virtual uint VGetTextureHeight(void) const = 0;

        virtual uint VGetBase(void) const = 0;

        virtual const CMFontStyle& VGetStyle(void) const = 0;

        virtual const CMCharMetric* VGetCharMetric(UCHAR c) const = 0;

        virtual ~IFont(void) {}
    };

    class IFontRenderer
    {
    public:
        virtual void VRenderText(const std::string& text, IFont* font, float x, float y, chimera::Color* color = NULL) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual ~IFontRenderer(void) { }
    };

    class IFontManager
    {
    public:
        virtual void VSetFontRenderer(IFontRenderer* renderer) = 0;

        virtual void VRenderText(std::string& text, float x, float y, chimera::Color* color = NULL) = 0;

        virtual void VRenderText(LPCSTR text, float x, float y, chimera::Color* color = NULL) = 0;

        virtual void VAddFont(const std::string& name, IFont* font) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual void VActivateFont(const std::string& name) = 0;

        virtual void VRemoveFont(const std::string& name) = 0;

        virtual IFont* VGetCurrentFont(void) = 0;

        virtual IFont* VGetFont(const std::string& name) const = 0;

        virtual ~IFontManager(void) {}
    };

    class IFontFactory
    {
    public:
        virtual IFont* VCreateFont(void) = 0;

        virtual IFontRenderer* VCreateFontRenderer(void) = 0;

        virtual IFontManager* VCreateFontManager(void) = 0;
    };
}