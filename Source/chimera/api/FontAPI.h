#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IFont
    {
    public:
        virtual BOOL VCreate(CONST std::string& file) = 0;

        virtual BOOL VIsBold(VOID) CONST = 0;

        virtual BOOL VIsItalic(VOID) CONST = 0;

        virtual VOID VActivate(VOID) = 0;

        virtual UINT VGetLineHeight(VOID) CONST = 0;

        virtual UINT VGetSize(VOID) CONST = 0;

        virtual UINT VGetCharCount (VOID) CONST = 0;

        virtual LPCSTR VGetFileName(VOID) CONST = 0;

        virtual UINT VGetTextureWidth(VOID) CONST = 0;

        virtual UINT VGetTextureHeight(VOID) CONST = 0;

        virtual UINT VGetBase(VOID) CONST = 0;

        virtual CONST CMFontStyle& VGetStyle(VOID) CONST = 0;

        virtual CONST CMCharMetric* VGetCharMetric(UCHAR c) CONST = 0;

        virtual ~IFont(VOID) {}
    };

    class IFontRenderer
    {
    public:
        virtual VOID VRenderText(CONST std::string& text, IFont* font, FLOAT x, FLOAT y, chimera::Color* color = NULL) = 0;

        virtual ~IFontRenderer(VOID) { }
    };

    class IFontManager
    {
    public:
        virtual VOID VSetFontRenderer(IFontRenderer* renderer) = 0;

        virtual VOID VRenderText(std::string& text, FLOAT x, FLOAT y, chimera::Color* color = NULL) = 0;

        virtual VOID VRenderText(LPCSTR text, FLOAT x, FLOAT y, chimera::Color* color = NULL) = 0;

        virtual VOID VAddFont(CONST std::string& name, IFont* font) = 0;

        virtual VOID VActivateFont(CONST std::string& name) = 0;

        virtual VOID VRemoveFont(CONST std::string& name) = 0;

        virtual IFont* VGetCurrentFont(VOID) = 0;

        virtual IFont* VGetFont(CONST std::string& name) CONST = 0;

        virtual ~IFontManager(VOID) {}
    };
}