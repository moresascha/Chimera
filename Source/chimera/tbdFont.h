#pragma once
#include "stdafx.h"
#include "Vec4.h"

namespace chimera
{

    /*
    lineHeight: is how much to move the cursor when going to the next line.

    base: this is the offset from the top of line, to where the base of each character is.

    scaleW and scaleH: This is the size of the texture. 

    pages: gives how many textures that are used for the font.

    id: is the character number in the ASCII table.

    x, y, width, and height: give the position and size of the character image in the texture.

    xoffset and yoffset: hold the offset with which to offset the cursor position when drawing the character image. Note, these shouldn't actually change the cursor position.

    xadvance: is how much the cursor position should be moved after each character.

    page: gives the texture where the character image is found.
    */
  
    class BMFont : public IFont
    {
    private:
        //GlyphMetric* m_metrics;
        std::map<UCHAR, CMCharMetric> m_metrics;
        CMFontStyle m_style;
        chimera::IDeviceTexture* m_pFontTexture;
        BOOL m_initialized;
    public:
        BMFont(VOID);

        BOOL VCreate(CONST std::string& file);

        VOID VDestroy(VOID);

        VOID VActivate(VOID);

        BOOL VIsBold(VOID) CONST;

        BOOL VIsItalic(VOID) CONST;

        UINT VGetLineHeight(VOID);

        UINT VGetLineHeight(VOID) CONST;

        UINT VGetSize(VOID) CONST;

        UINT VGetCharCount (VOID) CONST;

        UINT VGetBase(VOID) CONST;

        LPCSTR VGetFileName(VOID) CONST;

        UINT VGetTextureWidth(VOID) CONST;

        UINT VGetTextureHeight(VOID) CONST;

        CONST CMFontStyle& VGetStyle(VOID) CONST;

        CONST CMCharMetric* VGetCharMetric(UCHAR c) CONST;

        ~BMFont(VOID);
    };

    class FontRenderer : public IFontRenderer
    {
        IShaderProgram* m_program;
        IGeometry* m_quad;
    public:
        FontRenderer(VOID);

        VOID VRenderText(CONST std::string& text, IFont* font, FLOAT x, FLOAT y, chimera::Color* color = NULL);

        BOOL VOnRestore(VOID);

        VOID Destroy(VOID);

        ~FontRenderer(VOID);
    };

    class FontManager : public IFontManager
    {
    private:
        std::map<std::string, IFont*> m_fonts;
        IFont* m_pCurrentFont;
        IFontRenderer* m_pCurrentRenderer;
    public:
        FontManager(VOID);

        BOOL VOnRestore(VOID);

        VOID VSetFontRenderer(IFontRenderer* renderer);

        VOID VRenderText(std::string& text, FLOAT x, FLOAT y, chimera::Color* color = NULL);

        VOID VRenderText(LPCSTR text, FLOAT x, FLOAT y, chimera::Color* color = NULL);

        VOID VAddFont(CONST std::string& name, IFont* font);

        VOID VActivateFont(CONST std::string& name);

        VOID VRemoveFont(CONST std::string& name);

        IFont* VGetCurrentFont(VOID);

        IFont* VGetFont(CONST std::string& name) CONST;

        ~FontManager(VOID);
    };
}