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
        bool m_initialized;
    public:
        BMFont(void);

        bool VCreate(const std::string& file);

        void VDestroy(void);

        void VActivate(void);

        bool VIsBold(void) const;

        bool VIsItalic(void) const;

        uint VGetLineHeight(void);

        uint VGetLineHeight(void) const;

        uint VGetSize(void) const;

        uint VGetCharCount (void) const;

        uint VGetBase(void) const;

        LPCSTR VGetFileName(void) const;

        uint VGetTextureWidth(void) const;

        uint VGetTextureHeight(void) const;

        const CMFontStyle& VGetStyle(void) const;

        const CMCharMetric* VGetCharMetric(UCHAR c) const;

        ~BMFont(void);
    };

    class FontRenderer : public IFontRenderer
    {
        IShaderProgram* m_program;
        IGeometry* m_quad;
    public:
        FontRenderer(void);

        void VRenderText(const std::string& text, IFont* font, float x, float y, chimera::Color* color = NULL);

        bool VOnRestore(void);

        void Destroy(void);

        ~FontRenderer(void);
    };

    class FontManager : public IFontManager
    {
    private:
        std::map<std::string, IFont*> m_fonts;
        IFont* m_pCurrentFont;
        IFontRenderer* m_pCurrentRenderer;
    public:
        FontManager(void);

        bool VOnRestore(void);

        void VSetFontRenderer(IFontRenderer* renderer);

        void VRenderText(std::string& text, float x, float y, chimera::Color* color = NULL);

        void VRenderText(LPCSTR text, float x, float y, chimera::Color* color = NULL);

        void VAddFont(const std::string& name, IFont* font);

        void VActivateFont(const std::string& name);

        void VRemoveFont(const std::string& name);

        IFont* VGetCurrentFont(void);

        IFont* VGetFont(const std::string& name) const;

        ~FontManager(void);
    };
}