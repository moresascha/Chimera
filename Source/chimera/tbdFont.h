#pragma once
#include "stdafx.h"
#include "Vec4.h"
namespace d3d
{
    class Texture2D;
    class ShaderProgram;
    class Geometry;
}

namespace tbd
{

    struct FontMetrics
    {
        FLOAT leftU;
        FLOAT rightU;
        UINT pixelWidth;
    };

    struct CharMetric
    {
        UCHAR id;
        UINT x;
        UINT y;
        UINT width;
        UINT height;
        INT xoffset;
        INT yoffset;
        UINT xadvance;
    };

    struct FontStyle
    {
        BOOL italic;
        BOOL bold;
        UINT charCount;
        UINT lineHeight;
        UINT texWidth;
        UINT texHeight;
        UINT size;
        UINT base;
        std::string textureFile;
        std::string metricFile;
        std::string name;
    };

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
        virtual CONST FontStyle& VGetStyle(VOID) CONST = 0;
        virtual CONST CharMetric* VGetCharMetric(UCHAR c) CONST = 0;
        VOID Print(VOID) CONST;
        virtual ~IFont(VOID) {}
    };

    class BMFont : public IFont
    {
    private:
        //GlyphMetric* m_metrics;
        std::map<UCHAR, CharMetric> m_metrics;
        FontStyle m_style;
        d3d::Texture2D* m_fontTexture;
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
        CONST FontStyle& VGetStyle(VOID) CONST;
        CONST CharMetric* VGetCharMetric(UCHAR c) CONST;
        ~BMFont(VOID);
    };

    class IFontRenderer
    {
    public:
        virtual VOID VRenderText(CONST std::string& text, IFont* font, FLOAT x, FLOAT y, util::Color* color = NULL) = 0;
        virtual ~IFontRenderer(VOID) { }
    };

    class D3DFontRenderer : public IFontRenderer
    {
        std::shared_ptr<d3d::ShaderProgram> m_program;
        d3d::Geometry* m_quad;
    public:
        D3DFontRenderer(VOID);
        VOID VRenderText(CONST std::string& text, IFont* font, FLOAT x, FLOAT y, util::Color* color = NULL);
        ~D3DFontRenderer(VOID);
    };

    class FontManager
    {
    private:
        std::map<std::string, IFont*> m_fonts;
        IFont* m_pCurrentFont;
        IFontRenderer* m_pCurrentRenderer;
    public:
        FontManager(VOID);
        VOID SetFontRenderer(IFontRenderer* renderer);
        VOID RenderText(std::string& text, FLOAT x, FLOAT y, util::Color* color = NULL);
        VOID RenderText(LPCSTR text, FLOAT x, FLOAT y, util::Color* color = NULL);
        VOID AddFont(CONST std::string& name, IFont* font);
        VOID ActivateFont(CONST std::string& name);
        VOID RemoveFont(CONST std::string& name);
        IFont* GetCurrentFont(VOID);
        IFont* GetFont(CONST std::string& name) CONST;
        ~FontManager(VOID);
    };
}