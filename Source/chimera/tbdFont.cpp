#include "tbdFont.h"
#include "GameApp.h"
#include "util.h"
#include "GeometryFactory.h"
#include "ShaderProgram.h"
#include "D3DRenderer.h"
#include <fstream>
#include <DWrite.h>
#include "d3d.h"
#include "Geometry.h"
#include "Texture.h"
#include "D3DRenderer.h"
#include "GameView.h"
namespace tbd
{
    VOID IFont::Print(VOID) CONST
    {
        std::stringstream ss;
        ss << "Size=" << VGetSize() << ", File="  << VGetFileName() << ", LineHeight=" << VGetLineHeight() << ", TexWidth=" << VGetTextureWidth();
        ss << ", TexHeight=" << VGetTextureHeight() << ", Italic=" << VIsItalic() << ", Bold=" << VIsBold();
        DEBUG_OUT(ss.str().c_str());
    }

    BMFont::BMFont(VOID) : m_initialized(FALSE), m_fontTexture(NULL)
    {

    }

    BOOL BMFont::VCreate(CONST std::string& file)
    {
        if(m_initialized)
        {
            return TRUE;
        }

        m_style.metricFile = file;

        m_initialized = TRUE;

        std::ifstream metricsStream(file);
        if(!metricsStream.good())
        {
            return FALSE;
        }
        //UINT pos = 0;
        while(metricsStream.good())
        {
            std::string s;
            std::getline(metricsStream, s);
            std::vector<std::string> ss = util::split(s, ' ');
            if(ss.size() == 0) 
            {
                continue;
            }
            if(ss[0].compare("info") == 0)
            {
                m_style.size = atoi((util::split(ss[2], '=')[1]).c_str());
                m_style.name = util::split(ss[1], '=')[1];
                m_style.bold = atoi((util::split(ss[3], '=')[1]).c_str());
                m_style.italic = atoi((util::split(ss[4], '=')[1]).c_str());
            }
            else if(ss[0].compare("common") == 0)
            {   
                m_style.lineHeight = atoi((util::split(ss[1], '=')[1]).c_str());
                m_style.base = atoi((util::split(ss[2], '=')[1]).c_str());
                m_style.texWidth = atoi((util::split(ss[3], '=')[1]).c_str());
                m_style.texHeight = atoi((util::split(ss[4], '=')[1]).c_str());
            }
            else if(ss[0].compare("page") == 0)
            {
                m_style.textureFile = util::split(ss[2], '=')[1];
            }
            else if(ss[0].compare("chars") == 0)
            {
                m_style.charCount = atoi((util::split(ss[1], '=')[1]).c_str());
            }
            else if(ss[0].compare("char") == 0)
            {
                std::vector<std::string> tokens;
                for(UINT i  = 1; i < ss.size(); ++i)
                {
                    if(ss[i].compare(""))
                    {
                        std::vector<std::string> split = util::split(ss[i], '=');
                        tokens.push_back(split[1]);
                    }
                }
                UCHAR c = (UCHAR)atoi(tokens[0].c_str());
                m_metrics[c].id = c;
                m_metrics[c].x = atoi(tokens[1].c_str());
                m_metrics[c].y = atoi(tokens[2].c_str());
                m_metrics[c].width = atoi(tokens[3].c_str());
                m_metrics[c].height = atoi(tokens[4].c_str());
                m_metrics[c].xoffset = atoi(tokens[5].c_str());
                m_metrics[c].yoffset = atoi(tokens[6].c_str());
                m_metrics[c].xadvance = atoi(tokens[7].c_str());
            }
        }
        metricsStream.close();
        Gdiplus::Bitmap* map = util::GetBitmapFromFile(util::string2wstring(VGetStyle().textureFile).c_str());
        if(map->GetLastStatus())
        {
            return FALSE;
        }
        CHAR* data = util::GetTextureData(map);

        m_fontTexture = new d3d::Texture2D(data, map->GetWidth(), map->GetHeight(), DXGI_FORMAT_R8G8B8A8_UNORM);
        m_fontTexture->SetBindflags(D3D11_BIND_SHADER_RESOURCE);
        m_fontTexture->VCreate();

        delete map;
        delete data;
        return TRUE;
    }

    CONST FontStyle& BMFont::VGetStyle(VOID) CONST
    {
        return m_style;
    }

    BOOL BMFont::VIsBold(VOID) CONST
    {
        return m_style.bold;
    }

    BOOL BMFont::VIsItalic(VOID) CONST
    {
        return m_style.italic;
    }

    UINT BMFont::VGetSize(VOID) CONST
    {
        return m_style.size;
    }

    CONST CharMetric* BMFont::VGetCharMetric(UCHAR c) CONST
    {
        auto it = m_metrics.find(c);
//#ifdef _DEBUG
        if(it == m_metrics.end())
        {
            //std::string s = "Glpyhmetrics not found for character: " + c;
            //LOG_ERROR(s);
            return NULL;
        }
//#endif
        return &it->second;
    }

    UINT BMFont::VGetLineHeight(VOID) CONST
    {
        return m_style.lineHeight;
    }

    LPCSTR BMFont::VGetFileName(VOID) CONST
    {
        return m_style.metricFile.c_str();
    }

    UINT BMFont::VGetCharCount(VOID) CONST
    {
        return m_style.charCount;
    }

    UINT BMFont::VGetTextureWidth(VOID) CONST
    {
        return m_style.texWidth;
    }

    UINT BMFont::VGetTextureHeight(VOID) CONST
    {
        return m_style.texHeight;
    }

    UINT BMFont::VGetBase(VOID) CONST
    {
        return m_style.base;
    }

    VOID BMFont::VActivate(VOID)
    {
       app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eGuiSampler, m_fontTexture->GetShaderResourceView());
    }

    BMFont::~BMFont(VOID)
    {
        SAFE_DELETE(m_fontTexture);
    }

    VOID FontManager::AddFont(CONST std::string& name, IFont* font)
    {

#ifdef _DEBUG
        if(m_fonts.find(name) != m_fonts.end())
        {
            LOG_CRITICAL_ERROR_A("Font with name '%s' already exists!", name.c_str());
        }
        else
#endif
        {
            m_fonts[name] = font;
        }

        if(!m_pCurrentFont)
        {
            m_pCurrentFont = font;
            m_pCurrentFont->VActivate();
        }
    }

    FontManager::FontManager(VOID) : m_pCurrentFont(NULL)
    {

    }

    IFont* FontManager::GetFont(CONST std::string& name) CONST
    {
        auto it = m_fonts.find(name);
#ifdef _DEBUG
        if(it == m_fonts.end())
        {
            LOG_CRITICAL_ERROR_A("Font %s does not exist!", name.c_str());
        }
#endif
        return it->second;
    }

    VOID FontManager::RemoveFont(CONST std::string& name)
    {
        m_fonts.erase(name);
    }

    VOID FontManager::RenderText(std::string& text, FLOAT x, FLOAT y, util::Color* color)
    {
        RenderText(text.c_str(), x, y, color);
    }

    VOID FontManager::RenderText(LPCSTR text, FLOAT x, FLOAT y, util::Color* color)
    {
        m_pCurrentRenderer->VRenderText(text, m_pCurrentFont, x, y, color);
    }

    VOID FontManager::ActivateFont(CONST std::string& name)
    {
        GetFont(name)->VActivate();
    }

    VOID FontManager::SetFontRenderer(IFontRenderer* renderer)
    {
        m_pCurrentRenderer = renderer;
    }

    IFont* FontManager::GetCurrentFont(VOID)
    {
        return m_pCurrentFont;
    }

    FontManager::~FontManager(VOID)
    {
        for(auto it = m_fonts.begin(); it != m_fonts.end(); ++it)
        {
            delete it->second;
        }
    }

    D3DFontRenderer::D3DFontRenderer(VOID) : m_quad(NULL)
    {
        m_program = d3d::ShaderProgram::CreateProgram("Font", L"Gui.hlsl", "Font_VS", "Font_PS", NULL);
        m_program->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        m_program->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        m_program->GenerateLayout();
        m_program->Bind();
        m_program = d3d::ShaderProgram::GetProgram("Font");
        m_quad = GeometryFactory::GetGlobalScreenQuadCPU();
    }

    D3DFontRenderer::~D3DFontRenderer(VOID)
    {

    }

    //TODO: refactoring in terms of performance
    VOID D3DFontRenderer::VRenderText(CONST std::string& text, IFont* font, FLOAT x, FLOAT y, util::Color* color)
    {
        if(x < 0 || x > 1 || y < 0 || y > 1)
        {
            return;
        }

        font->VActivate();

        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eGuiColorBuffer);

        XMFLOAT4* c = (XMFLOAT4*)buffer->Map();
        c->x = color ? color->r : 1;
        c->y = color ? color->g : 1;
        c->z = color ? color->b : 1;
        c->w = 0;
        buffer->Unmap();

        CONST CHAR* str = text.c_str();
        m_program->Bind();

        UINT curserX = 0;
        UINT curserY = 0;
        
        d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateBlendAlpha, NULL, 0xffffff);

        FLOAT w = (FLOAT) d3d::g_width;
        FLOAT h = (FLOAT) d3d::g_height;

        for(UINT i = 0; i < text.size(); ++i)
        {
            CHAR c = str[i];

            //assert (c < (s_charCount+32) && c > 31);
            CONST CharMetric* metric = font->VGetCharMetric(c);
            if(metric != NULL || c == '\n')
            {
                if(c == '\n')
                {
                    curserY -= font->VGetLineHeight();
                    curserX = 0;
                    continue;
                }

                FLOAT u0 = metric->x / (FLOAT)font->VGetTextureWidth();
                FLOAT v0 = 1 - metric->y/ (FLOAT)font->VGetTextureHeight();
                FLOAT u1 = (metric->x + metric->width ) / (FLOAT)font->VGetTextureWidth();
                FLOAT v1 = 1 - (metric->y + metric->height) / (FLOAT)font->VGetTextureHeight();

                UINT quadPosX = (UINT)(x * w);
                UINT quadPosY = (UINT)((1-y) * h);

                quadPosX += curserX + metric->xoffset;
                quadPosY -= metric->yoffset - curserY;
                
                FLOAT nposx = 2.0f * quadPosX / w - 1.0f;
                FLOAT nposy = 2.0f * quadPosY / h - 1.0f;

                FLOAT nposx1 = 2.0f * (quadPosX + metric->width) / w - 1.0f;
                FLOAT nposy1 = 2.0f * (quadPosY - metric->height) / h - 1.0f;

                //DEBUG_OUT(c);
                //DEBUG_OUT_A("%d, %d,", metric->xoffset, metric->yoffset);

                FLOAT localVertices[20] = 
                {
                    nposx,  nposy1, 0, u0, v1,
                    nposx1, nposy1, 0, u1, v1,
                    nposx,  nposy, 0, u0, v0,
                    nposx1, nposy, 0, u1, v0,
                };

                /*
                FLOAT localVertices[20] = 
                {
                   -1, -1, 0, u0, v0,
                    1, -1, 0, u1, v0,
                    -1, 1, 0, u0, v1,
                    1, 1, 0, u1, v1,
                }; */

                curserX += metric->xadvance;

                D3D11_MAPPED_SUBRESOURCE* ress = m_quad->GetVertexBuffer()->Map();
                memcpy(ress->pData, localVertices, 20 * sizeof(FLOAT));
                m_quad->GetVertexBuffer()->Unmap();
                m_quad->Bind();
                m_quad->Draw();
            }
            else
            {
                LOG_CRITICAL_ERROR("metric not found for character: " + c);
            }
        }

        d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateNoBlending, NULL, 0xffffff);
    }
}