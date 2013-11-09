#include "tbdFont.h"
#include <fstream>
#include "util.h"
#include "GeometryFactory.h"

namespace chimera
{
    /*VOID IFont::Print(VOID) CONST
    {
        std::stringstream ss;
        ss << "Size=" << VGetSize() << ", File="  << VGetFileName() << ", LineHeight=" << VGetLineHeight() << ", TexWidth=" << VGetTextureWidth();
        ss << ", TexHeight=" << VGetTextureHeight() << ", Italic=" << VIsItalic() << ", Bold=" << VIsBold();
        DEBUG_OUT(ss.str().c_str());
    } */

    BMFont::BMFont(VOID) : m_initialized(FALSE), m_pFontTexture(NULL)
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

        CMTextureDescription desc;
        ZeroMemory(&desc, sizeof(CMTextureDescription));

        desc.data = data;
        desc.format = eFormat_R8G8B8A8_UNORM;
        desc.width = map->GetWidth();
        desc.height = map->GetHeight();

        m_pFontTexture = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateTexture(&desc).release();

        m_pFontTexture->VCreate();

        SAFE_DELETE(map);
        SAFE_DELETE(data);
        return TRUE;
    }

    CONST CMFontStyle& BMFont::VGetStyle(VOID) CONST
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

    CONST CMCharMetric* BMFont::VGetCharMetric(UCHAR c) CONST
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
       CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture(chimera::eGuiSampler, m_pFontTexture);
    }

    BMFont::~BMFont(VOID)
    {
        m_pFontTexture->VDestroy();
        SAFE_DELETE(m_pFontTexture);
    }

    VOID FontManager::VAddFont(CONST std::string& name, IFont* font)
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

    IFont* FontManager::VGetFont(CONST std::string& name) CONST
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

    VOID FontManager::VRemoveFont(CONST std::string& name)
    {
        m_fonts.erase(name);
    }

    VOID FontManager::VRenderText(std::string& text, FLOAT x, FLOAT y, chimera::Color* color)
    {
        VRenderText(text.c_str(), x, y, color);
    }

    VOID FontManager::VRenderText(LPCSTR text, FLOAT x, FLOAT y, chimera::Color* color)
    {
        m_pCurrentRenderer->VRenderText(text, m_pCurrentFont, x, y, color);
    }

    VOID FontManager::VActivateFont(CONST std::string& name)
    {
        VGetFont(name)->VActivate();
    }

    BOOL FontManager::VOnRestore(VOID)
    {
        return m_pCurrentRenderer->VOnRestore();
    }

    VOID FontManager::VSetFontRenderer(IFontRenderer* renderer)
    {
        m_pCurrentRenderer = renderer;
    }

    IFont* FontManager::VGetCurrentFont(VOID)
    {
        return m_pCurrentFont;
    }

    FontManager::~FontManager(VOID)
    {
        for(auto it = m_fonts.begin(); it != m_fonts.end(); ++it)
        {
            delete it->second;
        }
        SAFE_DELETE(m_pCurrentRenderer);
    }

    FontRenderer::FontRenderer(VOID) : m_quad(NULL), m_program(NULL)
    {
 
    }

    BOOL FontRenderer::VOnRestore(VOID)
    {
        Destroy();

        CMShaderProgramDescription shaderDesc;
        shaderDesc.vs.file = L"Gui.hlsl";
        shaderDesc.vs.function = "Font_VS";

        shaderDesc.vs.layoutCount = 2;

        shaderDesc.vs.inputLayout[0].instanced = FALSE;
        shaderDesc.vs.inputLayout[0].name = "POSITION";
        shaderDesc.vs.inputLayout[0].position = 0;
        shaderDesc.vs.inputLayout[0].slot = 0;
        shaderDesc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

        shaderDesc.vs.inputLayout[1].instanced = FALSE;
        shaderDesc.vs.inputLayout[1].name = "TEXCOORD";
        shaderDesc.vs.inputLayout[1].position = 1;
        shaderDesc.vs.inputLayout[1].slot = 0;
        shaderDesc.vs.inputLayout[1].format = eFormat_R32G32_FLOAT;

        shaderDesc.fs.file = L"Gui.hlsl";
        shaderDesc.fs.function = "Font_PS";

        m_program = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("Font", &shaderDesc);

        m_quad = geometryfactroy::CreateScreenQuad(TRUE);//CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry().release();

/*
        FLOAT vertices[20] = {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
        UINT indices[4] = {0,1,2,3};

        m_quad->VSetIndexBuffer(indices, 4);
        m_quad->VSetTopology(eTopo_TriangleStrip);
        m_quad->VSetVertexBuffer(vertices, 20, 5 * sizeof(FLOAT), TRUE);
        m_quad->VCreate();*/

        return TRUE;
    }

    VOID FontRenderer::Destroy(VOID)
    {
        if(m_quad)
        {
            m_quad->VDestroy();
        }
        SAFE_DELETE(m_quad);
    }

    FontRenderer::~FontRenderer(VOID)    
    {
        Destroy();
    }

    //TODO: refactoring in terms of performance
    VOID FontRenderer::VRenderText(CONST std::string& text, IFont* font, FLOAT x, FLOAT y, chimera::Color* color)
    {
        if(x < 0 || x > 1 || y < 0 || y > 1)
        {
            return;
        }

        font->VActivate();

        IConstShaderBuffer* buffer = CmGetApp()->VGetRenderer()->VGetConstShaderBuffer(chimera::eGuiColorBuffer);

        XMFLOAT4* c = (XMFLOAT4*)buffer->VMap();
        c->x = color ? color->r : 1;
        c->y = color ? color->g : 1;
        c->z = color ? color->b : 1;
        c->w = 0;
        buffer->VUnmap();

        CONST CHAR* str = text.c_str();
        m_program->VBind();

        UINT curserX = 0;
        UINT curserY = 0;
        
        //chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateBlendAlpha, NULL, 0xffffff);
        CmGetApp()->VGetRenderer()->VPushAlphaBlendState();

        FLOAT w = (FLOAT) CmGetApp()->VGetWindowWidth();
        FLOAT h = (FLOAT) CmGetApp()->VGetWindowHeight();

        for(UINT i = 0; i < text.size(); ++i)
        {
            CHAR c = str[i];

            //assert (c < (s_charCount+32) && c > 31);
            CONST CMCharMetric* metric = font->VGetCharMetric(c);
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

                m_quad->VGetVertexBuffer()->VSetData(localVertices, 20 * sizeof(FLOAT));
                m_quad->VBind();
                m_quad->VDraw();
            }
            else
            {
                LOG_CRITICAL_ERROR("metric not found for character: " + c);
            }
        }

        CmGetApp()->VGetRenderer()->VPopBlendState();
        //chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateNoBlending, NULL, 0xffffff);
    }
}