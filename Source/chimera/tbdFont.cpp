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

    BMFont::BMFont(void) : m_initialized(false), m_pFontTexture(NULL)
    {

    }

    bool BMFont::VCreate(const std::string& file)
    {
        if(m_initialized)
        {
            return true;
        }

        m_style.metricFile = file;

        m_initialized = true; 

        std::ifstream metricsStream(file);
        if(!metricsStream.good())
        {
            return false;
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
                m_style.bold = atoi((util::split(ss[3], '=')[1]).c_str()) != 0;
                m_style.italic = atoi((util::split(ss[4], '=')[1]).c_str()) != 0;
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
                for(uint i  = 1; i < ss.size(); ++i)
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
        std::string texFile = CmGetApp()->VGetConfig()->VGetString("sResCache") + VGetStyle().textureFile;
        Gdiplus::Bitmap* map = util::GetBitmapFromFile(util::string2wstring(texFile).c_str());
        if(map->GetLastStatus())
        {
            return false;
        }
        char* data = util::GetTextureData(map);

        CMTextureDescription desc;
        ZeroMemory(&desc, sizeof(CMTextureDescription));

        desc.data = data;
        desc.format = eFormat_R8G8B8A8_UNORM;
        desc.width = map->GetWidth();
        desc.height = map->GetHeight();
        desc.miscflags = eTextureMiscFlags_BindShaderResource;

        m_pFontTexture = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateTexture(&desc).release();

        m_pFontTexture->VCreate();

        SAFE_DELETE(map);
        SAFE_DELETE(data);
        return true;
    }

    const CMFontStyle& BMFont::VGetStyle(void) const
    {
        return m_style;
    }

    bool BMFont::VIsBold(void) const
    {
        return m_style.bold;
    }

    bool BMFont::VIsItalic(void) const
    {
        return m_style.italic;
    }

    uint BMFont::VGetSize(void) const
    {
        return m_style.size;
    }

    const CMCharMetric* BMFont::VGetCharMetric(UCHAR c) const
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

    uint BMFont::VGetLineHeight(void) const
    {
        return m_style.lineHeight;
    }

    LPCSTR BMFont::VGetFileName(void) const
    {
        return m_style.metricFile.c_str();
    }

    uint BMFont::VGetCharCount(void) const
    {
        return m_style.charCount;
    }

    uint BMFont::VGetTextureWidth(void) const
    {
        return m_style.texWidth;
    }

    uint BMFont::VGetTextureHeight(void) const
    {
        return m_style.texHeight;
    }

    uint BMFont::VGetBase(void) const
    {
        return m_style.base;
    }

    void BMFont::VActivate(void)
    {
       CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture(chimera::eGuiSampler, m_pFontTexture);
    }

    BMFont::~BMFont(void)
    {
        if(m_pFontTexture)
        {
            m_pFontTexture->VDestroy();
        }
        SAFE_DELETE(m_pFontTexture);
    }

    void FontManager::VAddFont(const std::string& name, IFont* font)
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

    FontManager::FontManager(void) : m_pCurrentFont(NULL)
    {

    }

    IFont* FontManager::VGetFont(const std::string& name) const
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

    void FontManager::VRemoveFont(const std::string& name)
    {
        m_fonts.erase(name);
    }

    void FontManager::VRenderText(std::string& text, float x, float y, chimera::Color* color)
    {
        VRenderText(text.c_str(), x, y, color);
    }

    void FontManager::VRenderText(LPCSTR text, float x, float y, chimera::Color* color)
    {
        m_pCurrentRenderer->VRenderText(text, m_pCurrentFont, x, y, color);
    }

    void FontManager::VActivateFont(const std::string& name)
    {
        VGetFont(name)->VActivate();
    }

    bool FontManager::VOnRestore(void)
    {
        return m_pCurrentRenderer->VOnRestore();
    }

    void FontManager::VSetFontRenderer(IFontRenderer* renderer)
    {
        m_pCurrentRenderer = renderer;
    }

    IFont* FontManager::VGetCurrentFont(void)
    {
        return m_pCurrentFont;
    }

    FontManager::~FontManager(void)
    {
        for(auto it = m_fonts.begin(); it != m_fonts.end(); ++it)
        {
            delete it->second;
        }
        SAFE_DELETE(m_pCurrentRenderer);
    }

    FontRenderer::FontRenderer(void) : m_quad(NULL), m_program(NULL)
    {
 
    }

    bool FontRenderer::VOnRestore(void)
    {
        Destroy();

        CMShaderProgramDescription shaderDesc;
        shaderDesc.vs.file = L"Gui.hlsl";
        shaderDesc.vs.function = "Font_VS";

        shaderDesc.vs.layoutCount = 2;

        shaderDesc.vs.inputLayout[0].instanced = false;
        shaderDesc.vs.inputLayout[0].name = "POSITION";
        shaderDesc.vs.inputLayout[0].position = 0;
        shaderDesc.vs.inputLayout[0].slot = 0;
        shaderDesc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

        shaderDesc.vs.inputLayout[1].instanced = false;
        shaderDesc.vs.inputLayout[1].name = "TEXCOORD";
        shaderDesc.vs.inputLayout[1].position = 1;
        shaderDesc.vs.inputLayout[1].slot = 0;
        shaderDesc.vs.inputLayout[1].format = eFormat_R32G32_FLOAT;

        shaderDesc.fs.file = L"Gui.hlsl";
        shaderDesc.fs.function = "Font_PS";

        m_program = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("Font", &shaderDesc);

        m_quad = geometryfactroy::CreateScreenQuad(true);//CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry().release();

/*
        FLOAT vertices[20] = {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
        UINT indices[4] = {0,1,2,3};

        m_quad->VSetIndexBuffer(indices, 4);
        m_quad->VSetTopology(eTopo_TriangleStrip);
        m_quad->VSetVertexBuffer(vertices, 20, 5 * sizeof(FLOAT), TRUE);
        m_quad->VCreate();*/

        return true;
    }

    void FontRenderer::Destroy(void)
    {
        if(m_quad)
        {
            m_quad->VDestroy();
        }
        SAFE_DELETE(m_quad);
    }

    FontRenderer::~FontRenderer(void)    
    {
        Destroy();
    }

    //TODO: refactoring in terms of performance
    void FontRenderer::VRenderText(const std::string& text, IFont* font, float x, float y, chimera::Color* color)
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

        const char* str = text.c_str();
        m_program->VBind();

        uint curserX = 0;
        uint curserY = 0;
        
        //chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateBlendAlpha, NULL, 0xffffff);
        CmGetApp()->VGetRenderer()->VPushAlphaBlendState();

        float w = (float) CmGetApp()->VGetWindowWidth();
        float h = (float) CmGetApp()->VGetWindowHeight();

        for(uint i = 0; i < text.size(); ++i)
        {
            char c = str[i];

            //assert (c < (s_charCount+32) && c > 31);
            const CMCharMetric* metric = font->VGetCharMetric(c);
            if(metric != NULL || c == '\n')
            {
                if(c == '\n')
                {
                    curserY -= font->VGetLineHeight();
                    curserX = 0;
                    continue;
                }

                float u0 = metric->x / (float)font->VGetTextureWidth();
                float v0 = 1 - metric->y/ (float)font->VGetTextureHeight();
                float u1 = (metric->x + metric->width ) / (float)font->VGetTextureWidth();
                float v1 = 1 - (metric->y + metric->height) / (float)font->VGetTextureHeight();

                uint quadPosX = (uint)(x * w);
                uint quadPosY = (uint)((1-y) * h);

                //fixme: offset of -1 produces wrong results if x is 0
                quadPosX += curserX + metric->xoffset;
                quadPosY -= metric->yoffset - curserY;
                
                float nposx = 2.0f * quadPosX / w - 1.0f;
                float nposy = 2.0f * quadPosY / h - 1.0f;

                float nposx1 = 2.0f * (quadPosX + metric->width) / w - 1.0f;
                float nposy1 = 2.0f * (quadPosY - metric->height) / h - 1.0f;

                //DEBUG_OUT(c);
                //DEBUG_OUT_A("%d, %d,", metric->xoffset, metric->yoffset);

                float localVertices[20] = 
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

                m_quad->VGetVertexBuffer()->VSetData(localVertices, 20 * sizeof(float));
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