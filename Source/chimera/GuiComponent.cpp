#include "GuiComponent.h"
#include "GeometryFactory.h"
#include "ShaderProgram.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "GameView.h"
#include "tbdFont.h"
#include <algorithm>
#include "Commands.h"
#include "Input.h"
#include "VRamManager.h"
#include "Effect.h"
#include "math.h"

namespace tbd
{
    namespace gui
    {

        D3D_GUI::D3D_GUI(VOID) : m_pVertexShader(NULL)
        {

        }

        BOOL D3D_GUI::VOnRestore(VOID)
        {
            m_pVertexShader = d3d::VertexShader::CreateShader("GuiDefaultVertexShader", L"Gui.hlsl", "GuiDefault_VS");
            if(!m_pVertexShader->GetInputLayout())
            {
                m_pVertexShader->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
                m_pVertexShader->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
                m_pVertexShader->GenerateLayout();
            }
            return ScreenElementContainer::VOnRestore();
        }

        VOID D3D_GUI::VDraw(VOID)
        {
            m_pVertexShader->Bind();
            app::g_pApp->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);
            ScreenElementContainer::VDraw();
            m_pVertexShader->Unbind();
            app::g_pApp->GetRenderer()->PopRasterizerState();
        }

        D3D_GUI::~D3D_GUI(VOID)
        {
        }

        //---imp

        D3DGuiComponent::D3DGuiComponent() : m_shaderName("GuiDefault_PS"), m_pPixelShader(NULL)
        {
        }

        VOID D3DGuiComponent::SetEffect(LPCSTR pixelShader)
        {
            m_shaderName = pixelShader;
        }

        BOOL D3DGuiComponent::VOnRestore(VOID)
        {
            m_pPixelShader = d3d::PixelShader::CreateShader(m_shaderName.c_str(), L"Gui.hlsl", m_shaderName.c_str());
            return ScreenElement::VOnRestore();
        }

        GuiRectangle::GuiRectangle(VOID)
        {
            SetTextureCoords(0, 0, 1, 1);
        }

        VOID GuiRectangle::SetTextureCoords(FLOAT x, FLOAT y, FLOAT u, FLOAT v)
        {
            m_tx = x;
            m_ty = y;
            m_u = u;
            m_v = v;
        }

        VOID GuiRectangle::VDraw(VOID)
        {
            if(VGetAlpha() < 1)
            {
                app::g_pApp->GetHumanView()->GetRenderer()->PushBlendState(d3d::g_pBlendStateBlendAlpha);
            }

            FLOAT x = -1.0f + 2 * VGetPosX() / (FLOAT)d3d::g_width;
            FLOAT y = 1.0f - 2 * VGetPosY() / (FLOAT)d3d::g_height;
            FLOAT w = x + 2 * VGetWidth() / (FLOAT)d3d::g_width;
            FLOAT h = y - 2 * VGetHeight() / (FLOAT)d3d::g_height;
            FLOAT localVertices[20] = 
            {
                x, h, 0, m_tx, m_ty,
                w, h, 0, m_u, m_ty,
                x, y, 0, m_tx, m_v,
                w, y, 0, m_u, m_v,
            };

            d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eGuiColorBuffer);

            XMFLOAT4* color = (XMFLOAT4*)buffer->Map();
            color->x = m_color.r;
            color->y = m_color.g;
            color->z = m_color.b;
            color->w = m_color.a;
            buffer->Unmap();
#ifdef _DEBUG
            if(!m_pPixelShader)
            {
                LOG_CRITICAL_ERROR("GuiProgram not initialized, forgot to call OnRestore?");
            }
#endif
            m_pPixelShader->Bind();
            d3d::Geometry* quad = GeometryFactory::GetGlobalScreenQuadCPU();
            D3D11_MAPPED_SUBRESOURCE* ress = quad->GetVertexBuffer()->Map();
            memcpy(ress->pData, localVertices, 20 * sizeof(FLOAT));
            quad->GetVertexBuffer()->Unmap();
            quad->Bind();
            quad->Draw();

            if(VGetAlpha() < 1)
            {
                app::g_pApp->GetHumanView()->GetRenderer()->PopBlendState();
            }
        }

        GuiRectangle::~GuiRectangle(VOID)
        {

        }

        GuiTextureComponent::GuiTextureComponent(VOID)
        {
            SetEffect("GuiTexture_PS");
            SetTexture("default64x641.png");
        }

        VOID GuiTextureComponent::VDraw(VOID)
        {
            if(m_textureHandle)
            {
                if(m_textureHandle->IsReady())
                {
                    m_textureHandle->Update();
                    app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eGuiSampler, m_textureHandle->GetShaderResourceView());
                    GuiRectangle::VDraw();
                    app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eGuiSampler, NULL);
                }
                else
                {
                    std::shared_ptr<tbd::VRamHandle> handle = app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_resource);
                    m_textureHandle = std::static_pointer_cast<d3d::Texture2D>(handle);
                }
            }
            else
            {
                GuiRectangle::VDraw();
            }
        }

        VOID GuiTextureComponent::SetTexture(LPCSTR texFile)
        {
            m_resource = texFile;
            if(strcmp(texFile, "") == 0)
            {
                SetEffect("GuiDefault_PS");
            }
            else
            {
                SetEffect("GuiTexture_PS");
            }
        }

        BOOL GuiTextureComponent::VOnRestore(VOID)
        {
            std::shared_ptr<tbd::VRamHandle> handle = app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_resource);
            m_textureHandle = std::static_pointer_cast<d3d::Texture2D>(handle);
            return GuiRectangle::VOnRestore();
        }

        GuiSpriteComponent::GuiSpriteComponent(UINT tx, UINT ty, UINT du, UINT dv) : m_tx(tx), m_ty(ty), m_u(du), m_v(dv)
        {

        }

        BOOL GuiSpriteComponent::VOnRestore(VOID)
        {
            GuiTextureComponent::VOnRestore();
            UINT x = m_tx;
            UINT y = m_ty;
            UINT u = m_tx + m_u;
            UINT v = m_ty + m_v;
            SetTextureCoords(x / (FLOAT)m_textureHandle->GetDescription().Width, y / (FLOAT)m_textureHandle->GetDescription().Height,
                u / (FLOAT)m_textureHandle->GetDescription().Width, v / (FLOAT)m_textureHandle->GetDescription().Height);

            return TRUE;
        }

        VOID GuiSpriteComponent::VDraw(VOID)
        {
            app::g_pApp->GetHumanView()->GetRenderer()->PushBlendState(d3d::g_pBlendStateBlendAlpha);
            GuiTextureComponent::VDraw();
            app::g_pApp->GetHumanView()->GetRenderer()->PopBlendState();
        }

        GuiInputComponent::GuiInputComponent(VOID) : m_onReturnDeactivate(TRUE)
        {

        }

        VOID GuiInputComponent::SetOnReturnDeactivate(BOOL deactivate)
        {
            m_onReturnDeactivate = deactivate;
        }

        GuiInputComponent::~GuiInputComponent(VOID)
        {

        }

        GuiTextComponent::GuiTextComponent(VOID) : m_appendDir(eDown), m_alignment(eLeft), m_textColor(1,1,1,1)
        {
            SetTexture("");
        }

        CONST std::vector<TextLine> GuiTextComponent::GetTextLines() CONST
        {
            return m_textLines;
        }

        VOID GuiTextComponent::SetAlignment(Alignment alignment)
        {
            m_alignment = alignment;
        }

        VOID GuiTextComponent::SetAppendDirection(AppendDirection dir)
        {
            m_appendDir = dir;
        }

        VOID GuiTextComponent::AppendText(CONST std::string& text)
        {
            UINT maxHistory = 100; //todo
            if(m_textLines.size() >= maxHistory)
            {
                m_textLines.erase(m_textLines.begin());
            }
            TextLine line;
            line.text = text;
            for(INT i = 0; i < text.size(); ++i)
            {
                CONST CharMetric* metric = app::g_pApp->GetFontManager()->GetCurrentFont()->VGetCharMetric(text[i]);
                if(!metric)
                {
                    metric = app::g_pApp->GetFontManager()->GetCurrentFont()->VGetCharMetric('?');
                }
                line.width += metric->xadvance;
            }
            m_textLines.push_back(line);
        }

        BOOL GuiTextComponent::VOnRestore(VOID)
        {
            if(m_resource != "")
            {
                return GuiTextureComponent::VOnRestore();
            }
            else
            {
                return GuiRectangle::VOnRestore();
            }
        }

        VOID GuiTextComponent::DrawText(TextLine& line, INT x, INT y)
        {
            INT mx = x;
            if(m_alignment == eCenter)
            {
                mx -= (INT)(0.5 * line.width);
            }
            else if(m_alignment == eRight)
            {
                mx -= line.width + 5;
            }
            if(mx + line.width > VGetPosX() + VGetWidth())
            {
                //slow implementation, only do this when dimension change, for example. Todo
                INT w = mx + line.width;
                std::string text = line.text;
                while(w > (INT)(VGetPosX() + VGetWidth()))
                {
                    CHAR c = text.substr(text.size() - 1, text.size())[0];
                    w -= app::g_pApp->GetFontManager()->GetCurrentFont()->VGetCharMetric(c)->xadvance;
                    text = text.substr(0, text.size() - 1);
                }
                app::g_pApp->GetFontManager()->RenderText(text, mx / (FLOAT)d3d::g_width, y / (FLOAT)d3d::g_height, &m_textColor);
            }
            else
            {
                app::g_pApp->GetFontManager()->RenderText(line.text, mx / (FLOAT)d3d::g_width, y / (FLOAT)d3d::g_height, &m_textColor);
            }
        }

        VOID GuiTextComponent::ClearText(VOID)
        {
            m_textLines.clear();
        }

        VOID GuiTextComponent::VDraw(VOID)
        {
            GuiTextureComponent::VDraw();

            INT y = VGetPosY();
            
            INT lineheight = app::g_pApp->GetFontManager()->GetCurrentFont()->VGetLineHeight();

            /*if(m_appendDir == eUp)
            {
                y = y + VGetHeight();
            }
            else if(m_appendDir == eDown)
            {
                y += lineheight;
            } */
            
            INT x = VGetPosX();

            INT centerX = 0;

            if(m_alignment == eCenter)
            {
                x = x + (INT)(0.5 * VGetWidth());
            }
            else if(m_alignment == eRight)
            {
                x = x + VGetWidth();
            }

            if(m_appendDir == eUp)
            {
                for(INT i = (INT)m_textLines.size() - 1; i >= 0; --i)
                {
                    TextLine line = m_textLines[i];
                    DrawText(line, x + 2, VGetHeight() - y - lineheight - 2);
                    y += lineheight;
                    if((VGetHeight()-y-lineheight) < (INT)VGetPosY())
                    {
                        break;
                    }
                }
            }
            else
            {
                for(INT i = 0; i < m_textLines.size(); ++i)
                {
                    TextLine line = m_textLines[i];
                    DrawText(line, x + 2, y - 2);
                    y += lineheight;
                    if((y+lineheight) > (INT)(VGetPosY() + VGetHeight()))
                    {
                        break;
                    }
                }
            }
        }

        VOID GuiTextComponent::SetTextColor(CONST util::Vec4& color)
        {
            m_textColor = color;
        }

        GuiTextComponent::~GuiTextComponent(VOID)
        {

        }

        GuiTextInput::GuiTextInput(VOID) : m_time(0), m_drawCurser(FALSE), m_curserPos(0), m_textColor(1,1,1,1)
        {
            tbd::KeyboardButtonPressedListener l0 = fastdelegate::MakeDelegate(this, &GuiTextInput::ComputeInput);
            tbd::KeyboardButtonRepeatListener l1 = fastdelegate::MakeDelegate(this, &GuiTextInput::ComputeInput);
            AddKeyPressedListener(l0);
            AddKeyRepeatListener(l1);
        }

        BOOL GuiTextInput::VOnRestore(VOID)
        {
            D3DGuiComponent::VOnRestore();
            return TRUE;
        }

        VOID GuiTextInput::VUpdate(ULONG millis)
        {
            m_time += millis;
            if(m_time > 400)
            {
                m_drawCurser = !m_drawCurser;
                m_time = 0;
            }
        }

        VOID GuiTextInput::AddChar(CHAR c)
        {
            INT nextPos = m_curserPos + app::g_pApp->GetFontManager()->GetCurrentFont()->VGetCharMetric(c)->xadvance;
            if(nextPos > (INT)(VGetWidth())-10)
            {
                return;
            }
            m_curserPos = nextPos;
            m_textLine += c;
        }

        VOID GuiTextInput::RemoveChar(VOID)
        {
            if(m_textLine.size() > 0)
            {
                CHAR c = m_textLine.substr(m_textLine.size() - 1, m_textLine.size())[0];
                m_textLine = m_textLine.substr(0, m_textLine.size() - 1);
                m_curserPos -= app::g_pApp->GetFontManager()->GetCurrentFont()->VGetCharMetric(c)->xadvance; //todo
            }
        }

        VOID GuiTextInput::SetText(CONST std::string& text)
        {
            m_curserPos = 0;
            m_textLine = "";
            for(INT i = 0; i < text.size(); ++i)
            {
                AddChar(text[i]);
            }
        }

        CONST std::string& GuiTextInput::GetText(VOID)
        {
            return m_textLine;
        }

        VOID GuiTextInput::SetTextColor(CONST util::Vec4& color)
        {
            m_textColor = color;
        }

        VOID GuiTextInput::ComputeInput(UINT CONST code)
        {
            CHAR c = GetCharFromVK(code);
            
            if(c >= 0x21 && c <= 0x7E)
            {
                if(c >= 0x41 && c <= 0x5A)
                {
                    std::string s;
                    s = (CHAR)code;
                    if(app::g_pApp->GetInputHandler()->IsKeyDown(KEY_LSHIFT))
                    {
                        std::transform(s.begin(), s.end(), s.begin(), ::towupper);
                    }
                    else
                    {
                        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                    }
                    c = s[0];
                }

                AddChar(c);
            }
            else if(code == KEY_RETURN && m_onReturnDeactivate)
            {
                Deactivate();
            }
            else if(code == KEY_BACKSPACE)
            {
                RemoveChar();
            }
            else if(code == KEY_SPACE)
            {
                AddChar(' ');
            }
        }

        VOID GuiTextInput::VDraw(VOID)
        {
            GuiRectangle::VDraw();

            FLOAT y = VGetPosY() / (FLOAT)d3d::g_height;
            FLOAT x = VGetPosX() / (FLOAT)d3d::g_width;

            INT lh = app::g_pApp->GetFontManager()->GetCurrentFont()->VGetLineHeight();

            FLOAT lineheight = lh / (FLOAT)d3d::g_height;

            app::g_pApp->GetFontManager()->RenderText(m_textLine.c_str(), 2.0f / (FLOAT)d3d::g_width + x, y, &m_textColor);

            if(m_drawCurser && IsActive())
            {
                app::g_pApp->GetFontManager()->RenderText("|", 2.0f / (FLOAT)d3d::g_width + m_curserPos / (FLOAT)d3d::g_width, y, &m_textColor);
            }
        }

        GuiTextInput::~GuiTextInput(VOID)
        {

        }

        GuiConsole::GuiConsole(VOID) : m_currentHistoryLine(0) , m_pAutoComplete(NULL), m_pTextLabel(NULL), m_pTextInput(NULL), m_currentAutoCompleteIndex(-1)
        {
            m_pTextInput = new GuiTextInput();
            AddComponent("input", m_pTextInput);

            m_pTextLabel = new GuiTextComponent();
            m_pTextLabel->SetTexture("gradient.png");

            AddComponent("textLabel", m_pTextLabel);

            m_pAutoComplete = new GuiTextComponent();
            AddComponent("autocomplete", m_pAutoComplete);

            tbd::KeyboardButtonPressedListener l0 = fastdelegate::MakeDelegate(this, &GuiConsole::ComputeInput);
            tbd::KeyboardButtonRepeatListener l1 = fastdelegate::MakeDelegate(this, &GuiConsole::ComputeInput);

            m_pTextInput->AddKeyPressedListener(l0);
            m_pTextInput->AddKeyRepeatListener(l1);
        }

        VOID GuiConsole::AppendText(CONST std::string& text)
        {
            m_pTextLabel->AppendText(text);
        }

        VOID GuiConsole::VDraw(VOID)
        {
            ScreenElementContainer::VDraw();
        }

        //Todo: create factory
        BOOL GuiConsole::VOnRestore(VOID)
        {
            m_pTextInput->SetOnReturnDeactivate(FALSE);
            m_pTextInput->VSetAlpha(0.85f);

            m_pTextLabel->SetAlignment(eLeft);
            m_pTextLabel->SetAppendDirection(eUp);
            m_pTextLabel->VSetAlpha(0.8f);

            m_pAutoComplete->VSetAlpha(0.0);
            
            util::Color c(0.5f, 0.5f, 0.5f, 0);

            Dimension dim;
            dim.x = VGetPosX();
            dim.y = VGetPosY() + (INT)(VGetHeight() * 0.75);
            dim.w = VGetWidth();
            dim.h = 16; //todo, font height here
            m_pTextInput->SetTextColor(c);
            m_pTextInput->VSetDimension(dim);

            dim.x = VGetPosX();
            dim.y = VGetPosY();
            dim.w = VGetWidth();
            dim.h = (INT)(VGetHeight() * 0.75);
            m_pTextLabel->VSetDimension(dim);
            m_pTextLabel->SetTextColor(c);

            dim.x = VGetPosX();
            dim.y = VGetPosY() + (INT)(VGetHeight() * 0.75) + 16;
            dim.w = 200;
            dim.h = 100;

            m_pAutoComplete->VSetDimension(dim);
            m_pAutoComplete->SetTextColor(c);

            return ScreenElementContainer::VOnRestore();
        }

        VOID GuiConsole::VSetActive(BOOL active)
        {
            ScreenElementContainer::VSetActive(active);
            if(active)
            {
                m_pTextInput->Activate();
            }
            else
            {
                m_pTextInput->Deactivate();
            }
        }

        VOID GuiConsole::SetAutoComplete(VOID)
        {
            CLAMP(m_currentAutoCompleteIndex, 0, (INT)m_pAutoComplete->GetTextLines().size()-1);
            if(m_pAutoComplete->GetTextLines().size() > 0 )
            {
                m_pTextInput->SetText(m_pAutoComplete->GetTextLines()[m_currentAutoCompleteIndex].text);
            }
        }

        VOID GuiConsole::ComputeInput(UINT CONST code)
        {
            if(code == KEY_RETURN)
            {
                if(m_pTextInput->GetText() == "")
                {
                    return;
                }
                m_pTextLabel->AppendText(m_pTextInput->GetText());
                if(!m_pTextInput->GetText().compare("exit"))
                {
                    app::g_pApp->GetHumanView()->ToggleConsole();
                }
                else
                {
                    m_commandHistory.push_back(m_pTextInput->GetText());
                    m_currentHistoryLine++;
                    if(!app::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(m_pTextInput->GetText().c_str()))
                    {
                        m_pTextLabel->AppendText("Unknown command: " + m_pTextInput->GetText());
                    }
                }
                m_pTextInput->SetText("");
            }
            else if(code == KEY_ARROW_UP)
            {

                if(m_pAutoComplete->GetTextLines().size() > 0)
                {
                    m_currentAutoCompleteIndex--;
                    SetAutoComplete();
                    return;
                }
                else
                {
                    m_currentHistoryLine--;
                    m_currentHistoryLine = m_currentHistoryLine < 0 ? 0 : m_currentHistoryLine;
                    if(m_commandHistory.size() > 0)
                    {
                        m_pTextInput->SetText(m_commandHistory[m_currentHistoryLine]);
                    }
                }
            }
            else if(code == KEY_ARROW_DOWN)
            {

                if(m_pAutoComplete->GetTextLines().size() > 0)
                {
                    m_currentAutoCompleteIndex++;
                    SetAutoComplete();
                    return;
                }
                else
                {
                    m_currentHistoryLine++;
                    m_currentHistoryLine = (INT)(m_currentHistoryLine >= m_commandHistory.size() ? m_commandHistory.size() - 1 : m_currentHistoryLine);
                    if(m_commandHistory.size() > 0)
                    {
                        m_pTextInput->SetText(m_commandHistory[m_currentHistoryLine]);
                    }
                }
            }
            else if(code == KEY_CIRCUMFLEX)
            {
                app::g_pApp->GetHumanView()->ToggleConsole();
            }
            else if(code == KEY_TAB)
            {
                SetAutoComplete();
                m_currentAutoCompleteIndex = 0;
            }

            m_pAutoComplete->ClearText();

            if(m_pTextInput->GetText().size() > 0)
            {
                std::list<std::string> l = app::g_pApp->GetLogic()->GetCommandInterpreter()->GetCommands();
                for(auto it = l.begin(); it != l.end(); ++it)
                {
                    CONST std::string& text = m_pTextInput->GetText();
                    if(it->substr(0, text.size()) == text)
                    {
                        m_pAutoComplete->AppendText(*it);
                    }
                }
            }
        }

        Histogram::Histogram(UINT iVal /* = 10 */, UINT uVal) : m_uVal(uVal), m_iVal(iVal), m_pos(0), m_time(0), m_pFloats(NULL), m_max(0)
        {

        }

        BOOL Histogram::VOnRestore(VOID)
        {
            SAFE_ARRAY_DELETE(m_pFloats);
            m_pFloats = new FLOAT[m_iVal];

            return GuiRectangle::VOnRestore();
        }

        VOID Histogram::AddValue(INT val)
        {
            m_vals.push_front(val);
            m_max = 0;

            if(m_vals.size() > m_iVal)
            {
                m_vals.pop_back();
            }

            TBD_FOR(m_vals)
            {
                INT v = *it;
                m_max = v < m_max ? m_max : v;
            }
            INT i = 0;
            TBD_FOR(m_vals)
            {
                m_pFloats[i] = *it / (FLOAT)m_max;
                i++;
            }
        }

        VOID Histogram::VDraw(VOID)
        {
            GuiRectangle::VDraw();

            FLOAT pos = 0;
            FLOAT delta = VGetWidth() / (FLOAT)m_iVal;
            delta = delta == 0 ? 1 : delta;
            FLOAT last = -1;
            TBD_FOR_INT(m_vals.size())
            {
                if(i == 0)
                {
                    last = m_pFloats[0];
                    ++i;
                }
                FLOAT v = m_pFloats[i];
                INT h0 = (INT)(last * VGetHeight());
                INT h1 = (INT)(v * VGetHeight());
                d3d::DrawLine((INT)(pos + VGetPosX() + 0.5f), d3d::g_height - VGetPosY() - VGetHeight() + h0, (INT)(delta+0.5f), h1 - h0);

                pos += delta;
                last = v;
            }
        }

        Histogram::~Histogram(VOID)
        {
            SAFE_ARRAY_DELETE(m_pFloats);
        }
    }
}
