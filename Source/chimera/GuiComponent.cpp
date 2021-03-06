#include "GuiComponent.h"
#include "GeometryFactory.h"
#include "util.h"

namespace chimera
{

    Gui::Gui(void)
    {
        MouseButtonPressedListener l0 = fastdelegate::MakeDelegate(this, &Gui::ComputeInput);
        AddMousePressedListener(l0);

        KeyboardButtonPressedListener l1 = fastdelegate::MakeDelegate(this, &Gui::ComputeKeyInput);
        AddKeyPressedListener(l1);
    }

    void Gui::VSetActive(bool activate)
    {
        if(activate)
        {
            ActivateInput();
            CmGetApp()->VGetInputHandler()->VGrabMouse(false);
        }
        else
        {
            DeactivateInput();
        }
        ScreenElementContainer::VSetActive(activate);
    }

    void Gui::VAddComponent(IGuiComponent* cmp)
    {
        ScreenElementContainer::VAddComponent(cmp);
        m_guiComponents.push_back(cmp);
    }

    void Gui::ComputeKeyInput(uint key)
    {
        if(key == KEY_ESC)
        {
            VSetActive(false);
        }
    }

    void Gui::ComputeInput(int x, int y, int s)
    {
        if(s != MOUSE_BTN_LEFT)
        {
            return; //for now
        }
        TBD_FOR(m_guiComponents)
        {
            IGuiComponent* element = *it;
            if(element->VIsIn(x, y))
            {
                element->VSetFocus(true);
            }
        }
    }

    Gui::~Gui(void)
    {
    }

    //---imp

    GuiComponent::GuiComponent() : m_shaderName("GuiDefault_PS"), m_pPixelShader(NULL)
    {
    }

    void GuiComponent::VSetEnabled(bool enable)
    {
        m_enabled = enable;
    }

    void GuiComponent::VSetFocus(bool focus)
    {
        if(focus)
        {
            ActivateInput();
        }
        else
        {
            DeactivateInput();
        }
    }

    void GuiComponent::VSetEffect(LPCSTR pixelShader)
    {
        m_shaderName = pixelShader;
    }

    bool GuiComponent::VOnRestore(void)
    {
        CMShaderDescription desc;
        desc.file = GUI_SHADER_FILE;
        desc.function = m_shaderName.c_str();
        m_pPixelShader = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShader(desc.function, &desc, eShaderType_FragmentShader);

        return ScreenElement::VOnRestore();
    }

    GuiComponent::~GuiComponent(void)
    {

    }

    GuiRectangle::GuiRectangle(void) : m_tx(0), m_ty(0), m_u(1), m_v(1)
    {

    }

    void GuiRectangle::VDraw(void)
    {
        if(VGetAlpha() < 1)
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VPushAlphaBlendState();
        }

        float x = -1.0f + 2 * VGetPosX() / (float)CmGetApp()->VGetWindowWidth();
        float y = 1.0f - 2 * VGetPosY() / (float)CmGetApp()->VGetWindowHeight();
        float w = x + 2 * VGetWidth() / (float)CmGetApp()->VGetWindowWidth();
        float h = y - 2 * VGetHeight() / (float)CmGetApp()->VGetWindowHeight();
        float localVertices[20] = 
        {
            x, h, 0, m_tx, m_ty,
            w, h, 0, m_u, m_ty,
            x, y, 0, m_tx, m_v,
            w, y, 0, m_u, m_v,
        };

        IConstShaderBuffer* buffer = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetConstShaderBuffer(eGuiColorBuffer);

        XMFLOAT4* color = (XMFLOAT4*)buffer->VMap();
        color->x = m_color.r;
        color->y = m_color.g;
        color->z = m_color.b;
        color->w = m_color.a;
        buffer->VUnmap();

        m_pPixelShader->VBind();
        IGeometry* quad = geometryfactroy::GetGlobalScreenQuadCPU();
        quad->VGetVertexBuffer()->VSetData(localVertices, 20 * sizeof(float));
        quad->VBind();
        quad->VDraw();

        if(VGetAlpha() < 1)
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VPopBlendState();
        }
    }

    GuiRectangle::~GuiRectangle(void)
    {

    }

    void GuiTextureComponent::SetTextureCoords(float x, float y, float u, float v)
    {
        m_tx = x;
        m_ty = y;
        m_u = u;
        m_v = v;
    }

    GuiTextureComponent::GuiTextureComponent(void)
    {
        VSetEffect("GuiTexture_PS");
        VSetTexture("default64x641.png");
        SetTextureCoords(0, 0, 1, 1);
    }

    void GuiTextureComponent::VDraw(void)
    {
        if(m_textureHandle)
        {
            if(m_textureHandle->VIsReady())
            {
                m_textureHandle->VUpdate();
                CmGetApp()->VGetRenderer()->VSetTexture(eGuiSampler, m_textureHandle.get());
                GuiRectangle::VDraw();
            }
            else
            {
                m_textureHandle = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_resource));
            }
        }
        else
        {
            GuiRectangle::VDraw();
        }
    }

    void GuiTextureComponent::VSetTexture(LPCSTR texFile)
    {
        m_resource = texFile;
        if(strcmp(texFile, "") == 0)
        {
            VSetEffect("GuiDefault_PS");
        }
        else
        {
            VSetEffect("GuiTexture_PS");
        }
    }

    bool GuiTextureComponent::VOnRestore(void)
    {
        m_textureHandle = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_resource));
        return GuiRectangle::VOnRestore();
    }

    /*
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
    }*/

    /*
    GuiInputComponent::GuiInputComponent(VOID) : m_onReturnDeactivate(TRUE)
    {

    }

    VOID GuiInputComponent::SetOnReturnDeactivate(BOOL deactivate)
    {
        m_onReturnDeactivate = deactivate;
    }

    GuiInputComponent::~GuiInputComponent(VOID)
    {

    }*/

    GuiTextComponent::GuiTextComponent(void) : m_appendDir(eGuiTextAppendDown), m_alignment(eGuiAlignLeft), m_textColor(1,1,1,1)
    {
        VSetTexture("");
    }

    const std::vector<TextLine>& GuiTextComponent::VGetTextLines() const
    {
        return m_textLines;
    }

    void GuiTextComponent::VSetAlignment(Alignment alignment)
    {
        m_alignment = alignment;
    }

    void GuiTextComponent::VSetTextAppendDirection(AppendDirection dir)
    {
        m_appendDir = dir;
    }

    void GuiTextComponent::VAppendText(const std::string& text)
    {
        uint maxHistory = 100; //todo
        if(m_textLines.size() >= maxHistory)
        {
            m_textLines.erase(m_textLines.begin());
        }
        std::vector<std::string> lines = util::split(text, '\n');
        if(lines.empty())
        {
            lines.push_back(text);
        }
        TBD_FOR(lines)
        {
            TextLine line;
            line.text = *it;
            for(int i = 0; i < line.text.size(); ++i)
            {
                const CMCharMetric* metric = CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetCharMetric(line.text[i]);
                if(!metric)
                {
                    metric = CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetCharMetric('?');
                }
                line.width += metric->xadvance;
            }
            m_textLines.push_back(line);
        }
    }

    bool GuiTextComponent::VOnRestore(void)
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

    void GuiTextComponent::DrawText(TextLine& line, int x, int y)
    {
        int mx = x;
        if(m_alignment == eGuiAlignCenter)
        {
            mx -= (int)(0.5 * line.width);
        }
        else if(m_alignment == eGuiAlignRight)
        {
            mx -= line.width + 5;
        }
        if(mx + line.width > VGetPosX() + VGetWidth())
        {
            //slow implementation, only do this when dimension change, for example. Todo
            int w = mx + line.width;
            std::string text = line.text;
            while(w > (int)(VGetPosX() + VGetWidth()))
            {
                char c = text.substr(text.size() - 1, text.size())[0];
                w -= CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetCharMetric(c)->xadvance;
                text = text.substr(0, text.size() - 1);
            }
            CmGetApp()->VGetHumanView()->VGetFontManager()->VRenderText(text, mx / (float)CmGetApp()->VGetWindowWidth(), y / (float)CmGetApp()->VGetWindowHeight(), &m_textColor);
        }
        else
        {
            CmGetApp()->VGetHumanView()->VGetFontManager()->VRenderText(line.text, mx / (float)CmGetApp()->VGetWindowWidth(), y / (float)CmGetApp()->VGetWindowHeight(), &m_textColor);
        }
    }

    void GuiTextComponent::VClearText(void)
    {
        m_textLines.clear();
    }

    void GuiTextComponent::VDraw(void)
    {
        GuiTextureComponent::VDraw();

        int y = VGetPosY();
            
        int lineheight = CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetLineHeight();

        /*if(m_appendDir == eUp)
        {
            y = y + VGetHeight();
        }
        else if(m_appendDir == eDown)
        {
            y += lineheight;
        } */
            
        int x = VGetPosX();

        int centerX = 0;

        if(m_alignment == eGuiAlignCenter)
        {
            x = x + (int)(0.5 * VGetWidth());
        }
        else if(m_alignment == eGuiAlignRight)
        {
            x = x + VGetWidth();
        }

        if(m_appendDir == eGuiTextAppendUp)
        {
            for(int i = (int)m_textLines.size() - 1; i >= 0; --i)
            {
                TextLine line = m_textLines[i];
                DrawText(line, x + 2, VGetHeight() - y - lineheight - 2);
                y += lineheight;
                if((VGetHeight()-y-lineheight) < (int)VGetPosY())
                {
                    break;
                }
            }
        }
        else
        {
            for(int i = 0; i < m_textLines.size(); ++i)
            {
                TextLine line = m_textLines[i];
                DrawText(line, x + 2, y);
                y += lineheight;
                if((y+lineheight) > (int)(VGetPosY() + VGetHeight()))
                {
                    break;
                }
            }
        }
    }

    void GuiTextComponent::VSetTextColor(const util::Vec4& color)
    {
        m_textColor = color;
    }

    GuiTextComponent::~GuiTextComponent(void)
    {

    }

    GuiTextInputComponent::GuiTextInputComponent(void) : m_time(0), m_drawCurser(false), m_curserPos(0), m_textColor(1,1,1,1)
    {
        chimera::KeyboardButtonPressedListener l0 = fastdelegate::MakeDelegate(this, &GuiTextInputComponent::ComputeInput);
        chimera::KeyboardButtonRepeatListener l1 = fastdelegate::MakeDelegate(this, &GuiTextInputComponent::ComputeInput);
        AddKeyPressedListener(l0);
        AddKeyRepeatListener(l1);
    }

    bool GuiTextInputComponent::VOnRestore(void)
    {
        GuiComponent::VOnRestore();
        return true;
    }

    void GuiTextInputComponent::VAddChar(char c)
    {
        int nextPos = m_curserPos + CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetCharMetric(c)->xadvance;
        if(nextPos > (int)(VGetWidth())-10)
        {
            return;
        }
        m_curserPos = nextPos;
        m_textLine += c;
    }

    void GuiTextInputComponent::VRemoveChar(void)
    {
        if(m_textLine.size() > 0)
        {
            char c = m_textLine.substr(m_textLine.size() - 1, m_textLine.size())[0];
            m_textLine = m_textLine.substr(0, m_textLine.size() - 1);
            m_curserPos -= CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetCharMetric(c)->xadvance;
        }
    }

    void GuiTextInputComponent::VSetText(const std::string& text)
    {
        m_curserPos = 0;
        m_textLine = "";
        for(int i = 0; i < text.size(); ++i)
        {
            VAddChar(text[i]);
        }
    }

    const std::string& GuiTextInputComponent::VGetText(void)
    {
        return m_textLine;
    }

    void GuiTextInputComponent::VSetTextColor(const util::Vec4& color)
    {
        m_textColor = color;
    }

    void GuiTextInputComponent::ComputeInput(uint const code)
    {
        char c = GetCharFromVK(code);
            
        if(c >= 0x21 && c <= 0x7E)
        {
            if(c >= 0x41 && c <= 0x5A)
            {
                std::string s;
                s = (char)code;
                if(CmGetApp()->VGetInputHandler()->VIsKeyDown(KEY_LSHIFT))
                {
                    std::transform(s.begin(), s.end(), s.begin(), ::towupper);
                }
                else
                {
                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                }
                c = s[0];
            }

            VAddChar(c);
        }
        else if(code == KEY_RETURN)
        {
            VSetFocus(false);
        }
        else if(code == KEY_BACKSPACE)
        {
            VRemoveChar();
        }
        else if(code == KEY_SPACE)
        {
            VAddChar(' ');
        }
    }

    void GuiTextInputComponent::VUpdate(ulong millis)
    {
        m_time += millis;
        if(m_time > 400)
        {
            m_drawCurser = !m_drawCurser;
            m_time = 0;
        }
    }

    void GuiTextInputComponent::VDraw(void)
    {
        GuiRectangle::VDraw();

        float y = VGetPosY() / (float)CmGetApp()->VGetWindowHeight();
        float x = VGetPosX() / (float)CmGetApp()->VGetWindowWidth();

        int lh = CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetLineHeight();

        float lineheight = lh / (float)CmGetApp()->VGetWindowHeight();

        CmGetApp()->VGetHumanView()->VGetFontManager()->VRenderText(m_textLine.c_str(), 2.0f / (float)CmGetApp()->VGetWindowWidth() + x, y, &m_textColor);

        if(m_drawCurser && IsInputActive())
        {
            CmGetApp()->VGetHumanView()->VGetFontManager()->VRenderText("|", 2.0f / (float)CmGetApp()->VGetWindowWidth() + m_curserPos / (float)CmGetApp()->VGetWindowWidth() + x, y, &m_textColor);
        }
    }

    GuiTextInputComponent::~GuiTextInputComponent(void)
    {

    }

    GuiConsole::GuiConsole(void) : m_currentHistoryLine(0) , m_pAutoComplete(NULL), m_pTextLabel(NULL), m_pTextInput(NULL), m_currentAutoCompleteIndex(-1)
    {
        m_pTextInput = CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextInputComponent();
        m_pTextInput->VSetName("input");
        VAddComponent(m_pTextInput);

        m_pTextLabel = CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextComponent();
        //m_pTextLabel->SetTexture("gradient.png");
        m_pTextLabel->VSetName("textlabel");
        VAddComponent(m_pTextLabel);

        m_pAutoComplete = CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextComponent();
        m_pAutoComplete->VSetName("autocomplete");
        VAddComponent(m_pAutoComplete);

        chimera::KeyboardButtonPressedListener l0 = fastdelegate::MakeDelegate(this, &GuiConsole::ComputeInput);
        chimera::KeyboardButtonRepeatListener l1 = fastdelegate::MakeDelegate(this, &GuiConsole::ComputeInput);

        m_pTextInput->AddKeyPressedListener(l0);
        m_pTextInput->AddKeyRepeatListener(l1);

        VSetBackgroundColor(0,0,0);
    }

    void GuiConsole::AppendText(const std::string& text)
    {
        m_pTextLabel->VAppendText(text);
    }

    void GuiConsole::VDraw(void)
    {
        ScreenElementContainer::VDraw();
    }

    //Todo: create factory
    bool GuiConsole::VOnRestore(void)
    {
        ScreenElementContainer::VOnRestore();

        m_pTextInput->VSetAlpha(0.75f);

        m_pTextLabel->VSetAlignment(eGuiAlignLeft);
        m_pTextLabel->VSetTextAppendDirection(eGuiTextAppendUp);
        m_pTextLabel->VSetAlpha(.65f);

        m_pAutoComplete->VSetAlpha(0.0);
            
        Color tc(1, 1, 1, 0);

        CMDimension dim;
        dim.x = VGetPosX();
        dim.y = VGetPosY() + (int)(VGetHeight() * 0.75);
        dim.w = VGetWidth();
        dim.h = CmGetApp()->VGetHumanView()->VGetFontManager()->VGetCurrentFont()->VGetLineHeight() + 2; //todo, font height here
        m_pTextInput->VSetTextColor(tc);
        m_pTextInput->VSetDimension(dim);

        dim.x = VGetPosX();
        dim.y = VGetPosY();
        dim.w = VGetWidth();
        dim.h = (int)(VGetHeight() * 0.75);
        m_pTextLabel->VSetDimension(dim);

        m_pTextLabel->VSetTextColor(tc);

        dim.x = VGetPosX();
        dim.y = VGetPosY() + (int)(VGetHeight() * 0.75) + 16;
        dim.w = 200;
        dim.h = 100;

        m_pAutoComplete->VSetDimension(dim);
        m_pAutoComplete->VSetTextColor(tc);

        return true; 
    }

    void GuiConsole::VSetActive(bool active)
    {
        ScreenElementContainer::VSetActive(active);
        m_pTextInput->VSetFocus(active);
    }

    void GuiConsole::SetAutoComplete(void)
    {
        m_currentAutoCompleteIndex = CLAMP(m_currentAutoCompleteIndex, 0, (int)m_pAutoComplete->VGetTextLines().size()-1);
        if(m_pAutoComplete->VGetTextLines().size() > 0 )
        {
            m_pTextInput->VSetText(m_pAutoComplete->VGetTextLines()[m_currentAutoCompleteIndex].text);
        }
    }

    void GuiConsole::ComputeInput(uint const code)
    {
        if(code == KEY_RETURN)
        {
            if(m_pTextInput->VGetText() == "")
            {
                return;
            }
            m_pTextLabel->VAppendText(m_pTextInput->VGetText());
            if(!m_pTextInput->VGetText().compare("close"))
            {
                VSetActive(!VIsActive());
            }
            else
            {
                m_commandHistory.push_back(m_pTextInput->VGetText());
                m_currentHistoryLine++;
                if(!CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(m_pTextInput->VGetText().c_str()))
                {
                    m_pTextLabel->VAppendText("Unknown command: " + m_pTextInput->VGetText());
                }
            }
            m_pTextInput->VSetText("");
            m_pTextInput->VSetFocus(true);
        }
        else if(code == KEY_ARROW_UP)
        {

            if(m_pAutoComplete->VGetTextLines().size() > 0)
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
                    m_pTextInput->VSetText(m_commandHistory[m_currentHistoryLine]);
                }
            }
        }
        else if(code == KEY_ARROW_DOWN)
        {

            if(m_pAutoComplete->VGetTextLines().size() > 0)
            {
                m_currentAutoCompleteIndex++;
                SetAutoComplete();
                return;
            }
            else
            {
                m_currentHistoryLine++;
                m_currentHistoryLine = (int)(m_currentHistoryLine >= m_commandHistory.size() ? m_commandHistory.size() - 1 : m_currentHistoryLine);
                if(m_commandHistory.size() > 0)
                {
                    m_pTextInput->VSetText(m_commandHistory[m_currentHistoryLine]);
                }
            }
        }
        else if(code == KEY_CIRCUMFLEX)
        {
            VSetActive(!VIsActive());
        }
        else if(code == CM_KEY_TAB)
        {
            SetAutoComplete();
            m_currentAutoCompleteIndex = 0;
        }

        m_pAutoComplete->VClearText();

        if(m_pTextInput->VGetText().size() > 0)
        {
            std::vector<std::string> l = CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VGetCommands();
            for(auto it = l.begin(); it != l.end(); ++it)
            {
                const std::string& text = m_pTextInput->VGetText();
                if(it->substr(0, text.size()) == text)
                {
                    m_pAutoComplete->VAppendText(*it);
                }
            }
        }
    }

    /*
    Histogram::Histogram(UINT iVal / * = 10 * /, UINT uVal) : m_uVal(uVal), m_iVal(iVal), m_pos(0), m_time(0), m_pFloats(NULL), m_max(0)
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
            
            //DrawLine((INT)(pos + VGetPosX() + 0.5f), CmGetApp()->VGetWindowHeight() - VGetPosY() - VGetHeight() + h0, (INT)(delta+0.5f), h1 - h0);

            pos += delta;
            last = v;
        }
    }

    Histogram::~Histogram(VOID)
    {
        SAFE_ARRAY_DELETE(m_pFloats);
    }*/
}
