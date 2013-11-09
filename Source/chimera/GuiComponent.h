#pragma once
#include "stdafx.h"
#include "ScreenElement.h"
#include "Input.h"

namespace chimera
{
    class GuiLookAndFeel : public IGuiLookAndFeel
    {
    public:
        Color VGetBackgroundColor(VOID) { return Color(0,0,0,0.75f); }

        Color VGetForegroundColor(VOID) { return Color(0,0,0,1); }

        Color VGetFontColor(VOID) { return Color(1,1,1,1); }
    };

    class GuiComponent : public virtual IGuiComponent, public virtual ScreenElement
    {
    protected:
        IShader* m_pPixelShader;
        std::string m_shaderName;
        BOOL m_hasFocus;
        BOOL m_enabled;

    public:
        GuiComponent(VOID);

        VOID VSetEffect(LPCSTR pixelShader);

        virtual BOOL VOnRestore(VOID);

        virtual VOID VSetEnabled(BOOL enable);

        virtual VOID VSetFocus(BOOL focus);

        virtual ~GuiComponent(VOID);
    };

    class Gui : public ScreenElementContainer, public IGui
    {
    private:
        std::vector<IGuiComponent*> m_guiComponents;
        VOID ComputeInput(INT, INT, INT);
        VOID ComputeKeyInput(UINT);

    public:
        Gui(VOID);

        VOID VAddComponent(IGuiComponent* cmp);

        VOID VSetActive(BOOL activate);

        ~Gui(VOID);
    };

    class GuiRectangle : public GuiComponent, public IGuiRectangle
    {
    protected:
        FLOAT m_tx, m_ty, m_u, m_v;

    public:
        GuiRectangle(VOID);
        virtual VOID VDraw(VOID);
        ~GuiRectangle(VOID);
    };

    class GuiTextureComponent : public GuiRectangle, public IGuiTextureComponent
    {
    protected:
        std::string m_resource;
        std::shared_ptr<IDeviceTexture> m_textureHandle;
        VOID SetTextureCoords(FLOAT x, FLOAT y, FLOAT u, FLOAT v);

    public:
        GuiTextureComponent(VOID);
        VOID VSetTexture(LPCSTR texture);
        virtual BOOL VOnRestore(VOID);
        virtual VOID VDraw(VOID);
    };

/*
    class GuiInputComponent : public GuiRectangle
    {
    protected:
        BOOL m_onReturnDeactivate;
    public:
        GuiInputComponent(VOID);
        VOID SetOnReturnDeactivate(BOOL deactivate);
        virtual ~GuiInputComponent(VOID);
    };*/


    class GuiTextComponent : public GuiTextureComponent, public IGuiTextComponent
    {
    private:
        Alignment m_alignment;
        AppendDirection m_appendDir;
        std::vector<TextLine> m_textLines;
        chimera::Color m_textColor;
        VOID DrawText(TextLine& line, INT x, INT y);

    public:
        GuiTextComponent(VOID);

        VOID VSetAlignment(Alignment alignment);

        VOID VSetTextAppendDirection(AppendDirection dir);

        //VOID AddText(CONST std::string& text, INT x, INT y);

        VOID VAppendText(CONST std::string& text);

        CONST std::vector<TextLine>& VGetTextLines(VOID) CONST;

        VOID VSetTextColor(CONST util::Vec4& color);

        VOID VClearText(VOID);

        virtual BOOL VOnRestore(VOID);

        virtual VOID VDraw(VOID);

        virtual ~GuiTextComponent(VOID);
    };

    class GuiTextInputComponent : public IGuiTextInputComponent, public GuiRectangle
    {
    protected:
        Color m_textColor;
        BOOL m_drawCurser;
        INT m_curserPos;
        std::string m_textLine;
        ULONG m_time;

        VOID ComputeInput(CONST UINT code);

    public:
        GuiTextInputComponent(VOID);
            
        virtual VOID VDraw(VOID);

        virtual VOID VAddChar(CHAR c);

        VOID VRemoveChar(VOID);

        VOID VAddText(std::string& text);
            
        CONST std::string& VGetText(VOID);

        VOID VSetTextColor(CONST util::Vec4& color);

        VOID VSetText(CONST std::string& text);

        VOID VClearText(VOID) { VSetText(""); }

        VOID VUpdate(ULONG millis);

        virtual BOOL VOnRestore(VOID);

        ~GuiTextInputComponent(VOID);
    };

    class GuiConsole : public ScreenElementContainer
    {
    private:
        std::vector<std::string> m_commandHistory;
        IGuiTextInputComponent* m_pTextInput;
        IGuiTextComponent* m_pTextLabel;
        IGuiTextComponent* m_pAutoComplete;
        INT m_currentHistoryLine;
        INT m_currentAutoCompleteIndex;

        VOID ComputeInput(UINT CONST code);

        VOID SetAutoComplete(VOID);

    public:
        GuiConsole(VOID);

        VOID VDraw(VOID);

        VOID VSetActive(BOOL activate);

        BOOL VOnRestore(VOID);

        VOID AppendText(CONST std::string& text);
            
        ~GuiConsole(VOID) {}
    };

    class Histogram : public GuiRectangle
    {
    private:
        std::list<INT> m_vals;
        FLOAT* m_pFloats;
        UINT m_pos;
        UINT m_iVal;
        UINT m_uVal;
        UINT m_time;
        INT m_max;
    public:
        Histogram(UINT iVal = 10, UINT uVal = 200);

        VOID AddValue(INT val);

        VOID VDraw(VOID);

        BOOL VOnRestore(VOID);

        ~Histogram(VOID);
    };
}


