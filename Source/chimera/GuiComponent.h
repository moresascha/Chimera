#pragma once
#include "stdafx.h"
#include "ScreenElement.h"
#include "Input.h"

namespace chimera
{
    class GuiLookAndFeel : public IGuiLookAndFeel
    {
    public:
        Color VGetBackgroundColor(void) { return Color(0,0,0,0.75f); }

        Color VGetForegroundColor(void) { return Color(0,0,0,1); }

        Color VGetFontColor(void) { return Color(1,1,1,1); }
    };

    class GuiComponent : public virtual IGuiComponent, public virtual ScreenElement
    {
    protected:
        IShader* m_pPixelShader;
        std::string m_shaderName;
        bool m_hasFocus;
        bool m_enabled;

    public:
        GuiComponent(void);

        void VSetEffect(LPCSTR pixelShader);

        virtual bool VOnRestore(void);

        virtual void VSetEnabled(bool enable);

        virtual void VSetFocus(bool focus);

        virtual ~GuiComponent(void);
    };

    class Gui : public ScreenElementContainer, public IGui
    {
    private:
        std::vector<IGuiComponent*> m_guiComponents;
        void ComputeInput(int, int, int);
        void ComputeKeyInput(uint);

    public:
        Gui(void);

        void VAddComponent(IGuiComponent* cmp);

        void VSetActive(bool activate);

        ~Gui(void);
    };

    class GuiRectangle : public GuiComponent, public IGuiRectangle
    {
    protected:
        float m_tx, m_ty, m_u, m_v;

    public:
        GuiRectangle(void);
        virtual void VDraw(void);
        ~GuiRectangle(void);
    };

    class GuiTextureComponent : public GuiRectangle, public IGuiTextureComponent
    {
    protected:
        std::string m_resource;
        std::shared_ptr<IDeviceTexture> m_textureHandle;
        void SetTextureCoords(float x, float y, float u, float v);

    public:
        GuiTextureComponent(void);
        void VSetTexture(LPCSTR texture);
        virtual bool VOnRestore(void);
        virtual void VDraw(void);
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
        void DrawText(TextLine& line, int x, int y);

    public:
        GuiTextComponent(void);

        void VSetAlignment(Alignment alignment);

        void VSetTextAppendDirection(AppendDirection dir);

        //VOID AddText(CONST std::string& text, INT x, INT y);

        void VAppendText(const std::string& text);

        const std::vector<TextLine>& VGetTextLines(void) const;

        void VSetTextColor(const util::Vec4& color);

        void VClearText(void);

        virtual bool VOnRestore(void);

        virtual void VDraw(void);

        virtual ~GuiTextComponent(void);
    };

    class GuiTextInputComponent : public IGuiTextInputComponent, public GuiRectangle
    {
    protected:
        Color m_textColor;
        bool m_drawCurser;
        int m_curserPos;
        std::string m_textLine;
        ulong m_time;

        void ComputeInput(const uint code);

    public:
        GuiTextInputComponent(void);
            
        virtual void VDraw(void);

        virtual void VAddChar(char c);

        void VRemoveChar(void);

        void VAddText(std::string& text);
            
        const std::string& VGetText(void);

        void VSetTextColor(const util::Vec4& color);

        void VSetText(const std::string& text);

        void VClearText(void) { VSetText(""); }

        void VUpdate(ulong millis);

        virtual bool VOnRestore(void);

        ~GuiTextInputComponent(void);
    };

    class GuiConsole : public ScreenElementContainer
    {
    private:
        std::vector<std::string> m_commandHistory;
        IGuiTextInputComponent* m_pTextInput;
        IGuiTextComponent* m_pTextLabel;
        IGuiTextComponent* m_pAutoComplete;
        int m_currentHistoryLine;
        int m_currentAutoCompleteIndex;

        void ComputeInput(uint const code);

        void SetAutoComplete(void);

    public:
        GuiConsole(void);

        void VDraw(void);

        void VSetActive(bool activate);

        bool VOnRestore(void);

        void AppendText(const std::string& text);
            
        ~GuiConsole(void) {}
    };

    class Histogram : public GuiRectangle
    {
    private:
        std::list<int> m_vals;
        float* m_pFloats;
        uint m_pos;
        uint m_iVal;
        uint m_uVal;
        uint m_time;
        int m_max;
    public:
        Histogram(uint iVal = 10, uint uVal = 200);

        void AddValue(int val);

        void VDraw(void);

        bool VOnRestore(void);

        ~Histogram(void);
    };
}


