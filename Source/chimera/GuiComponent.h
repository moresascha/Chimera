#pragma once
#include "stdafx.h"
#include "ScreenElement.h"

namespace chimera
{
    namespace gui
    {
        class D3DGuiComponent : public ScreenElement
        {
        protected:
            chimera::PixelShader* m_pPixelShader;
            std::string m_shaderName;
        public:
            D3DGuiComponent(VOID);
            VOID SetEffect(LPCSTR pixelShader);
            virtual VOID VUpdate(ULONG millis) {}
            virtual BOOL VOnRestore(VOID);
        };

        class D3D_GUI : public ScreenElementContainer
        {
            IShader* m_pVertexShader;
            IShader* m_pPixelShader;
        public:
            D3D_GUI(VOID);
            BOOL VOnRestore(VOID);
            VOID VDraw(VOID);
            ~D3D_GUI(VOID);
        };

        class GuiRectangle : public D3DGuiComponent
        {
        private:
            FLOAT m_tx, m_ty, m_u, m_v;

        protected:
            VOID SetTextureCoords(FLOAT x, FLOAT y, FLOAT u, FLOAT v);

        public:
            GuiRectangle(VOID);
            virtual VOID VDraw(VOID);
            ~GuiRectangle(VOID);
        };

        class GuiTextureComponent : public GuiRectangle
        {
        protected:
            std::string m_resource;
            std::shared_ptr<chimera::D3DTexture2D> m_textureHandle;
        public:
            GuiTextureComponent(VOID);
            VOID SetTexture(LPCSTR texFile);
            virtual BOOL VOnRestore(VOID);
            virtual VOID VDraw(VOID);
        };

        class GuiSpriteComponent : public GuiTextureComponent
        {
        private:
            UINT m_tx, m_ty, m_u, m_v;

        public:
            GuiSpriteComponent(UINT tx, UINT ty, UINT du, UINT dv);
            BOOL VOnRestore(VOID);
            VOID VDraw(VOID);
        };

        class GuiInputComponent : public GuiRectangle, public InputAdapter
        {
        protected:
            BOOL m_onReturnDeactivate;
        public:
            GuiInputComponent(VOID);
            VOID SetOnReturnDeactivate(BOOL deactivate);
            virtual ~GuiInputComponent(VOID);
        };

        enum Alignment
        {
            eRight,
            eLeft,
            eCenter
        };

        enum AppendDirection
        {
            eUp,
            eDown
        };

        struct TextLine
        {
            UINT width;
            UINT height;
            std::string text;
            TextLine(VOID) : width(0), height(0)
            {

            }
        };

        class GuiTextComponent : public GuiTextureComponent
        {
        private:
            Alignment m_alignment;
            AppendDirection m_appendDir;
            std::vector<TextLine> m_textLines;
            util::Color m_textColor;
            VOID DrawText(TextLine& line, INT x, INT y);

        public:
            GuiTextComponent(VOID);

            VOID SetAlignment(Alignment alignment);

            VOID SetAppendDirection(AppendDirection dir);

            //VOID AddText(CONST std::string& text, INT x, INT y);

            VOID AppendText(CONST std::string& text);

            CONST std::vector<TextLine> GetTextLines(VOID) CONST;

            VOID SetTextColor(CONST util::Vec4& color);

            VOID ClearText(VOID);

            virtual BOOL VOnRestore(VOID);

            virtual VOID VDraw(VOID);

            virtual ~GuiTextComponent(VOID);
        };

        class GuiTextInput : public GuiInputComponent
        {
        protected:
            util::Color m_textColor;
            BOOL m_drawCurser;
            INT m_curserPos;
            std::string m_textLine;
            ULONG m_time;

            VOID ComputeInput(CONST UINT code);

        public:
            GuiTextInput(VOID);
            
            virtual VOID VDraw(VOID);
            
            virtual VOID VUpdate(ULONG millis);

            virtual VOID AddChar(CHAR c);

            VOID RemoveChar(VOID);

            VOID AddText(std::string& text);
            
            CONST std::string& GetText(VOID);

            VOID SetTextColor(CONST util::Vec4& color);

            VOID SetText(CONST std::string& text);

            virtual BOOL VOnRestore(VOID);

            ~GuiTextInput(VOID);
        };

        class GuiConsole : public ScreenElementContainer, public InputAdapter
        {
        private:
            std::vector<std::string> m_commandHistory;
            GuiTextInput* m_pTextInput;
            GuiTextComponent* m_pTextLabel;
            GuiTextComponent* m_pAutoComplete;
            INT m_currentHistoryLine;
            INT m_currentAutoCompleteIndex;

            VOID ComputeInput(UINT CONST code);

            VOID SetAutoComplete(VOID);

        public:
            GuiConsole(VOID);
            
            VOID VSetActive(BOOL active);

            VOID VDraw(VOID);

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
}


