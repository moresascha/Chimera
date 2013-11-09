#pragma once
#include "CMTypes.h"
#include "CMCommon.h"

#define GUI_SHADER_FILE L"Gui.hlsl"

namespace chimera
{
    class IGuiComponent : virtual public IScreenElement, public InputAdapter
    {
    public:
        virtual VOID VSetEffect(LPCSTR ps) = 0;

        virtual VOID VSetFocus(BOOL focus) = 0;

        virtual VOID VSetEnabled(BOOL enable) = 0;

        virtual ~IGuiComponent(VOID) { }
    };

    class IGui : public virtual IScreenElementContainer, public InputAdapter
    {
    public:
        virtual VOID VAddComponent(IGuiComponent* cmp) = 0;

        virtual ~IGui(VOID) {}
    };

    class IGuiRectangle : public virtual IGuiComponent
    {

    };

    class IGuiTextureComponent : public virtual IGuiComponent
    {
    public:
        virtual VOID VSetTexture(LPCSTR texture) = 0;

        virtual ~IGuiTextureComponent(VOID) { }
    };

    class IGuiTextComponent : public virtual IGuiComponent
    {
    public:
        virtual VOID VSetAlignment(Alignment e) = 0;

        virtual VOID VSetTextAppendDirection(AppendDirection dir) = 0;

        virtual VOID VAppendText(CONST std::string& text) = 0;

        virtual VOID VAppendText(LPCSTR text) { VAppendText(std::string(text)); }

        virtual CONST std::vector<TextLine>& VGetTextLines(VOID) CONST = 0;

        virtual VOID VSetTextColor(CONST Color& c) = 0;

        virtual VOID VClearText(VOID) = 0;

        virtual ~IGuiTextComponent(VOID) { }
    };

    class IGuiTextInputComponent : public virtual IGuiComponent
    {
    public:
        virtual VOID VAddChar(CHAR c) = 0;

        virtual VOID VRemoveChar(VOID) = 0;

        virtual CONST std::string& VGetText(VOID) = 0;

        virtual VOID VSetText(CONST std::string& text) = 0;

        virtual VOID VSetTextColor(CONST Color& c) = 0;

        virtual VOID VClearText(VOID) = 0;
    };

    class IGuiFactory
    {
    public:
        virtual IGuiRectangle* VCreateRectangle(VOID) = 0;

        virtual IGuiTextureComponent* VCreateTextureComponent(VOID) = 0;

        virtual IGuiTextComponent* VCreateTextComponent(VOID) = 0;

        virtual IGuiTextInputComponent* VCreateTextInputComponent(VOID) = 0;

        virtual IGui* VCreateGui(VOID) = 0;
    };

    class IGuiLookAndFeel
    {
    public:
        virtual Color VGetBackgroundColor(VOID) = 0;
        
        virtual Color VGetForegroundColor(VOID) = 0;

        virtual Color VGetFontColor(VOID) = 0;
    };
}