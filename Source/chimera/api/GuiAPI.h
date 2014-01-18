#pragma once
#include "CMTypes.h"
#include "CMCommon.h"

#define GUI_SHADER_FILE L"Gui.hlsl"

namespace chimera
{
    class IGuiComponent : virtual public IScreenElement, public InputAdapter
    {
    public:
        virtual void VSetEffect(LPCSTR ps) = 0;

        virtual void VSetFocus(bool focus) = 0;

        virtual void VSetEnabled(bool enable) = 0;

        virtual ~IGuiComponent(void) { }
    };

    class IGui : public virtual IScreenElementContainer, public InputAdapter
    {
    public:
        virtual void VAddComponent(IGuiComponent* cmp) = 0;

        virtual ~IGui(void) {}
    };

    class IGuiRectangle : public virtual IGuiComponent
    {

    };

    class IGuiTextureComponent : public virtual IGuiComponent
    {
    public:
        virtual void VSetTexture(LPCSTR texture) = 0;

        virtual ~IGuiTextureComponent(void) { }
    };

    class IGuiTextComponent : public virtual IGuiComponent
    {
    public:
        virtual void VSetAlignment(Alignment e) = 0;

        virtual void VSetTextAppendDirection(AppendDirection dir) = 0;

        virtual void VAppendText(const std::string& text) = 0;

        virtual void VAppendText(LPCSTR text) { VAppendText(std::string(text)); }

        virtual const std::vector<TextLine>& VGetTextLines(void) const = 0;

        virtual void VSetTextColor(const Color& c) = 0;

        virtual void VClearText(void) = 0;

        virtual ~IGuiTextComponent(void) { }
    };

    class IGuiTextInputComponent : public virtual IGuiComponent
    {
    public:
        virtual void VAddChar(char c) = 0;

        virtual void VRemoveChar(void) = 0;

        virtual const std::string& VGetText(void) = 0;

        virtual void VSetText(const std::string& text) = 0;

        virtual void VSetTextColor(const Color& c) = 0;

        virtual void VClearText(void) = 0;
    };

    class IGuiFactory
    {
    public:
        virtual IGuiRectangle* VCreateRectangle(void) = 0;

        virtual IGuiTextureComponent* VCreateTextureComponent(void) = 0;

        virtual IGuiTextComponent* VCreateTextComponent(void) = 0;

        virtual IGuiTextInputComponent* VCreateTextInputComponent(void) = 0;

        virtual IGui* VCreateGui(void) = 0;
    };

    class IGuiLookAndFeel
    {
    public:
        virtual Color VGetBackgroundColor(void) = 0;
        
        virtual Color VGetForegroundColor(void) = 0;

        virtual Color VGetFontColor(void) = 0;
    };
}