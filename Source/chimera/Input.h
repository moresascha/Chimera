#pragma once
#include "stdafx.h"
#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#include "tbdStack.h"

namespace chimera 
{
    struct InputMessage
    {
        UCHAR message;
        INT mouseX;
        INT mouseDX;
        UINT mouseY;
        UINT mouseDY;
        UINT mouseWheelDelta;
        UINT button;

        InputMessage(VOID) : message(0), mouseX(0), mouseY(0), mouseDX(0), mouseDY(0), mouseWheelDelta(0), button(0)
        {

        }
    };

    class KeyAdapterListener : public IKeyListener
    {
    public:
        virtual BOOL VOnKeyDown(UINT CONST code) { return TRUE; }
        virtual BOOL VOnKeyPressed(UINT CONST code) { return TRUE; }
        virtual BOOL VOnKeyReleased(UINT CONST code) { return TRUE; }
        virtual BOOL VOnKeyRepeat(UINT CONST code) { return TRUE; }
    };
    class MouseAdapterListener : public IMouseListener
    {
    public:
        virtual BOOL VOnMouseButtonPressed(INT x, INT y, INT button) { return TRUE; }
        virtual BOOL VOnMouseButtonDown(INT x, INT y, INT button) { return TRUE; }
        virtual BOOL VOnMouseButtonReleased(INT x, INT y, INT button) { return TRUE; }
        virtual BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy) { return TRUE; }
        virtual BOOL VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button) { return TRUE; }
        virtual BOOL VOnMousePressed(INT x, INT y, INT button) { return TRUE; }
        virtual BOOL VOnMouseWheel(INT x, INT y, INT delta) { return TRUE; }
    };

    class DefaultWinInputHandler : public IInputHandler
    {
    private:
        BOOL m_isMouseGrabbed;
        UINT m_lastMousePosX;
        UINT m_lastMousePosY;
        INT m_curserXOffset;
        INT m_curserYOffset;
        tagPOINT m_mousePosition;
        HWND m_hwnd;

    private:
        util::tbdStack<IKeyListener*> m_keyStack;
        util::tbdStack<IMouseListener*> m_mouseStack;

    protected:
        BOOL UpdateListener(InputMessage msg);
        BOOL m_isKeyDown[0xFE];

    public:
        DefaultWinInputHandler(VOID);

        BOOL VInit(CM_INSTANCE hinstance, CM_HWND hwnd, UINT width, UINT height);

        BOOL VOnRestore(VOID);

        VOID VOnUpdate(VOID);

        BOOL VOnMessage(CONST MSG&);

        BOOL VGrabMouse(BOOL aq);

        BOOL VIsMouseGrabbed(VOID);

        VOID VSetCurserOffsets(INT x, INT y);

        VOID VPushKeyListener(IKeyListener* listener);

        VOID VPopKeyListener(VOID);

        VOID VPushMouseListener(IMouseListener* listener);

        VOID VPopMouseListener(VOID);

        VOID VRemoveKeyListener(IKeyListener* listener);

        VOID VRemoveMouseListener(IMouseListener* listener);
        
        BOOL VIsKeyDown(UINT code)
        {
            return m_isKeyDown[code];
        }
    };

    /*
    class __declspec(deprecated("** DirectInput is deprecated **")) DirectInput : public InputHandler
    {
    private:
        IDirectInput8* m_directInput;
        IDirectInputDevice8* m_mouse;
        IDirectInputDevice8* m_keyBoard;
        HINSTANCE m_hinstance;
        HWND m_hwnd;
        UINT m_width, m_height;
        tagPOINT m_mousePosition;
        BOOL m_initialized;
        BOOL m_mouseGrabbed;
        DIMOUSESTATE m_mouseState[2];
        UINT m_mouseX, m_mouseY;
        UCHAR m_keys[2][256];
        UINT m_currentLayer;
        VOID Delete(VOID);
        BOOL ReadKeyboardState(VOID);
        BOOL ReadMouseState(VOID);

    public:

        DirectInput(VOID);

        BOOL VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height);

        BOOL VOnRestore(UINT width, UINT height);

        BOOL VIsKeyDown(UCHAR key);

        BOOL VIsKeyPressed(UCHAR key);

        BOOL VIsKeyReleased(UCHAR key);

        BOOL VIsMouseButtonPressed(INT button);

        BOOL VIsMouseButtonReleased(INT button);

        BOOL VIsMouseButtonDown(INT button);

        INT VGetDMouseWheel(VOID);

        BOOL VIsMouseGrabbed(VOID);

        BOOL VGrabMouse(BOOL grab);

        INT VGetPosX(VOID);

        INT VGetPosY(VOID);

        INT VGetDX(VOID);

        INT VGetDY(VOID);

        BOOL VUpdateState(VOID);

        BOOL VIsEscPressed(VOID);

        ~DirectInput(VOID);
    }; */
}
