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
        int mouseX;
        int mouseDX;
        uint mouseY;
        uint mouseDY;
        uint mouseWheelDelta;
        uint button;

        InputMessage(void) : message(0), mouseX(0), mouseY(0), mouseDX(0), mouseDY(0), mouseWheelDelta(0), button(0)
        {

        }
    };

    class KeyAdapterListener : public IKeyListener
    {
    public:
        virtual bool VOnKeyDown(uint const code) { return true; }
        virtual bool VOnKeyPressed(uint const code) { return true; }
        virtual bool VOnKeyReleased(uint const code) { return true; }
        virtual bool VOnKeyRepeat(uint const code) { return true; }
    };
    class MouseAdapterListener : public IMouseListener
    {
    public:
        virtual bool VOnMouseButtonPressed(int x, int y, int button) { return true; }
        virtual bool VOnMouseButtonDown(int x, int y, int button) { return true; }
        virtual bool VOnMouseButtonReleased(int x, int y, int button) { return true; }
        virtual bool VOnMouseMoved(int x, int y, int dx, int dy) { return true; }
        virtual bool VOnMouseDragged(int x, int y, int dx, int dy, int button) { return true; }
        virtual bool VOnMousePressed(int x, int y, int button) { return true; }
        virtual bool VOnMouseWheel(int x, int y, int delta) { return true; }
    };

    class DefaultWinInputHandler : public IInputHandler
    {
    private:
        bool m_isMouseGrabbed;
        uint m_lastMousePosX;
        uint m_lastMousePosY;
        int m_curserXOffset;
        int m_curserYOffset;
        tagPOINT m_mousePosition;
        HWND m_hwnd;

    private:
        util::tbdStack<IKeyListener*> m_keyStack;
        util::tbdStack<IMouseListener*> m_mouseStack;

    protected:
        bool UpdateListener(InputMessage msg);
        bool m_isKeyDown[0xFE];

    public:
        DefaultWinInputHandler(void);

        bool VInit(CM_INSTANCE hinstance, CM_HWND hwnd, uint width, uint height);

        bool VOnRestore(void);

        void VOnUpdate(void);

        bool VOnMessage(const MSG&);

        bool VGrabMouse(bool aq);

        bool VIsMouseGrabbed(void);

        void VSetCurserOffsets(int x, int y);

        void VPushKeyListener(IKeyListener* listener);

        void VPopKeyListener(void);

        void VPushMouseListener(IMouseListener* listener);

        void VPopMouseListener(void);

        void VRemoveKeyListener(IKeyListener* listener);

        void VRemoveMouseListener(IMouseListener* listener);
        
        bool VIsKeyDown(uint code)
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
