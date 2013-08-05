#pragma once
#include "stdafx.h"

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#include <algorithm>
#include "tbdStack.h"

namespace tbd 
{
    #define KEY_TAB 0x09
    #define KEY_0 0x30
    #define KEY_1 0x31
    #define KEY_2 0x32
    #define KEY_3 0x33
    #define KEY_4 0x34
    #define KEY_5 0x35
    #define KEY_6 0x36
    #define KEY_7 0x37
    #define KEY_8 0x38
    #define KEY_9 0x39
    #define KEY_A 0x41
    #define KEY_B 0x42
    #define KEY_C 0x43
    #define KEY_D 0x44
    #define KEY_E 0x45
    #define KEY_F 0x46
    #define KEY_G 0x47
    #define KEY_H 0x48
    #define KEY_I 0x49
    #define KEY_J 0x4A
    #define KEY_K 0x4B
    #define KEY_L 0x4C
    #define KEY_M 0x4D
    #define KEY_N 0x4E
    #define KEY_O 0x4F
    #define KEY_P 0x50
    #define KEY_Q 0x51
    #define KEY_R 0x52
    #define KEY_S 0x53
    #define KEY_T 0x54
    #define KEY_U 0x55
    #define KEY_V 0x56
    #define KEY_W 0x57
    #define KEY_X 0x58
    #define KEY_Y 0x59
    #define KEY_Z 0x5A
    #define KEY_SPACE VK_SPACE
    #define KEY_LSHIFT VK_SHIFT
    #define KEY_ESC VK_ESCAPE
    #define KEY_RETURN VK_RETURN
    #define KEY_DELETE VK_DELETE
    #define KEY_SPACE VK_SPACE
    #define KEY_BACKSPACE VK_BACK
    #define KEY_ARROW_DOWN VK_DOWN
    #define KEY_ARROW_UP VK_UP
    #define KEY_CIRCUMFLEX VK_OEM_5 //todo

    #define MOUSE_BTN_LEFT VK_LBUTTON
    #define MOUSE_BTN_RIGHT VK_RBUTTON


    #define KEY_DOWN 0x01
    #define KEY_RELEASED 0x02
    #define KEY_PRESSED 0x03
    #define MOUSE_BUTTON_DOWN 0x04
    #define MOUSE_BUTTON_PRESSED 0x06
    #define MOUSE_BUTTON_RELEASED 0x05
    #define MOUSE_WHEEL 0x07
    #define TBD_MOUSE_MOVED 0x08
    #define MOUSE_DRAGGED 0x09
    #define KEY_REPEAT 0x0A

    UINT GetVKFromchar(CHAR c);
    CHAR GetCharFromVK(UINT c);

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

    class DLL_EXPORT IKeyListener 
    {
    public:
        virtual BOOL VOnKeyDown(UINT CONST code) = 0;
        virtual BOOL VOnKeyPressed(UINT CONST code) = 0;
        virtual BOOL VOnKeyReleased(UINT CONST code) = 0;
        virtual BOOL VOnKeyRepeat(UINT CONST code) = 0;
    };
    class DLL_EXPORT IMouseListener 
    {
    public:
        virtual BOOL VOnMouseButtonPressed(INT x, INT y, INT button) = 0;
        virtual BOOL VOnMouseButtonDown(INT x, INT y, INT button) = 0;
        virtual BOOL VOnMouseButtonReleased(INT x, INT y, INT button) = 0;
        virtual BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy) = 0;
        virtual BOOL VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button) = 0;
        virtual BOOL VOnMouseWheel(INT x, INT y, INT delta) = 0;
    };

    class DLL_EXPORT KeyAdapterListener : public IKeyListener
    {
    public:
        virtual BOOL VOnKeyDown(UINT CONST code) { return TRUE; }
        virtual BOOL VOnKeyPressed(UINT CONST code) { return TRUE; }
        virtual BOOL VOnKeyReleased(UINT CONST code) { return TRUE; }
        virtual BOOL VOnKeyRepeat(UINT CONST code) { return TRUE; }
    };
    class DLL_EXPORT MouseAdapterListener : public IMouseListener
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

    class DLL_EXPORT InputHandler
    {
    private:
        util::tbdStack<IKeyListener*> m_keyStack;
        util::tbdStack<IMouseListener*> m_mouseStack;

    protected:
        BOOL UpdateListener(InputMessage msg);
        BOOL m_isKeyDown[0xFE];

    public:
        InputHandler(VOID);
    
        virtual BOOL VOnMessage(CONST MSG& msg) = 0;

        VOID OnUpdate(VOID);
        
        virtual BOOL VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual BOOL VIsEscPressed(VOID) = 0;

        virtual BOOL VGrabMouse(BOOL aq) = 0;

        virtual BOOL VIsMouseGrabbed(VOID) = 0;

        virtual VOID VSetCurserOffsets(INT x, INT y) {}

        BOOL IsKeyDown(UINT code)
        {
            return m_isKeyDown[code];
        }

        VOID PushKeyListener(IKeyListener* listener);

        VOID PopKeyListener(VOID);

        VOID PushMouseListener(IMouseListener* listener);

        VOID PopMouseListener(VOID);

        /*
        //interface
        virtual BOOL VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual BOOL VUpdateState(CONST MSG& msg) = 0;

        virtual BOOL VIsKeyDown(UCHAR key) = 0;

        virtual BOOL VIsKeyPressed(UCHAR key) = 0;

        virtual BOOL VIsKeyReleased(UCHAR key) = 0;

        virtual BOOL VIsEscPressed(VOID) = 0;

        virtual BOOL VGrabMouse(BOOL aq) = 0;

        virtual BOOL VIsMouseGrabbed(VOID) = 0;

        virtual BOOL VIsMouseButtonPressed(INT button) = 0;

        virtual BOOL VIsMouseButtonReleased(INT button) = 0;

        virtual BOOL VIsMouseButtonDown(INT button) = 0;

        virtual INT VGetDMouseWheel(VOID) = 0;

        virtual INT VGetPosX(VOID) = 0;

        virtual INT VGetPosY(VOID) = 0;

        virtual INT VGetDX(VOID) = 0;

        virtual INT VGetDY(VOID) = 0; */

        virtual ~InputHandler(VOID);
    };

    class DefaultWinInputHandler : public InputHandler
    {
    private:
        BOOL m_isMouseGrabbed;
        BOOL m_isEscapePressed;
        UINT m_lastMousePosX;
        UINT m_lastMousePosY;
        INT m_curserXOffset;
        INT m_curserYOffset;
        tagPOINT m_mousePosition;
        HWND m_hwnd;

    public:
        DefaultWinInputHandler(VOID);

        BOOL VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height);

        BOOL VOnRestore(VOID);

        BOOL VOnMessage(CONST MSG&);

        BOOL VIsEscPressed(VOID);

        BOOL VGrabMouse(BOOL aq);

        BOOL VIsMouseGrabbed(VOID);

        VOID VSetCurserOffsets(INT x, INT y);

        /*
        BOOL VIsKeyDown(UCHAR key);

        BOOL VIsKeyPressed(UCHAR key);

        BOOL VIsKeyReleased(UCHAR key);

        BOOL VIsMouseButtonPressed(INT button);

        BOOL VIsMouseButtonReleased(INT button);

        BOOL VIsMouseButtonDown(INT button);

        INT VGetDMouseWheel(VOID);

        INT VGetPosX(VOID);

        INT VGetPosY(VOID);

        INT VGetDX(VOID);

        INT VGetDY(VOID);
        */
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
    typedef fastdelegate::FastDelegate1<UINT> KeyboardButtonPressedListener;
    typedef fastdelegate::FastDelegate1<UINT> KeyboardButtonReleasedListener;
    typedef fastdelegate::FastDelegate1<UINT> KeyboardButtonDownListener;
    typedef fastdelegate::FastDelegate1<UINT> KeyboardButtonRepeatListener;

    typedef fastdelegate::FastDelegate3<INT, INT, INT> MouseButtonPressedListener;
    typedef fastdelegate::FastDelegate3<INT, INT, INT> MouseButtonReleasedListener;
    typedef fastdelegate::FastDelegate3<INT, INT, INT> MouseButtonDownListener;
    typedef fastdelegate::FastDelegate3<INT, INT, INT> MouseWheelListener;
    typedef fastdelegate::FastDelegate4<INT, INT, INT, INT> MouseMovedListener;
    typedef fastdelegate::FastDelegate5<INT, INT, INT, INT, INT> MouseDraggedListener;

    class InputAdapter : public IKeyListener, public IMouseListener
    {
    protected:
        std::list<KeyboardButtonPressedListener> m_keyBoardButtonPressedListener;
        std::list<KeyboardButtonDownListener> m_keyBoardButtonDownListener;
        std::list<KeyboardButtonReleasedListener> m_keyBoardButtonReleasedListener;
        std::list<KeyboardButtonRepeatListener> m_keyBoardButtonRepeatListener;

        std::list<MouseButtonPressedListener> m_mouseButtonPressedListener;
        std::list<MouseButtonReleasedListener> m_mouseButtonReleasedListener;
        std::list<MouseButtonDownListener> m_mouseButtonDownListener;

        std::list<MouseDraggedListener> m_mouseDraggedListener;

        std::list<MouseWheelListener> m_mouseWheelListener;
        std::list<MouseMovedListener> m_mouseMovedListener;

        BOOL m_active;

    public:

        InputAdapter(VOID);

        virtual VOID Activate(VOID);

        virtual VOID Deactivate(VOID);

        BOOL IsActive(VOID);

        VOID AddKeyPressedListener(KeyboardButtonPressedListener listener);

        VOID AddKeyReleasedListener(KeyboardButtonReleasedListener listener);

        VOID AddKeyDownListener(KeyboardButtonDownListener listener);

        VOID AddKeyRepeatListener(KeyboardButtonRepeatListener listener);

        VOID AddMousePressedListener(MouseButtonPressedListener listener);

        VOID AddMouseReleasedListener(MouseButtonReleasedListener listener);

        VOID AddMouseDownListener(MouseButtonDownListener listener);

        VOID AddMouseMovedListener(MouseMovedListener listener);
        
        VOID AddMouseDraggedListener(MouseDraggedListener listener);

        VOID AddMouseScrollListener(MouseWheelListener listener);

        BOOL VOnMouseButtonPressed(INT x, INT y, INT button);

        BOOL VOnMouseButtonDown(INT x, INT y, INT button);

        BOOL VOnMouseButtonReleased(INT x, INT y, INT button);

        BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy);

        BOOL VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button);

        BOOL VOnMouseWheel(INT x, INT y, INT delta);

        BOOL VOnKeyDown(UINT CONST code);

        BOOL VOnKeyPressed(UINT CONST code);

        BOOL VOnKeyReleased(UINT CONST code);

        BOOL VOnKeyRepeat(UINT CONST code);

        virtual ~InputAdapter(VOID) { }
    };
}
