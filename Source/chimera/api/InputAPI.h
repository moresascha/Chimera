#pragma once
#include "CMTypes.h"

namespace chimera
{
    CM_INLINE CHAR GetCharFromVK(UINT key)
    {
        return MapVirtualKey(key, MAPVK_VK_TO_CHAR);
    }

    CM_INLINE UINT GetVKFromchar(CHAR key)
    {
        return VkKeyScan(key);
    }

    class IInputHandler
    {
    public:
        virtual BOOL VOnMessage(CONST MSG& msg) = 0;

        virtual VOID VOnUpdate(VOID) = 0;

        virtual BOOL VInit(CM_INSTANCE hinstance, CM_HWND hwnd, UINT width, UINT height) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual BOOL VGrabMouse(BOOL aq) = 0;

        virtual BOOL VIsMouseGrabbed(VOID) = 0;

        virtual VOID VSetCurserOffsets(INT x, INT y) {}

        virtual BOOL VIsKeyDown(UINT code) = 0;

        virtual VOID VPushKeyListener(IKeyListener* listener) = 0;

        virtual VOID VPopKeyListener(VOID) = 0;

        virtual VOID VPushMouseListener(IMouseListener* listener) = 0;

        virtual VOID VPopMouseListener(VOID) = 0;

        virtual VOID VRemoveKeyListener(IKeyListener* listener) = 0;

        virtual VOID VRemoveMouseListener(IMouseListener* listener) = 0;

        virtual ~IInputHandler(VOID) {}
    };

    class IKeyListener 
    {
    public:
        virtual BOOL VOnKeyDown(UINT CONST code) = 0;

        virtual BOOL VOnKeyPressed(UINT CONST code) = 0;

        virtual BOOL VOnKeyReleased(UINT CONST code) = 0;

        virtual BOOL VOnKeyRepeat(UINT CONST code) = 0;
    };

    class IMouseListener 
    {
    public:
        virtual BOOL VOnMouseButtonPressed(INT x, INT y, INT button) = 0;

        virtual BOOL VOnMouseButtonDown(INT x, INT y, INT button) = 0;

        virtual BOOL VOnMouseButtonReleased(INT x, INT y, INT button) = 0;

        virtual BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy) = 0;

        virtual BOOL VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button) = 0;

        virtual BOOL VOnMouseWheel(INT x, INT y, INT delta) = 0;
    };

    class IInputFactory
    {
    public:
        virtual IInputHandler* VCreateInputHanlder(VOID) = 0;
    };

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
        std::vector<KeyboardButtonPressedListener> m_keyBoardButtonPressedListener;
        std::vector<KeyboardButtonDownListener> m_keyBoardButtonDownListener;
        std::vector<KeyboardButtonReleasedListener> m_keyBoardButtonReleasedListener;
        std::vector<KeyboardButtonRepeatListener> m_keyBoardButtonRepeatListener;

        std::vector<MouseButtonPressedListener> m_mouseButtonPressedListener;
        std::vector<MouseButtonReleasedListener> m_mouseButtonReleasedListener;
        std::vector<MouseButtonDownListener> m_mouseButtonDownListener;

        std::vector<MouseDraggedListener> m_mouseDraggedListener;

        std::vector<MouseWheelListener> m_mouseWheelListener;
        std::vector<MouseMovedListener> m_mouseMovedListener;

        BOOL m_active;

    public:

        CM_INLINE InputAdapter(VOID);

        CM_INLINE virtual VOID ActivateInput(VOID);

        CM_INLINE virtual VOID DeactivateInput(VOID);

        CM_INLINE BOOL IsInputActive(VOID);

        CM_INLINE VOID AddKeyPressedListener(KeyboardButtonPressedListener listener);

        CM_INLINE VOID AddKeyReleasedListener(KeyboardButtonReleasedListener listener);

        CM_INLINE VOID AddKeyDownListener(KeyboardButtonDownListener listener);

        CM_INLINE VOID AddKeyRepeatListener(KeyboardButtonRepeatListener listener);

        CM_INLINE VOID AddMousePressedListener(MouseButtonPressedListener listener);

        CM_INLINE VOID AddMouseReleasedListener(MouseButtonReleasedListener listener);

        CM_INLINE VOID AddMouseDownListener(MouseButtonDownListener listener);

        CM_INLINE VOID AddMouseMovedListener(MouseMovedListener listener);

        CM_INLINE VOID AddMouseDraggedListener(MouseDraggedListener listener);

        CM_INLINE VOID AddMouseScrollListener(MouseWheelListener listener);

        CM_INLINE BOOL VOnMouseButtonPressed(INT x, INT y, INT button);

        CM_INLINE BOOL VOnMouseButtonDown(INT x, INT y, INT button);

        CM_INLINE BOOL VOnMouseButtonReleased(INT x, INT y, INT button);

        CM_INLINE BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy);

        CM_INLINE BOOL VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button);

        CM_INLINE BOOL VOnMouseWheel(INT x, INT y, INT delta);

        CM_INLINE BOOL VOnKeyDown(UINT CONST code);

        CM_INLINE BOOL VOnKeyPressed(UINT CONST code);

        CM_INLINE BOOL VOnKeyReleased(UINT CONST code);

        CM_INLINE BOOL VOnKeyRepeat(UINT CONST code);

        virtual ~InputAdapter(VOID) { }
    };

    InputAdapter::InputAdapter(VOID) : m_active(FALSE)
    {

    }

    BOOL InputAdapter::IsInputActive(VOID)
    {
        return m_active;
    }

    VOID InputAdapter::ActivateInput(VOID)
    {
        if(!m_active)
        {
            CmGetApp()->VGetInputHandler()->VPushKeyListener(this);
            CmGetApp()->VGetInputHandler()->VPushMouseListener(this);
            m_active = TRUE;
        }
    }

    VOID InputAdapter::DeactivateInput(VOID)
    {
        if(m_active)
        {
            CmGetApp()->VGetInputHandler()->VPopKeyListener();
            CmGetApp()->VGetInputHandler()->VPopMouseListener();
            m_active = FALSE;
        }
    }

    VOID InputAdapter::AddKeyPressedListener(KeyboardButtonPressedListener listener)
    {
        m_keyBoardButtonPressedListener.push_back(listener);
    }

    VOID InputAdapter::AddKeyReleasedListener(KeyboardButtonReleasedListener listener)
    {
        m_keyBoardButtonReleasedListener.push_back(listener);
    }

    VOID InputAdapter::AddKeyDownListener(KeyboardButtonDownListener listener)
    {
        m_keyBoardButtonDownListener.push_back(listener);;
    }

    VOID InputAdapter::AddKeyRepeatListener(KeyboardButtonRepeatListener listener)
    {
        m_keyBoardButtonRepeatListener.push_back(listener);
    }

    VOID InputAdapter::AddMousePressedListener(MouseButtonPressedListener listener)
    {
        m_mouseButtonPressedListener.push_back(listener);
    }

    VOID InputAdapter::AddMouseScrollListener(MouseWheelListener listener)
    {
        m_mouseWheelListener.push_back(listener);
    }

    BOOL InputAdapter::VOnMouseButtonPressed(INT x, INT y, INT button)
    {
        TBD_FOR(m_mouseButtonPressedListener)
        {
            (*it)(x, y, button);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnMouseButtonDown(INT x, INT y, INT button)
    {
        TBD_FOR(m_mouseButtonDownListener)
        {
            (*it)(x, y, button);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnMouseButtonReleased(INT x, INT y, INT button)
    {
        TBD_FOR(m_mouseButtonPressedListener)
        {
            (*it)(x, y, button);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnMouseMoved(INT x, INT y, INT dx, INT dy)
    {
        TBD_FOR(m_mouseMovedListener)
        {
            (*it)(x, y, dx, dy);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button)
    {
        TBD_FOR(m_mouseDraggedListener)
        {
            (*it)(x, y, dx, dy, button);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnMouseWheel(INT x, INT y, INT delta)
    {
        TBD_FOR(m_mouseWheelListener)
        {
            (*it)(x, y, delta);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnKeyDown(UINT CONST code)
    {
        TBD_FOR(m_keyBoardButtonDownListener)
        {
            (*it)(code);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnKeyPressed(UINT CONST code)
    {
        TBD_FOR(m_keyBoardButtonPressedListener)
        {
            (*it)(code);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnKeyReleased(UINT CONST code)
    {
        TBD_FOR(m_keyBoardButtonReleasedListener)
        {
            (*it)(code);
        }
        return TRUE;
    }

    BOOL InputAdapter::VOnKeyRepeat(UINT CONST code)
    {
        TBD_FOR(m_keyBoardButtonRepeatListener)
        {
            (*it)(code);
        }
        return TRUE;
    }
}