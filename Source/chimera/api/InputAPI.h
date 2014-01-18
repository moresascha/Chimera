#pragma once
#include "CMTypes.h"

namespace chimera
{
    CM_INLINE char GetCharFromVK(uint key)
    {
        return MapVirtualKey(key, MAPVK_VK_TO_CHAR);
    }

    CM_INLINE uint GetVKFromchar(char key)
    {
        return VkKeyScan(key);
    }

    class IInputHandler
    {
    public:
        virtual bool VOnMessage(const MSG& msg) = 0;

        virtual void VOnUpdate(void) = 0;

        virtual bool VInit(CM_INSTANCE hinstance, CM_HWND hwnd, uint width, uint height) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual bool VGrabMouse(bool aq) = 0;

        virtual bool VIsMouseGrabbed(void) = 0;

        virtual void VSetCurserOffsets(int x, int y) {}

        virtual bool VIsKeyDown(uint code) = 0;

        virtual void VPushKeyListener(IKeyListener* listener) = 0;

        virtual void VPopKeyListener(void) = 0;

        virtual void VPushMouseListener(IMouseListener* listener) = 0;

        virtual void VPopMouseListener(void) = 0;

        virtual void VRemoveKeyListener(IKeyListener* listener) = 0;

        virtual void VRemoveMouseListener(IMouseListener* listener) = 0;

        virtual ~IInputHandler(void) {}
    };

    class IKeyListener 
    {
    public:
        virtual bool VOnKeyDown(uint const code) = 0;

        virtual bool VOnKeyPressed(uint const code) = 0;

        virtual bool VOnKeyReleased(uint const code) = 0;

        virtual bool VOnKeyRepeat(uint const code) = 0;
    };

    class IMouseListener 
    {
    public:
        virtual bool VOnMouseButtonPressed(int x, int y, int button) = 0;

        virtual bool VOnMouseButtonDown(int x, int y, int button) = 0;

        virtual bool VOnMouseButtonReleased(int x, int y, int button) = 0;

        virtual bool VOnMouseMoved(int x, int y, int dx, int dy) = 0;

        virtual bool VOnMouseDragged(int x, int y, int dx, int dy, int button) = 0;

        virtual bool VOnMouseWheel(int x, int y, int delta) = 0;
    };

    class IInputFactory
    {
    public:
        virtual IInputHandler* VCreateInputHanlder(void) = 0;
    };

    typedef fastdelegate::FastDelegate1<uint> KeyboardButtonPressedListener;
    typedef fastdelegate::FastDelegate1<uint> KeyboardButtonReleasedListener;
    typedef fastdelegate::FastDelegate1<uint> KeyboardButtonDownListener;
    typedef fastdelegate::FastDelegate1<uint> KeyboardButtonRepeatListener;

    typedef fastdelegate::FastDelegate3<int, int, int> MouseButtonPressedListener;
    typedef fastdelegate::FastDelegate3<int, int, int> MouseButtonReleasedListener;
    typedef fastdelegate::FastDelegate3<int, int, int> MouseButtonDownListener;
    typedef fastdelegate::FastDelegate3<int, int, int> MouseWheelListener;
    typedef fastdelegate::FastDelegate4<int, int, int, int> MouseMovedListener;
    typedef fastdelegate::FastDelegate5<int, int, int, int, int> MouseDraggedListener;

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

        bool m_active;

    public:

        CM_INLINE InputAdapter(void);

        CM_INLINE virtual void ActivateInput(void);

        CM_INLINE virtual void DeactivateInput(void);

        CM_INLINE bool IsInputActive(void);

        CM_INLINE void AddKeyPressedListener(KeyboardButtonPressedListener listener);

        CM_INLINE void AddKeyReleasedListener(KeyboardButtonReleasedListener listener);

        CM_INLINE void AddKeyDownListener(KeyboardButtonDownListener listener);

        CM_INLINE void AddKeyRepeatListener(KeyboardButtonRepeatListener listener);

        CM_INLINE void AddMousePressedListener(MouseButtonPressedListener listener);

        CM_INLINE void AddMouseReleasedListener(MouseButtonReleasedListener listener);

        CM_INLINE void AddMouseDownListener(MouseButtonDownListener listener);

        CM_INLINE void AddMouseMovedListener(MouseMovedListener listener);

        CM_INLINE void AddMouseDraggedListener(MouseDraggedListener listener);

        CM_INLINE void AddMouseScrollListener(MouseWheelListener listener);

        CM_INLINE bool VOnMouseButtonPressed(int x, int y, int button);

        CM_INLINE bool VOnMouseButtonDown(int x, int y, int button);

        CM_INLINE bool VOnMouseButtonReleased(int x, int y, int button);

        CM_INLINE bool VOnMouseMoved(int x, int y, int dx, int dy);

        CM_INLINE bool VOnMouseDragged(int x, int y, int dx, int dy, int button);

        CM_INLINE bool VOnMouseWheel(int x, int y, int delta);

        CM_INLINE bool VOnKeyDown(uint const code);

        CM_INLINE bool VOnKeyPressed(uint const code);

        CM_INLINE bool VOnKeyReleased(uint const code);

        CM_INLINE bool VOnKeyRepeat(uint const code);

        virtual ~InputAdapter(void) { }
    };

    InputAdapter::InputAdapter(void) : m_active(false)
    {

    }

    bool InputAdapter::IsInputActive(void)
    {
        return m_active;
    }

    void InputAdapter::ActivateInput(void)
    {
        if(!m_active)
        {
            CmGetApp()->VGetInputHandler()->VPushKeyListener(this);
            CmGetApp()->VGetInputHandler()->VPushMouseListener(this);
            m_active = true;
        }
    }

    void InputAdapter::DeactivateInput(void)
    {
        if(m_active)
        {
            CmGetApp()->VGetInputHandler()->VPopKeyListener();
            CmGetApp()->VGetInputHandler()->VPopMouseListener();
            m_active = false;
        }
    }

    void InputAdapter::AddKeyPressedListener(KeyboardButtonPressedListener listener)
    {
        m_keyBoardButtonPressedListener.push_back(listener);
    }

    void InputAdapter::AddKeyReleasedListener(KeyboardButtonReleasedListener listener)
    {
        m_keyBoardButtonReleasedListener.push_back(listener);
    }

    void InputAdapter::AddKeyDownListener(KeyboardButtonDownListener listener)
    {
        m_keyBoardButtonDownListener.push_back(listener);;
    }

    void InputAdapter::AddKeyRepeatListener(KeyboardButtonRepeatListener listener)
    {
        m_keyBoardButtonRepeatListener.push_back(listener);
    }

    void InputAdapter::AddMousePressedListener(MouseButtonPressedListener listener)
    {
        m_mouseButtonPressedListener.push_back(listener);
    }

    void InputAdapter::AddMouseScrollListener(MouseWheelListener listener)
    {
        m_mouseWheelListener.push_back(listener);
    }

    bool InputAdapter::VOnMouseButtonPressed(int x, int y, int button)
    {
        TBD_FOR(m_mouseButtonPressedListener)
        {
            (*it)(x, y, button);
        }
        return true;
    }

    bool InputAdapter::VOnMouseButtonDown(int x, int y, int button)
    {
        TBD_FOR(m_mouseButtonDownListener)
        {
            (*it)(x, y, button);
        }
        return true;
    }

    bool InputAdapter::VOnMouseButtonReleased(int x, int y, int button)
    {
        TBD_FOR(m_mouseButtonPressedListener)
        {
            (*it)(x, y, button);
        }
        return true;
    }

    bool InputAdapter::VOnMouseMoved(int x, int y, int dx, int dy)
    {
        TBD_FOR(m_mouseMovedListener)
        {
            (*it)(x, y, dx, dy);
        }
        return true;
    }

    bool InputAdapter::VOnMouseDragged(int x, int y, int dx, int dy, int button)
    {
        TBD_FOR(m_mouseDraggedListener)
        {
            (*it)(x, y, dx, dy, button);
        }
        return true;
    }

    bool InputAdapter::VOnMouseWheel(int x, int y, int delta)
    {
        TBD_FOR(m_mouseWheelListener)
        {
            (*it)(x, y, delta);
        }
        return true;
    }

    bool InputAdapter::VOnKeyDown(uint const code)
    {
        TBD_FOR(m_keyBoardButtonDownListener)
        {
            (*it)(code);
        }
        return true;
    }

    bool InputAdapter::VOnKeyPressed(uint const code)
    {
        TBD_FOR(m_keyBoardButtonPressedListener)
        {
            (*it)(code);
        }
        return true;
    }

    bool InputAdapter::VOnKeyReleased(uint const code)
    {
        TBD_FOR(m_keyBoardButtonReleasedListener)
        {
            (*it)(code);
        }
        return true;
    }

    bool InputAdapter::VOnKeyRepeat(uint const code)
    {
        TBD_FOR(m_keyBoardButtonRepeatListener)
        {
            (*it)(code);
        }
        return true;
    }
}