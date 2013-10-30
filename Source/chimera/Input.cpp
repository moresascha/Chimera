#include "Input.h"

namespace chimera 
{
    KeyAdapterListener g_keyAdapter;
    MouseAdapterListener g_mouseAdapter;

    CHAR GetCharFromVK(UINT key)
    {
        return MapVirtualKey(key, MAPVK_VK_TO_CHAR);
    }

    UINT GetVKFromchar(CHAR key)
    {
        return VkKeyScan(key);
    }

    DefaultWinInputHandler::DefaultWinInputHandler(VOID) : m_hwnd(NULL)
    {
        for(UINT i = 0; i < 0xFE; ++i)
        {
            m_isKeyDown[i] = FALSE;
        }
        VPushKeyListener(&g_keyAdapter);
        VPushMouseListener(&g_mouseAdapter);

        VOnRestore();
    }

    VOID DefaultWinInputHandler::VPushKeyListener(IKeyListener* listener)
    {
        m_keyStack.Push(listener);
    }

    VOID DefaultWinInputHandler::VPopKeyListener(VOID)
    {
        m_keyStack.Pop();
    }

    VOID DefaultWinInputHandler::VPushMouseListener(IMouseListener* listener)
    {
        m_mouseStack.Push(listener);
    }

    VOID DefaultWinInputHandler::VPopMouseListener(VOID)
    {
        m_mouseStack.Pop();
    }

    VOID DefaultWinInputHandler::VRemoveKeyListener(IKeyListener* listener)
    {
        m_keyStack.Remove(listener);
    }

    VOID DefaultWinInputHandler::VRemoveMouseListener(IMouseListener* listener)
    {
        m_mouseStack.Remove(listener);
    }

    BOOL DefaultWinInputHandler::UpdateListener(InputMessage msg) {
        BOOL result = 0;
        
        switch(msg.message) 
        {
        case KEY_DOWN: 
            {
                m_keyStack.Front()->VOnKeyDown(msg.button);
            } break;
        case KEY_RELEASED: 
            {
                m_keyStack.Front()->VOnKeyReleased(msg.button);
            } break;
        case KEY_PRESSED: 
            {
                m_keyStack.Front()->VOnKeyPressed(msg.button);
            } break;
        case KEY_REPEAT: 
            {
                m_keyStack.Front()->VOnKeyRepeat(msg.button);
            } break;
        case MOUSE_BUTTON_DOWN:
            {
                m_mouseStack.Front()->VOnMouseButtonDown(msg.mouseX, msg.mouseY, msg.button);
            } break;
        case MOUSE_BUTTON_RELEASED:
            {
                m_mouseStack.Front()->VOnMouseButtonReleased(msg.mouseX, msg.mouseY, msg.button);
            } break;

        case MOUSE_BUTTON_PRESSED:
            {
                m_mouseStack.Front()->VOnMouseButtonPressed(msg.mouseX, msg.mouseY, msg.button);
            } break;
        case TBD_MOUSE_MOVED:
            {
                m_mouseStack.Front()->VOnMouseMoved(msg.mouseX, msg.mouseY, msg.mouseDX, msg.mouseDY);
            } break;
        case MOUSE_DRAGGED:
            {
                m_mouseStack.Front()->VOnMouseDragged(msg.mouseX, msg.mouseY, msg.mouseDX, msg.mouseDY, msg.button);
            } break;
        case MOUSE_WHEEL:
            {
                m_mouseStack.Front()->VOnMouseWheel(msg.mouseY, msg.mouseX, msg.mouseWheelDelta);
            } break;
        }
        return result;
    }

    VOID DefaultWinInputHandler::VOnUpdate(VOID)
    {
        for(UINT i = 0; i < 0xFE; ++i)
        {
            if(m_isKeyDown[i])
            {
                InputMessage keymsg;
                keymsg.button = i;
                keymsg.message = KEY_DOWN;
                UpdateListener(keymsg);
            }
        }
    }


    BOOL DefaultWinInputHandler::VInit(CM_INSTANCE hinstance, CM_HWND hwnd, UINT width, UINT height)
    {
        m_hwnd = (HWND)hwnd;
        return TRUE;
    }

    BOOL DefaultWinInputHandler::VOnRestore(VOID)
    {
        m_isMouseGrabbed = FALSE;
        m_lastMousePosX = 0; //todo
        m_lastMousePosY = 0; //todo
        //VGrabMouse(FALSE);
        return TRUE;
    }

    VOID DefaultWinInputHandler::VSetCurserOffsets(INT x, INT y)
    {
        m_curserXOffset = x;
        m_curserYOffset = y;
    }

    BOOL DefaultWinInputHandler::VOnMessage(CONST MSG& msg)
    {
        switch(msg.message)
        {
        case WM_MOUSEMOVE:
            {
                INT posx = GET_X_LPARAM(msg.lParam);
                INT posy = GET_Y_LPARAM(msg.lParam);
                InputMessage imsg;

                imsg.mouseX = posx;
                imsg.mouseY = posy;

                if(m_isMouseGrabbed)
                {
                    RECT rc;
                    GetWindowRect(m_hwnd, &rc);
                    INT x = (rc.right - rc.left) / 2;
                    INT y = (rc.bottom - rc.top) / 2;

                    imsg.mouseDY = posy - y;
                    imsg.mouseDX = posx - x;
                    imsg.message = TBD_MOUSE_MOVED;

                    if(imsg.mouseDX != 0 || imsg.mouseDY != 0)
                    {
                        //add 8 and 30 pixels to compensate borders, TODO
                        SetCursorPos(rc.left + x + m_curserXOffset, rc.top + y + m_curserYOffset);
                    }
                }
                else
                {
                    imsg.mouseDY = posy - m_lastMousePosY;
                    imsg.mouseDX = posx - m_lastMousePosX;
                    imsg.message = TBD_MOUSE_MOVED;

                    m_lastMousePosX = posx;
                    m_lastMousePosY = posy;
                }
                if(imsg.mouseDX != 0 || imsg.mouseDY)
                {
                    UpdateListener(imsg);
                }

            } break;
        case WM_LBUTTONDOWN:
            {
                INT posx = GET_X_LPARAM(msg.lParam);
                INT posy = GET_X_LPARAM(msg.lParam);
                InputMessage imsg;
                imsg.mouseX = posx;
                imsg.mouseY = posy;
                imsg.button = MOUSE_BTN_LEFT;
                imsg.message = MOUSE_BUTTON_PRESSED;
                UpdateListener(imsg);
            } break;
        case WM_RBUTTONDOWN:
            {
                INT posx = GET_X_LPARAM(msg.lParam);
                INT posy = GET_X_LPARAM(msg.lParam);
                InputMessage imsg;
                imsg.mouseX = posx;
                imsg.mouseY = posy;
                imsg.button = MOUSE_BTN_RIGHT;
                imsg.message = MOUSE_BUTTON_PRESSED;
                UpdateListener(imsg);
            } break;
        case WM_LBUTTONUP:
            {
                INT posx = GET_X_LPARAM(msg.lParam);
                INT posy = GET_X_LPARAM(msg.lParam);
                InputMessage imsg;
                imsg.mouseX = posx;
                imsg.mouseY = posy;
                imsg.button = MOUSE_BTN_LEFT;
                imsg.message = MOUSE_BUTTON_RELEASED;
                UpdateListener(imsg);
            } break;
        case WM_RBUTTONUP:
            {
                INT posx = GET_X_LPARAM(msg.lParam);
                INT posy = GET_X_LPARAM(msg.lParam);
                InputMessage imsg;
                imsg.mouseX = posx;
                imsg.mouseY = posy;
                imsg.button = MOUSE_BTN_RIGHT;
                imsg.message = MOUSE_BUTTON_RELEASED;
                UpdateListener(imsg);
            } break;
        case WM_KEYDOWN:
            {
                InputMessage imsg;
                imsg.button = (UINT)msg.wParam;
                imsg.message = (KF_REPEAT & HIWORD(msg.lParam)) ? KEY_REPEAT : KEY_PRESSED;
                UpdateListener(imsg);
                m_isKeyDown[imsg.button] = TRUE;

            } break;
        case WM_KEYUP:
            {
                InputMessage imsg;
                imsg.button = (UINT)msg.wParam;
                imsg.message = KEY_RELEASED;
                UpdateListener(imsg);
                m_isKeyDown[imsg.button] = FALSE;
            } break;
        case WM_MOUSEWHEEL:
            {
                INT posx = GET_X_LPARAM(msg.lParam);
                INT posy = GET_X_LPARAM(msg.lParam);
                InputMessage imsg;
                imsg.mouseX = posx;
                imsg.mouseY = posy;
                imsg.message = MOUSE_WHEEL;
                imsg.mouseWheelDelta = GET_WHEEL_DELTA_WPARAM(msg.wParam);
                UpdateListener(imsg);
            } break;
        }
        return TRUE;
    }

    BOOL DefaultWinInputHandler::VGrabMouse(BOOL grab)
    {
        if(grab)
        {
            while(ShowCursor(FALSE) >= 0);
            GetCursorPos(&m_mousePosition);
            SetCapture(m_hwnd);
            /*RECT rc;
            GetWindowRect(m_hwnd, &rc);
            ClipCursor(&rc); */
        }
        else
        {
            SetCursor(LoadCursor(NULL, IDC_ARROW));
            while(ShowCursor(TRUE) <= 0);
            ReleaseCapture();
            if(m_isMouseGrabbed)
            {
                SetCursorPos(m_mousePosition.x, m_mousePosition.y);
            }
        }

        m_isMouseGrabbed = grab;
        return TRUE;
    }

    BOOL DefaultWinInputHandler::VIsMouseGrabbed(VOID)
    {
        return m_isMouseGrabbed;
    }

    //adapter

    InputAdapter::InputAdapter(VOID) : m_active(FALSE)
    {

    }

    BOOL InputAdapter::IsActive(VOID)
    {
        return m_active;
    }

    VOID InputAdapter::Activate(VOID)
    {
        if(!m_active)
        {
            chimera::CmGetApp()->VGetInputHandler()->VPushKeyListener(this);
            chimera::CmGetApp()->VGetInputHandler()->VPushMouseListener(this);
            m_active = TRUE;
        }
    }

    VOID InputAdapter::Deactivate(VOID)
    {
        if(m_active)
        {
            chimera::CmGetApp()->VGetInputHandler()->VPopKeyListener();
            chimera::CmGetApp()->VGetInputHandler()->VPopMouseListener();
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