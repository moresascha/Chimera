#include "Input.h"
#include <iostream>
#include "GameApp.h"

namespace tbd 
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

    InputHandler::InputHandler(VOID) 
    {
        for(UINT i = 0; i < 0xFE; ++i)
        {
            m_isKeyDown[i] = FALSE;
        }
        PushKeyListener(&g_keyAdapter);
        PushMouseListener(&g_mouseAdapter);
    }

   /* BOOL InputHandler::UpdateListener(InputMessage msg) {
        BOOL result = 0;
        switch(msg.message) 
        {
        case KEY_DOWN: 
            {
                for(auto it = this->m_keyListener.begin(); it!= this->m_keyListener.end(); ++it)
                {
                    result |= (*it)->VOnKeyDown(msg.button);
                }
            } break;
        case KEY_RELEASED: 
            {
                for(auto it = this->m_keyListener.begin(); it!= this->m_keyListener.end(); ++it)
                {
                    result |= (*it)->VOnKeyReleased(msg.button);
                }
            } break;
        case KEY_PRESSED: 
            {
                for(auto it = this->m_keyListener.begin(); it!= this->m_keyListener.end(); ++it)
                {
                    result |= (*it)->VOnKeyPressed(msg.button);
                }
            } break;
        case MOUSE_BUTTON_DOWN:
            {
                for(auto it = this->m_mouseListener.begin(); it!= this->m_mouseListener.end(); ++it)
                {
                    result |= (*it)->VOnMouseButtonDown(msg.mouseX, msg.mouseY, msg.button);
                }
            } break;
        case MOUSE_BUTTON_RELEASED:
            {
                for(auto it = this->m_mouseListener.begin(); it!= this->m_mouseListener.end(); ++it)
                {
                    result |= (*it)->VOnMouseButtonReleased(msg.mouseX, msg.mouseY, msg.button);
                }
            } break;

        case MOUSE_BUTTON_PRESSED:
            {
                for(auto it = this->m_mouseListener.begin(); it!= this->m_mouseListener.end(); ++it)
                {
                    result |= (*it)->VOnMouseButtonPressed(msg.mouseX, msg.mouseY, msg.button);
                }
            } break;
        case TBD_MOUSE_MOVED:
            {
                for(auto it = this->m_mouseListener.begin(); it!= this->m_mouseListener.end(); ++it)
                {
                    result |= (*it)->VOnMouseMoved(msg.mouseX, msg.mouseY, msg.mouseDX, msg.mouseDY);
                }
            } break;
        case MOUSE_DRAGGED:
            {
                for(auto it = this->m_mouseListener.begin(); it!= this->m_mouseListener.end(); ++it)
                {
                    result |= (*it)->VOnMouseDragged(msg.mouseX, msg.mouseY, msg.mouseDX, msg.mouseDY, msg.button);
                }
            } break;
        case MOUSE_WHEEL:
            {
                for(auto it = this->m_mouseListener.begin(); it!= this->m_mouseListener.end(); ++it)
                {
                    result |= (*it)->VOnMouseWheel(msg.mouseY, msg.mouseX, msg.mouseWheelDelta);
                }
            } break;
        }
        return result;
    } */

    VOID InputHandler::PushKeyListener(IKeyListener* listener)
    {
        m_keyStack.Push(listener);
    }

    VOID InputHandler::PopKeyListener(VOID)
    {
        m_keyStack.Pop();
    }

    VOID InputHandler::PushMouseListener(IMouseListener* listener)
    {
        m_mouseStack.Push(listener);
    }

    VOID InputHandler::PopMouseListener(VOID)
    {
        m_mouseStack.Pop();
    }

    BOOL InputHandler::UpdateListener(InputMessage msg) {
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

    VOID InputHandler::OnUpdate(VOID)
    {
        for(UINT i = 0; i < 0xFE; ++i)
        {
            if(m_isKeyDown[i])
            {
                InputMessage keymsg;
                keymsg.button = i;
                keymsg.message = KEY_DOWN;
                this->UpdateListener(keymsg);
            }
        }
    }

    /*
    BOOL InputHandler::Update(VOID)
    {
        InputMessage keymsg;
        for(UINT i = 0; i < 256; ++i)
        {
            keymsg.button = i;
            if(VIsKeyPressed(i))
            {
                keymsg.message = KEY_PRESSED;
                this->UpdateListener(keymsg);
            }
            else if(VIsKeyReleased(i))
            {
                keymsg.message = KEY_RELEASED;
                this->UpdateListener(keymsg);
            }
            else if(VIsKeyDown(i))
            {
                keymsg.message = KEY_DOWN;
                this->UpdateListener(keymsg);
            }
        }

        InputMessage mousnmsg;
        mousnmsg.mouseWheelDelta = VGetDMouseWheel();
        mousnmsg.mouseX = VGetPosX();
        mousnmsg.mouseY = VGetPosY();
        mousnmsg.mouseDX = VGetDX();
        mousnmsg.mouseDY = VGetDY();
        if(mousnmsg.mouseDX != 0 || mousnmsg.mouseDY != 0)
        {
            if(VIsMouseButtonDown(0))
            {
                mousnmsg.message = MOUSE_DRAGGED;
            }
            else
            {
                mousnmsg.message = TBD_MOUSE_MOVED;
            }
            this->UpdateListener(mousnmsg);
        }

        //if(mousnmsg.message != MOUSE_DRAGGED) //for now
        for(INT i = 0; i < 4; ++i)
        {
            mousnmsg.button = i;
            if(VIsMouseButtonDown(i))
            {
                mousnmsg.message = MOUSE_BUTTON_DOWN;
                this->UpdateListener(mousnmsg);
            }
            else if(VIsMouseButtonPressed(i))
            {
                mousnmsg.message = MOUSE_BUTTON_PRESSED;
                this->UpdateListener(mousnmsg);
            }
            else if(VIsMouseButtonReleased(i))
            {
                mousnmsg.message = MOUSE_BUTTON_RELEASED;
                this->UpdateListener(mousnmsg);
            }
            else if(mousnmsg.mouseWheelDelta != 0)
            {
                mousnmsg.message = MOUSE_WHEEL;
                this->UpdateListener(mousnmsg);
            }
        }

        return TRUE;
    }
    */

    InputHandler::~InputHandler(VOID) 
    {

    }

    //win input

    DefaultWinInputHandler::DefaultWinInputHandler(VOID) : m_hwnd(NULL)
    {
        VOnRestore();
    }

    BOOL DefaultWinInputHandler::VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height)
    {
        m_hwnd = hwnd;
        return TRUE;
    }

    BOOL DefaultWinInputHandler::VOnRestore(VOID)
    {
        m_isMouseGrabbed = FALSE;
        m_isEscapePressed = FALSE;
        m_lastMousePosX = 0; //todo
        m_lastMousePosY = 0; //todo
        VGrabMouse(FALSE);
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

    BOOL DefaultWinInputHandler::VIsEscPressed(VOID)
    {
        return m_isEscapePressed;
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
            app::g_pApp->GetInputHandler()->PushKeyListener(this);
            app::g_pApp->GetInputHandler()->PushMouseListener(this);
            m_active = TRUE;
        }
    }

    VOID InputAdapter::Deactivate(VOID)
    {
        if(m_active)
        {
            app::g_pApp->GetInputHandler()->PopKeyListener();
            app::g_pApp->GetInputHandler()->PopMouseListener();
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

    //----DirectInput
    /*
    DirectInput::DirectInput(VOID) 
        : m_directInput(NULL), m_keyBoard(NULL), m_mouse(NULL), 
        m_height(0), m_width(0), m_hinstance(0), m_hwnd(0), m_initialized(FALSE),
        m_mouseX(0), m_mouseY(0), m_mouseGrabbed(FALSE)
    {
        for(UINT i = 0; i < 256; ++i)
        {
            m_keys[0][i] = 0;
            m_keys[1][i] = 0;
        }

        ZeroMemory(&m_mouseState[0], sizeof(DIMOUSESTATE));
        ZeroMemory(&m_mouseState[1], sizeof(DIMOUSESTATE));

        m_currentLayer = 0;
    }

    BOOL DirectInput::VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height)
    {
        if(m_initialized) return TRUE;
        m_initialized = TRUE;
        m_hwnd = hwnd;
        m_hinstance = hinstance;
        return VOnRestore(width, height);
    }

    BOOL DirectInput::VOnRestore(UINT width, UINT height)
    {
        Delete();
        HRESULT result = DirectInput8Create(m_hinstance, DIRECTINPUT_VERSION, IID_IDirectInput8, (VOID**)&m_directInput, NULL);
        RETURN_IF_FAILED(result == DI_OK);

        result = m_directInput->CreateDevice(GUID_SysKeyboard, (LPDIRECTINPUTDEVICE8W*)&m_keyBoard, NULL);
        RETURN_IF_FAILED(result == DI_OK);

        result = m_directInput->CreateDevice(GUID_SysMouse, (LPDIRECTINPUTDEVICE8W*)&m_mouse, NULL);
        RETURN_IF_FAILED(result == DI_OK);

        RETURN_IF_FAILED(DI_OK == m_keyBoard->SetDataFormat(&c_dfDIKeyboard));
        RETURN_IF_FAILED(DI_OK == m_keyBoard->SetCooperativeLevel(m_hwnd, DISCL_FOREGROUND | DISCL_NONEXCLUSIVE));//DISCL_NONEXCLUSIVE));//DISCL_EXCLUSIVE));
        RETURN_IF_FAILED(DI_OK == m_keyBoard->Acquire());

        RETURN_IF_FAILED(DI_OK == m_mouse->SetDataFormat(&c_dfDIMouse));
        RETURN_IF_FAILED(DI_OK == m_mouse->SetCooperativeLevel(m_hwnd, DISCL_FOREGROUND | DISCL_EXCLUSIVE));//DISCL_EXCLUSIVE));
        RETURN_IF_FAILED(DI_OK == m_mouse->Acquire());

        m_height = height;
        m_width = width;

        return TRUE;
    }

    BOOL DirectInput::VUpdateState(VOID)
    {
        RETURN_IF_FAILED(ReadKeyboardState());
        RETURN_IF_FAILED(ReadMouseState());

        m_mouseX += m_mouseState[m_currentLayer].lX;
        m_mouseY += m_mouseState[m_currentLayer].lY;

        if(m_mouseX < 0)  { m_mouseX = 0; }
        if(m_mouseY < 0)  { m_mouseY = 0; }

        if(m_mouseX > m_width)  { m_mouseX = m_width; }
        if(m_mouseY > m_height) { m_mouseY = m_height; }

        m_currentLayer = (m_currentLayer + 1) % 2;

        return TRUE;
    }

    BOOL DirectInput::VIsEscPressed(VOID)
    {
        return VIsKeyDown(DIK_ESCAPE);
    }

    BOOL DirectInput::VIsKeyDown(UCHAR key)
    {
        return m_keys[m_currentLayer][key] & 0x80;
    }

    BOOL DirectInput::VIsKeyPressed(UCHAR key)
    {
        return (!(m_keys[m_currentLayer][key] & 0x80)) && (m_keys[(m_currentLayer + 1) % 2][key] & 0x80);
    }

    BOOL DirectInput::VIsKeyReleased(UCHAR key)
    {
        return (m_keys[m_currentLayer][key] & 0x80) && (!(m_keys[(m_currentLayer + 1) % 2][key] & 0x80));
    }

    BOOL DirectInput::VIsMouseButtonPressed(INT button)
    {
        return (!(m_mouseState[m_currentLayer].rgbButtons[button] & 0x80)) && (m_mouseState[(m_currentLayer + 1) % 2].rgbButtons[button] & 0x80);
    }

    BOOL DirectInput::VIsMouseButtonReleased(INT button)
    {
        return (m_mouseState[m_currentLayer].rgbButtons[button] & 0x80) && (!(m_mouseState[(m_currentLayer + 1) % 2].rgbButtons[button] & 0x80));
    }

    BOOL DirectInput::VIsMouseButtonDown(INT button)
    {
        return m_mouseState[m_currentLayer].rgbButtons[button] & 0x80;
    }

    INT DirectInput::VGetDMouseWheel(VOID)
    {
        return m_mouseState[m_currentLayer].lZ;
    }

    INT DirectInput::VGetPosX(VOID)
    {
        return m_mouseX;
    }

    INT DirectInput::VGetPosY(VOID)
    {
        return m_mouseY;
    }

    INT DirectInput::VGetDX(VOID)
    {
        return m_mouseState[m_currentLayer].lX;
    }

    INT DirectInput::VGetDY(VOID)
    {
        return m_mouseState[m_currentLayer].lY;
    }

    BOOL DirectInput::VIsMouseGrabbed(VOID)
    {
        return m_mouseGrabbed;
    }

    BOOL DirectInput::VGrabMouse(BOOL grab)
    {
        //DWORD flags = 0;
        if(grab)
        {
            //flags = DISCL_EXCLUSIVE | DISCL_FOREGROUND;
            RETURN_IF_FAILED(DI_OK == m_mouse->Unacquire());
            RETURN_IF_FAILED(DI_OK == m_mouse->SetCooperativeLevel(m_hwnd, DISCL_FOREGROUND | DISCL_EXCLUSIVE));//DISCL_NONEXCLUSIVE));//DISCL_EXCLUSIVE));
            RETURN_IF_FAILED(DI_OK == m_mouse->Acquire());

            while(ShowCursor(FALSE) >= 0);
            GetCursorPos(&m_mousePosition);
        }
        else
        {
            //flags = DISCL_NONEXCLUSIVE | DISCL_FOREGROUND;
            RETURN_IF_FAILED(DI_OK == m_mouse->Unacquire());
            RETURN_IF_FAILED(DI_OK == m_mouse->SetCooperativeLevel(m_hwnd, DISCL_FOREGROUND | DISCL_NONEXCLUSIVE));//DISCL_NONEXCLUSIVE));//DISCL_EXCLUSIVE));
            SetCursor(LoadCursor(NULL, IDC_ARROW));
            while(ShowCursor(TRUE) <= 0);
            RETURN_IF_FAILED(DI_OK == m_mouse->Acquire());
            if(m_mouseGrabbed)
            {
                SetCursorPos(m_mousePosition.x, m_mousePosition.y);
            }
        }
        m_mouseGrabbed = grab;
        return TRUE;
    }

    BOOL DirectInput::ReadKeyboardState(VOID)
    {
        HRESULT result = m_keyBoard->GetDeviceState(256, &m_keys[m_currentLayer]);

        if(FAILED(result))
        {
            if((result == DIERR_INPUTLOST) || (result == DIERR_NOTACQUIRED))
            {
                m_keyBoard->Acquire();
            }
            else
            {
                return FALSE;
            }
        }

        return TRUE;
    }

    BOOL DirectInput::ReadMouseState(VOID)
    {
        HRESULT result = m_mouse->GetDeviceState(sizeof(DIMOUSESTATE), &m_mouseState[m_currentLayer]);

        if(FAILED(result))
        {
            if((result == DIERR_INPUTLOST) || (result == DIERR_NOTACQUIRED))
            {
                m_mouse->Acquire();
            }
            else
            {
                return FALSE;
            }
        }

        return TRUE;
    }

    VOID DirectInput::Delete(VOID)
    {
        if(m_mouse)
        {
            m_mouse->Unacquire();
        }
    
        if(m_keyBoard)
        {
            m_keyBoard->Unacquire();
        }
    
        SAFE_RELEASE(m_mouse);
        SAFE_RELEASE(m_keyBoard);
        SAFE_RELEASE(m_directInput);
    }

    DirectInput::~DirectInput(VOID)
    {
        Delete();
    } */
}