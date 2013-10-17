#pragma once
#include "CMTypes.h"

namespace chimera
{
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
}