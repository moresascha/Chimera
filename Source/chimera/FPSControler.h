#pragma once
#include "stdafx.h"
#include "Input.h"
#include "Camera.h"
namespace util 
{
    struct MatrixBuffer 
    {
        XMFLOAT4X4 view;
        XMFLOAT4X4 proj;
    };

    class FPSControler : public chimera::IKeyListener, public chimera::IMouseListener 
    {
    private:
        INT m_lastPosX;
        INT m_lastPosY;
        FLOAT m_minSpeed;
        FLOAT m_maxSpeed;
        BOOL m_isKeyDown[0xFE];

    protected:
        MatrixBuffer* m_pBuffer;
        std::shared_ptr<ICamera> p_camera;

    public:

        FPSControler(std::shared_ptr<ICamera> camera);

        virtual VOID VOnUpdate() = 0;

        VOID Update(UINT millis);

        virtual BOOL VOnKeyDown(UINT CONST code);
        virtual BOOL VOnKeyPressed(UINT CONST code);
        virtual BOOL VOnKeyReleased(UINT CONST code);

        virtual BOOL VOnMouseButtonDown(INT x, INT y, INT button);
        virtual BOOL VOnMouseButtonReleased(INT x, INT y, INT button);
        virtual BOOL VOnMousePressed(INT x, INT y, INT button);
        virtual BOOL VOnMouseMoved(INT x, INT y);
        virtual BOOL VOnMouseDragged(INT x, INT y, INT button);
        virtual BOOL VOnMouseWheel(INT x, INT y, INT delta);

         ~FPSControler(VOID);
    };
}
