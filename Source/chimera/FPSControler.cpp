#include "FPSControler.h"
#include "Vec3.h"
namespace util {

FPSControler::FPSControler(std::shared_ptr<ICamera> camera) : p_camera(camera), m_lastPosX(-1), m_lastPosY(-1), m_minSpeed(1), m_maxSpeed(3) {
    for(UINT i = 0; i < 0xFE; ++i) 
    {
        m_isKeyDown[i] = NULL;
    }
}

VOID FPSControler::Update(UINT millis) {
    util::Vec3 move;
    
    FLOAT factor = 1e-3f * millis;
    FLOAT speed = 1;
    if(m_isKeyDown[KEY_W]) 
    {
        move.z = factor;
    }
    if(m_isKeyDown[KEY_S]) 
    {
          move.z = -factor;
    }
    if(m_isKeyDown[KEY_A]) 
    {
          move.x = -factor;
    }
    if(m_isKeyDown[KEY_D]) 
    {
          move.x = factor;
    }
    if(m_isKeyDown[VK_SPACE]) 
    {
        move.y = factor;
    }
    if(m_isKeyDown[KEY_C]) 
    {
        move.y = -factor;
    }
    if(move.x != 0 || move.y != 0 || move.z != 0)
    {
        speed = m_isKeyDown[KEY_LSHIFT] ? m_maxSpeed : m_minSpeed;
        move.Scale(speed);
         p_camera->Move(move);
        this->VOnUpdate();
    }
}

BOOL FPSControler::VOnKeyDown(UINT CONST code) {
    m_isKeyDown[code] = TRUE;
    return TRUE;
}

BOOL FPSControler::VOnKeyPressed(UINT CONST code) {
    return TRUE;
}

BOOL FPSControler::VOnKeyReleased(UINT CONST code) {
    m_isKeyDown[code] = FALSE;
    return TRUE;
}

BOOL FPSControler::VOnMouseButtonDown(INT x, INT y, INT button) { 
    this->m_lastPosX = x;
    this->m_lastPosY = y;
    return TRUE; 
};

BOOL FPSControler::VOnMouseButtonReleased(INT x, INT y, INT button) { return FALSE; };
BOOL FPSControler::VOnMousePressed(INT x, INT y, INT button) { return FALSE; };
BOOL FPSControler::VOnMouseMoved(INT x, INT y) { return FALSE; };

BOOL FPSControler::VOnMouseDragged(INT x, INT y, INT button) {
    if(button & 1) {
        INT dx = x - m_lastPosX;
        INT dy = y - m_lastPosY;
        p_camera->Rotate(2 * dx * (FLOAT)(1e-3), 2 * dy * (FLOAT)(1e-3));
        this->m_lastPosX = x;
        this->m_lastPosY = y;
        this->VOnUpdate();
    }
    return TRUE; 
};
BOOL FPSControler::VOnMouseWheel(INT x, INT y, INT delta) { return FALSE; };


FPSControler::~FPSControler(VOID) {
}
}