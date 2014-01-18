#include "FPSControler.h"
#include "Vec3.h"
namespace util {

FPSControler::FPSControler(std::shared_ptr<ICamera> camera) : p_camera(camera), m_lastPosX(-1), m_lastPosY(-1), m_minSpeed(1), m_maxSpeed(3) {
    for(uint i = 0; i < 0xFE; ++i) 
    {
        m_isKeyDown[i] = NULL;
    }
}

void FPSControler::Update(uint millis) {
    util::Vec3 move;
    
    float factor = 1e-3f * millis;
    float speed = 1;
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

bool FPSControler::VOnKeyDown(uint const code) {
    m_isKeyDown[code] = true;
    return true;
}

bool FPSControler::VOnKeyPressed(uint const code) {
    return true;
}

bool FPSControler::VOnKeyReleased(uint const code) {
    m_isKeyDown[code] = false;
    return true;
}

bool FPSControler::VOnMouseButtonDown(int x, int y, int button) { 
    this->m_lastPosX = x;
    this->m_lastPosY = y;
    return true; 
};

bool FPSControler::VOnMouseButtonReleased(int x, int y, int button) { return false; };
bool FPSControler::VOnMousePressed(int x, int y, int button) { return false; };
bool FPSControler::VOnMouseMoved(int x, int y) { return false; };

bool FPSControler::VOnMouseDragged(int x, int y, int button) {
    if(button & 1) {
        int dx = x - m_lastPosX;
        int dy = y - m_lastPosY;
        p_camera->Rotate(2 * dx * (float)(1e-3), 2 * dy * (float)(1e-3));
        this->m_lastPosX = x;
        this->m_lastPosY = y;
        this->VOnUpdate();
    }
    return true; 
};
bool FPSControler::VOnMouseWheel(int x, int y, int delta) { return false; };


FPSControler::~FPSControler(void) {
}
}