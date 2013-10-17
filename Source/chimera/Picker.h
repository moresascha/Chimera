#pragma once
#include "stdafx.h"
#include "Actor.h"
#include "RenderTarget.h"
#include "ShaderProgram.h"
#include "Texture.h"
#include "Mat4.h"

namespace chimera
{
    class ActorPicker : public IPicker
    {
    private:
        ActorId m_currentActor;
        chimera::RenderTarget* m_renderTarget;
        chimera::ShaderProgram* m_shaderProgram;
        chimera::D3DTexture2D* m_texture;
        BOOL m_created;
        util::Mat4 m_projection;
    public:
        ActorPicker(VOID) : m_currentActor(INVALID_ACTOR_ID), m_created(FALSE)
        {
        }
        
        BOOL VCreate(VOID);

        chimera::RenderTarget* GetTarget(VOID);
        
        VOID VPostRender(VOID);

        VOID VRender(VOID);
        
        BOOL VHasPicked(VOID) CONST;
        
        ActorId VPick(VOID) CONST;
        
        ~ActorPicker(VOID);
    };
}
