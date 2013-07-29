#pragma once
#include "stdafx.h"
#include "Actor.h"
#include "RenderTarget.h"
#include "RenderPath.h"
#include "ShaderProgram.h"
#include "Texture.h"
#include "Mat4.h"

namespace tbd
{
    class IPicker
    {
    public:
        IPicker(VOID) {}

        virtual BOOL VCreate(VOID) = 0;

        virtual VOID VPostRender(VOID) = 0;

        virtual VOID VRender(VOID) = 0;

        virtual BOOL VHasPicked(VOID) CONST  = 0;

        virtual ActorId VPick(VOID) CONST = 0;

        virtual ~IPicker(VOID) {}
    };

    class ActorPicker : public IPicker
    {
    private:
        ActorId m_currentActor;
        d3d::RenderTarget* m_renderTarget;
        d3d::ShaderProgram* m_shaderProgram;
        d3d::Texture2D* m_texture;
        BOOL m_created;
        util::Mat4 m_projection;
    public:
        ActorPicker(VOID) : m_currentActor(INVALID_ACTOR_ID), m_created(FALSE)
        {
        }
        
        BOOL VCreate(VOID);
        
        VOID VPostRender(VOID);

        VOID VRender(VOID);
        
        BOOL VHasPicked(VOID) CONST;
        
        ActorId VPick(VOID) CONST;
        
        ~ActorPicker(VOID);
    };
}
