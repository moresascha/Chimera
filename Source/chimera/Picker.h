#pragma once
#include "stdafx.h"
#include "Mat4.h"

namespace chimera
{
    class ActorPicker : public IPicker
    {
    private:
        ActorId m_currentActor;
        IRenderTarget* m_pRenderTarget;
        IShaderProgram* m_pShaderProgram;
        IDeviceTexture* m_pTexture;
        bool m_created;
        chimera::util::Mat4 m_projection;

        bool Create(VOID);

        IRenderTarget* GetTarget(void);

        void Render(void);

        void PostRender(void);

    public:
        ActorPicker(void);

        bool VHasPicked(void) const;
        
        ActorId VPick(void);

        bool VOnRestore(void);

        void PickActorDelegate(IEventPtr ptr);

        void ActorDeletedDelegate(IEventPtr ptr);
        
        ~ActorPicker(void);
    };
}
