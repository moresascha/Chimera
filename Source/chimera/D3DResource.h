#pragma once
#include "stdafx.h"
#include "d3d.h"

namespace d3d
{
    class D3DResource
    {
    private:
        UINT m_bindFlags;
        D3D11_RESOURCE_MISC_FLAG  m_miscFlags;
        D3D11_USAGE m_usage;
        D3D11_CPU_ACCESS_FLAG m_cpuAccess;

    public:
        D3DResource(VOID) : m_bindFlags((D3D11_BIND_FLAG)0), m_miscFlags((D3D11_RESOURCE_MISC_FLAG)0), m_usage(D3D11_USAGE_DEFAULT), m_cpuAccess((D3D11_CPU_ACCESS_FLAG)0) {}

        virtual UINT GetBindflags(VOID)
        {
            return m_bindFlags;
        }

        virtual D3D11_USAGE GetUsage(VOID)
        {
            return m_usage;
        }

        virtual D3D11_CPU_ACCESS_FLAG GetCPUAccess(VOID)
        {
            return m_cpuAccess;
        }
        
        virtual VOID SetBindflags(UINT flags)
        {
            m_bindFlags = flags;
        }

        virtual VOID SetUsage(D3D11_USAGE usage)
        {
            m_usage = usage;
        }

        virtual VOID SetCPUAccess(D3D11_CPU_ACCESS_FLAG cpua)
        {
            m_cpuAccess = cpua;
        }

        VOID SetMiscflags(D3D11_RESOURCE_MISC_FLAG flags)
        {
            m_miscFlags = flags;
        }

        D3D11_RESOURCE_MISC_FLAG GetMiscflags(VOID)
        {
            return m_miscFlags;
        }

        virtual ~D3DResource(VOID) {}
    };
};



