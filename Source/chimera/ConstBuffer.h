#pragma once
#include "d3d.h"
#include "Mat4.h"
namespace d3d {

#define ACTIVATE_FS 1
#define ACTIVATE_VS 2
#define ACTIVATE_GS 4
#define ACTIVATE_TS 8
#define ACTIVATE_ALL (ACTIVATE_FS|ACTIVATE_VS|ACTIVATE_GS|ACTIVATE_TS)
class ConstBuffer
{
private:
    ID3D11Buffer* m_buffer;
     D3D11_MAPPED_SUBRESOURCE m_ressource;
    UINT m_byteSize;
    VOID* m_sharedData;
public:
    ConstBuffer(VOID);
    VOID Init(UINT byteSize, VOID* sharedData = 0);
     VOID* Map(VOID);
     VOID Unmap(VOID);
    VOID Update(VOID);
    VOID SetFromMatrix(CONST util::Mat4& value);
    VOID SetFromFloat4x4(CONST XMFLOAT4X4& value);
    VOID Activate(UINT slot, UINT shader = ACTIVATE_ALL);
     ID3D11Buffer* GetBuffer(VOID);
    ~ConstBuffer(VOID);
};
}