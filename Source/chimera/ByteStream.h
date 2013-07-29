#pragma once
#include "stdafx.h"
namespace tbd {
class ByteStream : public IStream
{
    CHAR* m_bytes;
    UINT m_size;
    LONG m_currentPos;
public:
    ByteStream(CHAR* bytes, UINT size) : m_bytes(bytes), m_size(size),  m_currentPos(0) {}

    HRESULT ByteStream::Seek(
        LARGE_INTEGER dlibMove,
        DWORD dwOrigin,
        ULARGE_INTEGER *plibNewPosition
        );

    HRESULT ByteStream::Stat(
        STATSTG *pstatstg,
        DWORD grfStatFlag
        );

    /* [local] */ HRESULT STDMETHODCALLTYPE Read( 
        /* [annotation] */ 
        __out_bcount_part(cb, *pcbRead)  void *pv,
        /* [in] */ ULONG cb,
        /* [annotation] */ 
        __out_opt  ULONG *pcbRead);

    /* [local] */ HRESULT STDMETHODCALLTYPE Write( 
        /* [annotation] */ 
        __in_bcount(cb)  const void *pv,
        /* [in] */ ULONG cb,
        /* [annotation] */ 
        __out_opt  ULONG *pcbWritten) { return S_OK; }

    HRESULT STDMETHODCALLTYPE SetSize( 
        /* [in] */ ULARGE_INTEGER libNewSize) { return S_OK; }

    /* [local] */ HRESULT STDMETHODCALLTYPE CopyTo( 
        /* [unique][in] */ IStream *pstm,
        /* [in] */ ULARGE_INTEGER cb,
        /* [annotation] */ 
        __out_opt  ULARGE_INTEGER *pcbRead,
        /* [annotation] */ 
        __out_opt  ULARGE_INTEGER *pcbWritten) { return S_OK; }

    HRESULT STDMETHODCALLTYPE Commit( 
        /* [in] */ DWORD grfCommitFlags) { return S_OK; }

    HRESULT STDMETHODCALLTYPE Revert( void) { return S_OK; }

    HRESULT STDMETHODCALLTYPE LockRegion( 
        /* [in] */ ULARGE_INTEGER libOffset,
        /* [in] */ ULARGE_INTEGER cb,
        /* [in] */ DWORD dwLockType) { return S_OK; }

    HRESULT STDMETHODCALLTYPE UnlockRegion( 
        /* [in] */ ULARGE_INTEGER libOffset,
        /* [in] */ ULARGE_INTEGER cb,
        /* [in] */ DWORD dwLockType) { return S_OK; }

    HRESULT STDMETHODCALLTYPE Clone( 
        /* [out] */ __RPC__deref_out_opt IStream **ppstm) { return S_OK; }

    HRESULT STDMETHODCALLTYPE QueryInterface( 
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ __RPC__deref_out void __RPC_FAR *__RPC_FAR *ppvObject) { return S_OK; }

    ULONG STDMETHODCALLTYPE AddRef(void)  { return S_OK; }

    ULONG STDMETHODCALLTYPE Release(void) { return S_OK; }

    ~ByteStream(VOID) {}
};
};
