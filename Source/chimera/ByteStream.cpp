#include "ByteStream.h"

namespace tbd {

    HRESULT ByteStream::Seek(
        LARGE_INTEGER dlibMove,
        DWORD dwOrigin,
        ULARGE_INTEGER *plibNewPosition
        )
    {
        LONG start = 0;
        LONG newPos;
        switch(dwOrigin)
        {
        case STREAM_SEEK_SET:
            start = 0;
            break;
        case STREAM_SEEK_CUR:
            start = m_currentPos;
            break;
        case STREAM_SEEK_END:
            start = m_size;
            break;
        default:
            return STG_E_INVALIDFUNCTION;
            break;
        }

        newPos = start + (LONG)dlibMove.QuadPart;

        if(newPos < 0 || newPos > (LONG)m_size)
        {
            return STG_E_SEEKERROR;
        }

        m_currentPos = newPos;

        if(plibNewPosition)
        {
            plibNewPosition->QuadPart = m_currentPos;
        }
        return S_OK;
    }

    HRESULT ByteStream::Stat(
        STATSTG *pstatstg,
        DWORD grfStatFlag
        )
    {
        if (pstatstg == NULL)
        {
            return STG_E_INVALIDFUNCTION;
        }

        ZeroMemory(pstatstg, sizeof(STATSTG));
        pstatstg->cbSize.QuadPart = m_size;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE ByteStream::Read( 
        __out_bcount_part(cb, *pcbRead)  void *pv,
        ULONG cb,
        __out_opt  ULONG *pcbRead)
    {
        size_t          bytes_left;
        size_t          bytes_out;

        if (pcbRead != NULL) *pcbRead = 0;
        if (m_currentPos == m_size) // EOF
        {
            return HRESULT_FROM_WIN32(ERROR_END_OF_MEDIA);
        }

        bytes_left = m_size - m_currentPos;
        bytes_out = min(cb, bytes_left);
        memcpy(pv, &m_bytes[m_currentPos], bytes_out);
        m_currentPos += (LONG)bytes_out;
        if (pcbRead != NULL) *pcbRead = (ULONG)bytes_out;
        return S_OK;
    }
};
