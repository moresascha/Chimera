#include "Sound.h"
#include <dsound.h>
#include "GameApp.h"
#include "Resources.h"

#ifdef _DEBUG
    #pragma comment(lib, "dsound.lib")
    #pragma comment(lib, "dxguid.lib")
#else
    #pragma comment(lib, "dsound.lib")
    #pragma comment(lib, "dxguid.lib")
#endif

namespace chimera
{

    void BaseSoundSystem::VPauseAll(void)
    {
        TBD_FOR(m_currentSoundBuffer)
        {
            (*it)->VPause();
        }
    }

    void BaseSoundSystem::VResumeAll(void)
    {
        TBD_FOR(m_currentSoundBuffer)
        {
            (*it)->VResume();
        }
    }

    void BaseSoundSystem::VStopAll(void)
    {
        TBD_FOR(m_currentSoundBuffer)
        {
            (*it)->VStop();
        }
    }

    BaseSoundSystem::~BaseSoundSystem(void)
    {
        TBD_FOR(m_currentSoundBuffer)
        {
            VReleaseSoundBuffer(*it);
        }
    }

    DirectSoundSystem::DirectSoundSystem(void) : m_pDirectSound(NULL), m_initialized(false)
    {

    }

    bool DirectSoundSystem::VInit(void)
    {
        if(m_initialized)
        {
            return true;
        }
        m_initialized = true;
        //d3d::g_hWnd
        HRESULT hr = DirectSoundCreate8(NULL, &m_pDirectSound, NULL);

        RETURN_IF_FAILED(hr == S_OK);

        hr = m_pDirectSound->SetCooperativeLevel(chimera::g_pApp->GetWindowHandle(), DSSCL_PRIORITY);

        RETURN_IF_FAILED(hr == S_OK);

        DSBUFFERDESC desc;
        desc.dwSize = sizeof(DSBUFFERDESC);
        desc.dwBufferBytes = 0;
        desc.dwFlags = DSBCAPS_PRIMARYBUFFER;
        desc.dwReserved = 0;
        desc.guid3DAlgorithm = GUID_NULL;
        desc.lpwfxFormat = NULL;

        hr = m_pDirectSound->CreateSoundBuffer(&desc, &m_pPrimaryBuffer, NULL);

        RETURN_IF_FAILED(hr == S_OK);

        WAVEFORMATEX format;
        ZeroMemory(&format, sizeof(format));
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nSamplesPerSec = 44100;
        format.wBitsPerSample = 16;
        format.nChannels = 2;
        format.nBlockAlign = format.wBitsPerSample / 8 * format.nChannels;
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
        format.cbSize = 0;

        hr = m_pPrimaryBuffer->SetFormat(&format);
        
        RETURN_IF_FAILED(hr == S_OK);

        SAFE_RELEASE(m_pPrimaryBuffer);

        return true;
    }

    void DirectSoundSystem::VReleaseSoundBuffer(ISoundBuffer* buffer)
    {
        buffer->VStop();
        m_currentSoundBuffer.remove(buffer);
        SAFE_DELETE(buffer);
    }

    ISoundBuffer* DirectSoundSystem::VCreateSoundBuffer(std::shared_ptr<chimera::ResHandle> handle)
    {
        ISoundBuffer* sound = new DirectWaveBuffer(this, handle);
        if(!sound->VInit())
        {
            LOG_CRITICAL_ERROR("failed to init soundbuffer");
        }
        m_currentSoundBuffer.push_back(sound);
        return sound;
    }

    DirectSoundSystem::~DirectSoundSystem(void)
    {
        if(m_initialized)
        {
            SAFE_RELEASE(m_pDirectSound);
        }
    }

    DirectWaveBuffer::DirectWaveBuffer(DirectSoundSystem* system, std::shared_ptr<chimera::ResHandle> handle) 
        : m_pSystem(system), m_pHandle(handle), m_pSoundBuffer(NULL), m_initialized(false)
    {

    }

    bool DirectWaveBuffer::VInit(void)
    {
        if(m_initialized)
        {
            return true;
        }
        m_initialized = true;

        DSBUFFERDESC desc;

        ZeroMemory(&desc, sizeof(DSBUFFERDESC)); 
        desc.dwSize = sizeof(DSBUFFERDESC); 
        desc.dwFlags = 
            DSBCAPS_CTRLPAN | DSBCAPS_CTRLVOLUME | DSBCAPS_CTRLFREQUENCY
            | DSBCAPS_GLOBALFOCUS; 
        desc.dwBufferBytes = m_pHandle->Size();
        desc.lpwfxFormat = &std::static_pointer_cast<chimera::WaveSoundExtraDatra>(m_pHandle->GetExtraData())->m_format;

        LPDIRECTSOUNDBUFFER pBuffer;
        HR_RETURN_IF_FAILED(m_pSystem->m_pDirectSound->CreateSoundBuffer(&desc, &pBuffer, NULL));

        HR_RETURN_IF_FAILED(pBuffer->QueryInterface(IID_IDirectSoundBuffer8, (LPVOID*) &m_pSoundBuffer));
        SAFE_RELEASE(pBuffer);

        return FillBuffer();
    }

    bool DirectWaveBuffer::FillBuffer(void)
    {
        UCHAR* bufferPtr;
        ulong size;
        char* wavData = m_pHandle->Buffer();
        int bufferSize = m_pHandle->Size();

        HR_RETURN_IF_FAILED(m_pSoundBuffer->Lock(0, bufferSize, (void**)&bufferPtr, (DWORD*)&size, NULL, 0, 0));

        memcpy(bufferPtr, wavData, bufferSize);

        HR_RETURN_IF_FAILED(m_pSoundBuffer->Unlock((void*)bufferPtr, m_pHandle->Size(), NULL, 0));

        HR_RETURN_IF_FAILED(m_pSoundBuffer->SetCurrentPosition(0));

        VSetVolume(DSBVOLUME_MAX);

        return true;
    }

    void DirectWaveBuffer::VSetVolume(LONG volume)
    {
        assert(volume >= 0 && volume <= 100);

        float norm = volume / 100.0f;
        float logProp = norm > 0.1f ? (1 + log10(norm)) : 0;
        float range = DSBVOLUME_MAX - DSBVOLUME_MIN;
        LONG vol = (LONG)((range * logProp) + DSBVOLUME_MIN);

        if(FAILED((m_pSoundBuffer->SetVolume(vol))))
        {
            LOG_CRITICAL_ERROR("failed to set volume");
        }
    }

    int DirectWaveBuffer::VGetVolume(void)
    {
        LONG v;
        m_pSoundBuffer->GetVolume(&v);
        return (int)v;
    }

    float DirectWaveBuffer::VGetProgress(void)
    {
        DWORD pos;
        m_pSoundBuffer->GetCurrentPosition(&pos, NULL);
        return (float)pos / (float)m_pHandle->Size();
    }

    bool DirectWaveBuffer::VIsPlaying(void)
    {
        DWORD status;
        m_pSoundBuffer->GetStatus(&status);
        return status & DSBSTATUS_PLAYING;
    }

    bool DirectWaveBuffer::VIsLooping(void)
    {
        DWORD status;
        m_pSoundBuffer->GetStatus(&status);
        return status & DSBSTATUS_LOOPING;
    }

    bool DirectWaveBuffer::VRestore(void)
    {
        DWORD status;
        m_pSoundBuffer->GetStatus(&status);

        if(status & DSBSTATUS_BUFFERLOST)
        {
            int count = 0;
            HRESULT hr = m_pSoundBuffer->Restore();
            if(FAILED(hr))
            {
                Sleep(10);
            }

            while((hr = m_pSoundBuffer->Restore() == DSERR_BUFFERLOST) && +count < 100);

            if(FAILED(hr))
            {
                return false;
            }
        }
        return FillBuffer();
    }

    void DirectWaveBuffer::VResume(void)
    {
        VPlay(VIsLooping());
    }

    void DirectWaveBuffer::VPause(void)
    {
        LOG_CRITICAL_ERROR("not implemented");
    }

    void DirectWaveBuffer::VSetPan(float pan)
    {
        LONG p = (LONG)((pan * 0.5 + 0.5) * (DSBPAN_RIGHT - DSBPAN_LEFT) + DSBPAN_LEFT);
        m_pSoundBuffer->SetPan(p);
    }

    void DirectWaveBuffer::VPlay(bool loop)
    {
        VStop();
        if(FAILED(m_pSoundBuffer->Play(0, 0, loop ? DSBPLAY_LOOPING : 0L)))
        {
            LOG_CRITICAL_ERROR("error playing sound");
        }
    }

    void DirectWaveBuffer::VStop(void)
    {
        if(FAILED(m_pSoundBuffer->Stop()))
        {
            LOG_CRITICAL_ERROR("error stopping sound");
        }
    }

    DirectWaveBuffer::~DirectWaveBuffer(void)
    {
        SAFE_RELEASE(m_pSoundBuffer);
    }
}