#pragma once
#include "stdafx.h"
#include "Event.h"

namespace tbd
{
    struct MaterialPair
    {
        std::string m0;
        std::string m1;

        BOOL operator==(CONST MaterialPair& pair)
        {
            return m0 == pair.m0 && m1 == pair.m1;
        }

        BOOL operator!=(CONST MaterialPair& pair)
        {
            return m0 != pair.m0 || m1 != pair.m1;
        }

        friend bool operator<(const MaterialPair& pair0, const MaterialPair& pair1)
        {
            return pair0.m0 < pair1.m0;
        }
    };

    class SoundEngine
    {
        std::map<MaterialPair, std::string> m_soundLibrary;
    public:
        SoundEngine(VOID);
        VOID ResgisterSound(std::string material0, std::string material1, std::string soundFile);
        VOID CollisionEventDelegate(event::IEventPtr event);
        VOID NewComponentDelegate(event::IEventPtr event);
        ~SoundEngine(VOID);
    };
}
