#pragma once
#include "stdafx.h"
#include <vector>
#include <sstream>
#include <random>

namespace chimera
{
    namespace util
    {

        static ULONG_PTR gdiplusToken;

        UINT getText(CONST CHAR* pFileName, CHAR** source);

        VOID InitGdiplus(VOID);

        Gdiplus::Bitmap* GetBitmapFromFile(CONST WCHAR* file);

        Gdiplus::Bitmap* GetBitmapFromBytes(CHAR* bytes, UINT size);

        CHAR* GetTextureData(CONST WCHAR* file);

        CHAR* GetTextureData(Gdiplus::Bitmap* map);

        VOID DestroyGdiplus(VOID);

        std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

        std::vector<std::string> split(const std::string &s, char delim);

        std::wstring string2wstring(const std::string& s);

        BOOL CheckIfFileExists(LPCSTR file);

        class cmRNG
        {
        private:
            INT m_seed;
            std::default_random_engine m_generator;
            std::uniform_int_distribution<UINT> m_distribution;
        public:
            cmRNG(INT seed = 1) : m_seed(seed)
            {
                m_generator.seed(m_seed);
            }

            FLOAT NextFloat(VOID)
            {
                return m_distribution(m_generator) / (FLOAT)UINT_MAX;
            }

            UINT NextInt(VOID)
            {
                return (UINT)m_distribution(m_generator);
            }
        };
    };
}

