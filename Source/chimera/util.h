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

        uint getText(const char* pFileName, char** source);

        void InitGdiplus(void);

        Gdiplus::Bitmap* GetBitmapFromFile(const WCHAR* file);

        Gdiplus::Bitmap* GetBitmapFromBytes(char* bytes, uint size);

        char* GetTextureData(const WCHAR* file);

        char* GetTextureData(Gdiplus::Bitmap* map);

        void DestroyGdiplus(void);

        std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

        std::vector<std::string> split(const std::string &s, char delim);

        std::wstring string2wstring(const std::string& s);

        bool CheckIfFileExists(LPCSTR file);

        class cmRNG
        {
        private:
            int m_seed;
            std::default_random_engine m_generator;
            std::uniform_int_distribution<uint> m_distribution;
        public:
            cmRNG(int seed = 1) : m_seed(seed)
            {
                m_generator.seed(m_seed);
            }

            float NextFloat(float scale)
            {
                return scale * m_distribution(m_generator) / (float)UINT_MAX;
            }

            float NextFloat(void)
            {
                return NextFloat(1);
            }

            float NextCubeFloat(float scale)
            {
                return -scale + 2 * NextFloat(scale);
            }

            float NextCubeFloat(void)
            {
                return NextCubeFloat(1);
            }

            uint NextInt(void)
            {
                return (uint)m_distribution(m_generator);
            }
        };
    };
}

