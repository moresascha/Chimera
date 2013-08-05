#pragma once
#include "stdafx.h"
#include <vector>
#include <sstream>

#pragma comment (lib, "Gdiplus.lib")

namespace util
{

    static ULONG_PTR gdiplusToken;

    DLL_EXPORT UINT getText(CONST CHAR* pFileName, CHAR** source);

    DLL_EXPORT VOID InitGdiplus(VOID);

    DLL_EXPORT Gdiplus::Bitmap* GetBitmapFromFile(CONST WCHAR* file);

    DLL_EXPORT Gdiplus::Bitmap* GetBitmapFromBytes(CHAR* bytes, UINT size);

    DLL_EXPORT CHAR* GetTextureData(CONST WCHAR* file);

    DLL_EXPORT CHAR* GetTextureData(Gdiplus::Bitmap* map);

    DLL_EXPORT VOID DestroyGdiplus(VOID);

    std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

    std::vector<std::string> split(const std::string &s, char delim);

    std::wstring string2wstring(const std::string& s);
};
