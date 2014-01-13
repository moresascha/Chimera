#pragma once

#pragma warning(disable: 4345) 
#pragma warning(disable: 4250) //inherits via dominance

#ifndef _DEBUG
#define NDEBUG
#endif

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#endif

#include "api/ChimeraAPI.h"
#include "APIError.h"

#include <gdiplus.h>
#pragma comment (lib, "Gdiplus.lib")

#ifndef SHR_NEW
#define SHR_NEW new(_NORMAL_BLOCK,__FILE__,__LINE__)
#endif

#ifdef _DEBUG
#define new SHR_NEW
#endif
