#pragma once

#ifndef _DEBUG
#define NDEBUG
#endif

#include <physx/PxPhysicsAPI.h>

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#endif
//#define DLL_EXPORT __declspec(dllexport)
#define DLL_EXPORT

#include <Windows.h>
#include <windowsx.h>
#include <gdiplus.h>

#ifndef SHR_NEW
#define SHR_NEW new(_NORMAL_BLOCK,__FILE__,__LINE__)
#endif

#ifdef _DEBUG
#define new SHR_NEW
#endif

#include <assert.h>
//#include <memory>

#include <xnamath.h>
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <fastdelegate\FastDelegate.h>

#include <tchar.h>

#define RETURN_IF_FAILED(__assume__) \
    if(!(__assume__)) { return FALSE; } \

#define HR_RETURN_IF_FAILED(__result__) \
    if(FAILED(__result__)) { return FALSE; } \

#include "Logger.h"

#define SAFE_RELEASE(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }
#define SAFE_DELETE(a) if( (a) != NULL ) delete (a); (a) = NULL;
#define SAFE_ARRAY_DELETE(a) if( (a) != NULL ) delete[] (a); (a) = NULL;

#define TBD_FOR(__iterable) for(auto it = __iterable.begin(); it != __iterable.end(); ++it)
#define TBD_FOR_INT(intVal) for(UINT i = 0; i < intVal; ++i)

#pragma warning(disable: 4345) 