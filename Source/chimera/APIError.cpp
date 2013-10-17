#include "APIError.h"
namespace chimera
{
    ErrorCode g_lastError;
    ErrorCode APIGetLastError(VOID)
    {
        return g_lastError;
    }

    VOID APISetError(ErrorCode code)
    {
        LOG_ERROR_A("CmError: %d\n", code);
        g_lastError = code;
    }
}