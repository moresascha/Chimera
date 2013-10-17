#include <windows.h>
#include "Cudah.h"
#include "Buffer.h"
#include "Fill.h"
#include "Module.h"
#include "Kernel.h"
#include "Sort.h"
#include <sstream>
#include <fstream>

int main(void)
{
    cudah::Init();

    /*

    UINT sortDataCnt = 1024;
    cudah::cuBuffer<FLOAT> csortData(sortDataCnt);
    
    for(cudah::cuIterator<FLOAT, BufferPtr, cudah::cuBuffer<FLOAT>> it = csortData.Begin(); it != csortData.End(); ++it)
    {
        it = (FLOAT)rand() / (FLOAT)RAND_MAX;
    }

    for(uint i = 0; i < sortDataCnt; ++i)
    {
        DEBUG_OUT_A("%f ", csortData[i]);
    }

    DEBUG_OUT("\n");

    bitonicSortPerGroup(csortData, 1024, 0, 0, 1024);

    for(uint i = 0; i < sortDataCnt; ++i)
    {
        DEBUG_OUT_A("%f ", csortData[i]);
    }

    DEBUG_OUT("\n");*/

    
    cudah::cuBuffer<INT> b(64);
    cudah::Fill(b.Begin(), b.End(), 10);

    cudah::cuBuffer<INT> c(1024);

    cudah::Generate(c.Begin(), c.End(), rand);
    //cudah::Copy(b.Begin(), b.End(), c.Begin(), c.End());
/*
    cudah::cuModule m("./ptx/KD.ptx");

    cudah::cuKernel kernel(m.GetFunction("spreadContent"));*/
    cudah::Sort(c);
    std::ofstream file;
    file.open("ouput.txt");
    TBD_FOR_INT(c.Size())
    {
        {
            std::stringstream ss;
            ss << c[i];
            ss << " ";
            OutputDebugStringA(ss.str().c_str());
            //file << c[i] << std::endl;
        }
    }
    file.close();
    system("pause");
    cudah::Destroy();

    return 0;
}