#include "k1_impl.h"
extern "C" __global__ __aicore__ void k1_manual(GM_ADDR p,GM_ADDR m,GM_ADDR n,GM_ADDR v,uint64_t b,uint32_t h,uint32_t w,uint32_t t,uint32_t np,uint32_t vp){K1<true> op;op.Init(p,m,n,v,b,h,w,t,np,vp);op.Run();}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void meow_k1_launch_manual(void* stream,uint32_t blocks,void* p,void* m,void* n,void* v,uint64_t b,uint32_t h,uint32_t w,uint32_t t,uint32_t np,uint32_t vp){
 k1_manual<<<blocks,nullptr,stream>>>((uint8_t*)p,(uint8_t*)m,(uint8_t*)n,(uint8_t*)v,b,h,w,t,np,vp);
}
#endif
