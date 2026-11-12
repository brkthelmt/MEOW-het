#pragma once
#include "kernel_operator.h"
using namespace AscendC;

// Identical arithmetic and tiling for both routes. Row-padded OUTPUT strides
// prevent unaligned byte-mask writes by adjacent cores from sharing a block.
template<bool MANUAL> class K1 {
    TPipe pipe;
    TQue<TPosition::VECIN,1> iq;
    TQue<TPosition::VECOUT,1> oq;
    TBuf<TPosition::VECCALC> ib, ob, scratch, indices, halfbuf, predicate;
    GlobalTensor<float> p,norm;
    GlobalTensor<uint8_t> m,valid;
    uint32_t T,H,W,NP,VP,C;
    uint64_t B;
    TEventID inReady,outReady,outFree,scalarReady,vectorDone;
    LocalTensor<float> f;
    LocalTensor<uint32_t> ix;
    __aicore__ inline LocalTensor<float> F(int slot) {return f[slot*T];}
    __aicore__ inline void Barrier(){PipeBarrier<PIPE_V>();}
    __aicore__ inline void Norm(int base) {
        Mul(F(30),F(base),F(base),C); Barrier();
        Mul(F(31),F(base+1),F(base+1),C); Barrier();
        Add(F(30),F(30),F(31),C); Barrier();
        Mul(F(31),F(base+2),F(base+2),C); Barrier();
        Add(F(30),F(30),F(31),C); Barrier();
        Sqrt(F(30),F(30),C); Barrier();
        Adds(F(30),F(30),1.e-12f,C); Barrier();
        for(int c=0;c<3;++c){Div(F(base+c),F(base+c),F(30),C); Barrier();}
    }
    __aicore__ inline void Load(LocalTensor<uint8_t> raw, uint64_t row, uint32_t x,uint32_t len){
        // Slots: center, up, left, down, right. Invalid rows use a safe row,
        // then Compute selects literal zero (not NaN * zero).
        for(int k=0;k<5;++k){
            uint64_t rr=row; uint32_t xx=x, count=len, left=0,right=0;
            uint32_t y=row%H;
            if(k==1 && y>0) rr=row-1;
            if(k==3 && y+1<H) rr=row+1;
            if(k==2){if(x>0)xx=x-1;else {left=1;count=len-1;}}
            if(k==4){xx=x+1;if(x+len==W){count=len-1;right=1;} if(count==0)xx=x;}
            // For a zero-length horizontal span, load one safe center pixel;
            // Compute zeroes the invalid lane. Never issue a zero-length DMA.
            if(count==0){count=1;left=0;right=0;xx=x;}
            DataCopyExtParams pp{1,static_cast<uint32_t>(12*count),0,0,0};
            DataCopyPadExtParams<float> pad{true,static_cast<uint8_t>(3*left),static_cast<uint8_t>(3*right),0.f};
            DataCopyPad(raw[k*12*T].template ReinterpretCast<float>(),p[(rr*W+xx)*3],pp,pad);
            DataCopyExtParams mp{1,count,0,0,0};
            DataCopyPadExtParams<uint8_t> pm{true,static_cast<uint8_t>(left),static_cast<uint8_t>(right),0};
            DataCopyPad(raw[60*T+k*T],m[rr*W+xx],mp,pm);
        }
    }
    __aicore__ inline void Compute(LocalTensor<uint8_t> input,LocalTensor<uint8_t> output,
                                   uint32_t y,uint32_t x,uint32_t len){
        C=len;
        auto h=halfbuf.Get<half>();
        // Mask byte conversion is deliberately two-stage on A2.
        for(int k=0;k<5;++k){
            Cast(h,input[60*T+k*T],RoundMode::CAST_NONE,C); Barrier();
            Cast(F(15+k),h,RoundMode::CAST_NONE,C); Barrier();
            for(int c=0;c<3;++c){
                Gather(F(k*3+c),input[k*12*T].template ReinterpretCast<float>(),ix,c*4,C); Barrier();
            }
        }
        if(y==0){for(int c=0;c<3;++c)Duplicate(F(3+c),0.f,C); Duplicate(F(16),0.f,C);}
        if(y+1==H){for(int c=0;c<3;++c)Duplicate(F(9+c),0.f,C); Duplicate(F(18),0.f,C);}
        Barrier();
        // Only border lanes require scalar edits, after all vector writes.
        SetFlag<HardEvent::V_S>(vectorDone); WaitFlag<HardEvent::V_S>(vectorDone);
        if(x==0){for(int c=0;c<3;++c)F(6+c).SetValue(0,0.f); F(17).SetValue(0,0.f);}
        if(x+len==W){for(int c=0;c<3;++c)F(12+c).SetValue(len-1,0.f); F(19).SetValue(len-1,0.f);}
        SetFlag<HardEvent::S_V>(scalarReady); WaitFlag<HardEvent::S_V>(scalarReady);
        for(int k=1;k<5;++k)for(int c=0;c<3;++c){Sub(F(k*3+c),F(k*3+c),F(c),C);Barrier();}
        Duplicate(F(29),0.f,T); Barrier();
        for(int i=0;i<4;++i){
            int a=3*(i+1),b=3*((i+1)%4+1);
            for(int c=0;c<3;++c){
                int u=(c+1)%3,v=(c+2)%3;
                Mul(F(20+c),F(a+u),F(b+v),C); Barrier();
                Mul(F(31),F(a+v),F(b+u),C); Barrier();
                Sub(F(20+c),F(20+c),F(31),C); Barrier();
            }
            Norm(20);
            Mul(F(28),F(15),F(16+i),C); Barrier();
            Mul(F(28),F(28),F(16+(i+1)%4),C); Barrier();
            Max(F(29),F(29),F(28),C); Barrier();
            for(int c=0;c<3;++c){
                Mul(F(20+c),F(20+c),F(28),C); Barrier();
                if(i==0) Muls(F(24+c),F(20+c),1.f,C);
                else Add(F(24+c),F(24+c),F(20+c),C);
                Barrier();
            }
        }
        Norm(24);
        auto pred=predicate.Get<uint8_t>();
        // A2 CompareScalar count must cover full 64-FP32 repeats.
        // Union tail is explicitly zero, so compare all T lanes safely.
        CompareScalar(pred,F(29),0.f,CMPMODE::NE,T); Barrier();
        for(int c=0;c<3;++c){Select(F(24+c),pred,F(24+c),0.f,SELMODE::VSEL_TENSOR_SCALAR_MODE,C);Barrier();}
        // ix[ T : T+3T ] holds the SoA -> AoS indices.
        Gather(output.template ReinterpretCast<float>(),F(24),ix[T],0,3*C); Barrier();
        Cast(h,F(29),RoundMode::CAST_NONE,C); Barrier();
        Cast(output[12*T],h,RoundMode::CAST_NONE,C); Barrier();
    }
public:
    __aicore__ inline void Init(GM_ADDR P,GM_ADDR M,GM_ADDR N,GM_ADDR V,uint64_t b,
                                uint32_t h,uint32_t w,uint32_t tile,uint32_t np,uint32_t vp){
        B=b; H=h; W=w; T=tile; NP=np; VP=vp;
        p.SetGlobalBuffer((__gm__ float*)P);m.SetGlobalBuffer((__gm__ uint8_t*)M);
        norm.SetGlobalBuffer((__gm__ float*)N);valid.SetGlobalBuffer((__gm__ uint8_t*)V);
        if constexpr(MANUAL){pipe.InitBuffer(ib,65*T);pipe.InitBuffer(ob,13*T);}
        else{pipe.InitBuffer(iq,1,65*T);pipe.InitBuffer(oq,1,13*T);}
        pipe.InitBuffer(scratch,40*4*T);pipe.InitBuffer(indices,4*4*T);
        pipe.InitBuffer(halfbuf,2*T);pipe.InitBuffer(predicate,T);
        f=scratch.Get<float>();ix=indices.Get<uint32_t>();
        inReady=pipe.FetchEventID(HardEvent::MTE2_V);
        outReady=pipe.FetchEventID(HardEvent::V_MTE3);
        outFree=pipe.FetchEventID(HardEvent::MTE3_V);
        scalarReady=pipe.FetchEventID(HardEvent::S_V);
        vectorDone=pipe.FetchEventID(HardEvent::V_S);
        for(uint32_t j=0;j<T;++j)ix.SetValue(j,12*j);
        for(uint32_t j=0;j<3*T;++j)ix.SetValue(T+j,4*((j%3)*T+j/3));
        SetFlag<HardEvent::S_V>(scalarReady);WaitFlag<HardEvent::S_V>(scalarReady);
    }
    __aicore__ inline void Run(){
        uint32_t cols=(W+T-1)/T;
        for(uint64_t id=GetBlockIdx();id<B*H*cols;id+=GetBlockNum()){
            uint64_t row=id/cols;uint32_t x=(id%cols)*T,len=(W-x<T?W-x:T);
            LocalTensor<uint8_t> a,b;
            if constexpr(MANUAL){a=ib.Get<uint8_t>(); b=ob.Get<uint8_t>();}
            else{a=iq.AllocTensor<uint8_t>();}
            Load(a,row,x,len);
            if constexpr(MANUAL){SetFlag<HardEvent::MTE2_V>(inReady);WaitFlag<HardEvent::MTE2_V>(inReady);}
            else{iq.EnQue(a);a=iq.DeQue<uint8_t>();b=oq.AllocTensor<uint8_t>();}
            Compute(a,b,row%H,x,len);
            if constexpr(MANUAL){SetFlag<HardEvent::V_MTE3>(outReady);WaitFlag<HardEvent::V_MTE3>(outReady);}
            else{iq.FreeTensor(a);oq.EnQue(b);b=oq.DeQue<uint8_t>();}
            DataCopyExtParams pn{1,12*len,0,0,0},pv{1,len,0,0,0};
            DataCopyPad(norm[row*NP+3*x],b.template ReinterpretCast<float>(),pn);
            DataCopyPad(valid[row*VP+x],b[12*T],pv);
            if constexpr(MANUAL){SetFlag<HardEvent::MTE3_V>(outFree);WaitFlag<HardEvent::MTE3_V>(outFree);}
            else{oq.FreeTensor(b);}
        }
        PipeBarrier<PIPE_ALL>();
        pipe.ReleaseEventID<HardEvent::MTE2_V>(inReady);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(outReady);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(outFree);
        pipe.ReleaseEventID<HardEvent::S_V>(scalarReady);
        pipe.ReleaseEventID<HardEvent::V_S>(vectorDone);
    }
};
