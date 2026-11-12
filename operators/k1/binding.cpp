#include <torch/library.h>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <limits>
#include "torch_npu/csrc/framework/OpCommand.h"
extern "C" void meow_k1_launch_queue(void*,uint32_t,void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" void meow_k1_launch_manual(void*,uint32_t,void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
static std::tuple<at::Tensor,at::Tensor> run(const at::Tensor& p,const at::Tensor& m,bool manual,int64_t tile,int64_t blocks){
    TORCH_CHECK(p.device().type()==c10::DeviceType::PrivateUse1 && m.device()==p.device(),"same NPU required");
    TORCH_CHECK(p.scalar_type()==at::kFloat && m.scalar_type()==at::kBool,"FP32/bool required");
    TORCH_CHECK(p.dim()==4 && p.size(3)==3 && m.dim()==3 && p.sizes().slice(0,3)==m.sizes(),"shape mismatch");
    TORCH_CHECK(p.is_contiguous() && m.is_contiguous(),"contiguous inputs required");
    TORCH_CHECK(!p.requires_grad(),"inference only");
    TORCH_CHECK((tile==64 || tile==128 || tile==256) && blocks>0 && blocks<=64,"invalid tiling");
    int64_t b=p.size(0),h=p.size(1),w=p.size(2);
    TORCH_CHECK(h>0 && w>0 && h<INT32_MAX && w<INT32_MAX/12,"invalid dimensions");
    c10_npu::NPUGuard guard(p.device());
    int64_t np=((w*3+7)/8)*8, vp=((w+31)/32)*32;
    auto n=at::empty_strided(p.sizes(),{h*np,np,3,1},p.options());
    auto v=at::empty_strided(m.sizes(),{h*vp,vp,1},m.options());
    if(b){
        uint64_t tasks=b*h*((w+tile-1)/tile);
        auto launch = [p,m,n,v,manual,blocks,tasks,b,h,w,tile,np,vp]() -> int {
            auto stream=c10_npu::getCurrentNPUStream(p.device().index()).stream(false);
            (manual ? meow_k1_launch_manual : meow_k1_launch_queue)(stream,std::min<uint64_t>(blocks,tasks),p.data_ptr(),m.data_ptr(),n.data_ptr(),v.data_ptr(),b,h,w,tile,np,vp);
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi(manual ? "MeowK1Manual" : "MeowK1Queue", launch);
    }
    return {n,v};
}
TORCH_LIBRARY(meow_k1,m){m.def("pointmap_to_normals(Tensor points, Tensor mask, bool manual=False, int tile=128, int blocks=32) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(meow_k1,PrivateUse1,m){m.impl("pointmap_to_normals",TORCH_FN(run));}
