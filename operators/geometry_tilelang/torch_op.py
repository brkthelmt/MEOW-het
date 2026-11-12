"""Dispatcher and exact FakeTensor layouts for capture of K2/K3."""
import torch
from .kernels import normal_candidates as _k2, mask_and_apply as _k3


@torch.library.custom_op('meow_tilelang::normal_candidates',mutates_args=(),device_types='npu')
def normal_candidates(normals:torch.Tensor, valid:torch.Tensor, degrees:float=5.0)->torch.Tensor:
    return _k2(normals,valid,degrees)


@normal_candidates.register_fake
def _k2_fake(normals,valid,degrees=5.0):
    b,h,w=valid.shape;pad=((w+31)//32)*32
    return torch.empty_strided((b,h,w),(h*pad,pad,1),device=valid.device,dtype=torch.bool)


@torch.library.custom_op('meow_tilelang::mask_and_apply',mutates_args=(),device_types='npu')
def mask_and_apply(candidates:torch.Tensor, depth:torch.Tensor, mask:torch.Tensor,
                   points:torch.Tensor, camera_points:torch.Tensor, depth_tol:float=0.03)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return _k3(candidates,depth,mask,points,camera_points,depth_tol)


@mask_and_apply.register_fake
def _k3_fake(candidates,depth,mask,points,camera_points,depth_tol=0.03):
    b,h,w=depth.shape;npad=((w*3+7)//8)*8;spad=((w+7)//8)*8;mpad=((w+31)//32)*32
    f=torch.empty_strided((b,h,w),(h*mpad,mpad,1),device=points.device,dtype=torch.bool)
    p=torch.empty_strided((b,h,w,3),(h*npad,npad,3,1),device=points.device,dtype=torch.float32)
    c=torch.empty_strided((b,h,w,3),(h*npad,npad,3,1),device=points.device,dtype=torch.float32)
    d=torch.empty_strided((b,h,w),(h*spad,spad,1),device=points.device,dtype=torch.float32)
    z=torch.empty_strided((b,h,w),(h*spad,spad,1),device=points.device,dtype=torch.float32)
    return f,p,c,d,z
