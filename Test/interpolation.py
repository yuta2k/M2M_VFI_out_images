#!/usr/bin/env python

import numpy
import PIL.Image
import torch

import os
import argparse
import re
import shutil

import model.m2m as m2m

##########################################################
torch.set_grad_enabled(False) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

netNetwork = m2m.M2M_PWC().cuda().eval()

netNetwork.load_state_dict(torch.load('./model.pkl'))

##########################################################

parser = argparse.ArgumentParser()

parser.add_argument('--src', type=str, required=True)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--dst_filename', type=str, default='%08d.png')
# Ex. specified 2, 30fps to 60fps
parser.add_argument('--factor', type=int, default=2)

args = parser.parse_args()

inv_factor = 1.0 / args.factor

##########################################################

def doInterpolate(srcFirst, srcSecond, dstFirstNum):
    npyOne = numpy.array(PIL.Image.open(srcFirst))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
    npyTwo = numpy.array(PIL.Image.open(srcSecond))[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)

    tenOne = torch.FloatTensor(numpy.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()

    for i in range(args.factor - 1):
        dstPath = os.path.join(args.dst, f'{args.dst_filename}' % (dstFirstNum + i))
        tenEstimate = netNetwork(tenOne, tenTwo, [torch.FloatTensor([inv_factor * (i + 1)]).view(1, 1, 1, 1).cuda()])[0]

        # `[:,:,::-1]` : reverse from BGR to RGB
        npyEstimate = (tenEstimate.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)[:,:,::-1] * 255.0) \
            .clip(0.0, 255.0).round().astype(numpy.uint8)

        PIL.Image.fromarray(npyEstimate).save(dstPath)
        print('[M2M_VFI] \'%s\' interpolated!'%dstPath)

if not os.path.exists(args.dst):
    os.mkdir(args.dst)

imgFileRegex = re.compile('^(\d+)\.(jpg|png|jpeg|bmp)$')
files = [f for f in os.listdir(args.src) if os.path.isfile(os.path.join(args.src, f))]
imgFiles = [f for f in files if imgFileRegex.match(f)]
imgFiles.sort(key=lambda x: int(imgFileRegex.match(x).group(1)))

outIndex = 1
for i, imgFile in enumerate(imgFiles):
    currSrcPath = os.path.join(args.src, imgFile)

    cpDstPath = os.path.join(args.dst, f'{args.dst_filename}' % outIndex)

    shutil.copyfile(currSrcPath, cpDstPath)
    print('[M2M_VFI] \'%s\' copied!'%cpDstPath)

    if i + 1 < len(imgFiles):
        nextSrcPath = os.path.join(args.src, imgFiles[i + 1])
        doInterpolate(currSrcPath, nextSrcPath, outIndex + 1)

    outIndex += args.factor

print('Completed')
