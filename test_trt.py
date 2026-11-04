import tensorrt as trt
import numpy as np
import torch
from collections import OrderedDict, namedtuple
import torch.nn as nn
import cv2
import os
import torchvision.transforms as T
import shutil

transform = T.Compose([
    T.ToTensor(),                  # HWC [0..255] -> CHW float32 [0..1]
])

def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

class TRT(nn.Module):
    def __init__(self,weight='model_best.engine'):
        super().__init__()
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
    def forward(self, im):
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]

class NEW_TRT(nn.Module):
    def __init__(self,weight='model_best.engine'):
        super().__init__()
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(model.num_io_tensors):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(name, tuple(model.get_profile_shape(0, i)[2]))
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_tensor_shape(name))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
    def forward(self, im):
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (left, top, right, bottom), new_unpad

def Run(model, img, device, input_size=(384, 640)):
    colors = [(205, 206, 244)]
    original_height, original_width = img.shape[:2]
    img, _, padding, _ = letterbox(img, new_shape=input_size)
    
    img = transform(img)
    img = img.unsqueeze(0)

    img = img.cuda().float() # uint8 to fp16/32
    if img.ndimension() == 3:
        img = torch.unsqueeze(img, 0)
    img = img.cuda()
    # start = time.time()
    img_out = model(img)
    # print("FPS:", 1/(time.time()-start))
    x0=img_out[0]

    _,da_predict=torch.max(x0, 1)

    pad_left, pad_top, pad_right, pad_bottom = padding
    DA = da_predict.byte().cpu().data.numpy()[0]
    DA = DA[pad_top:DA.shape[0] - pad_bottom, pad_left:DA.shape[1] - pad_right]
    DA = cv2.resize(DA, (original_width, original_height), interpolation=cv2.INTER_NEAREST) * 255
    img_rs = np.zeros((original_height, original_width, 3))
    
    img_rs[DA>100]=[117, 240, 107]
    
    if img_rs.dtype == np.float64:  
        img_rs = cv2.convertScaleAbs(img_rs) 
    img_rs = cv2.resize(img_rs, (320, 240))
    img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img_rs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_image=np.zeros((240,320,3),np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(black_image, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel)
    black_image = cv2.erode(black_image, kernel, iterations = 3)
    black_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
    
    return black_image

if __name__ == ""'__main__':
    model = NEW_TRT('./test_full/model_39.engine')

    image_list=os.listdir('./inference/images')
    shutil.rmtree('./inference/results')
    os.mkdir('./inference/results')
    device = select_device(None, '0')
    
    from LaneKeeping import LaneKeeping
    lanekeeping = LaneKeeping()
    
    for i, imgName in enumerate(image_list):
        img = cv2.imread(os.path.join('./inference/images',imgName))
        
        # Segment
        img=Run(model, img, device)
        
        # Lane Keeping
        speed, angle, img, deviation = lanekeeping.AngCal(img)
        
        # print(len(img))
        cv2.imwrite(os.path.join('./inference/results',imgName),img)
