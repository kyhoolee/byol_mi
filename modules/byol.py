"""
MIT License

Copyright (c) 2020 Phil Wang
https://github.com/lucidrains/byol-pytorch/

Adjusted to de-couple for data loading, parallel training
"""

import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

import math


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


'''
Write code for singleton instance attribute 
Call initialize function only once - iff not exist
Then use the initialized instance 
'''
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# loss fn
def soft_cos(x,y, temperature=0.1):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    soft = nn.Softplus()
    result = soft(-1/temperature * cos(x,y))
    return result

def loss_fn(x, y, gpu, loss_type='mi', temperature=0.1):
    '''
    x - from online 
    y - from target 
    '''
    if loss_type == 'mi':
        return mi_loss_fn(x,y,gpu,temperature)
    else:
        return byol_loss(x,y)

def mi_loss_fn(x, y, gpu, temperature=0.1):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    soft = nn.Softplus()
    positive = soft_cos(x,y, temperature)
    
    
    dim = 0
    rotate = [y.shape[dim]-1] + list(range(0, y.shape[dim]-1))
    rotate = torch.LongTensor(rotate)
    
    if x.is_cuda:
        rotate = rotate.cuda(gpu)
    
    # print('X:', x.is_cuda, x.get_device())
    # print('Y:', y.is_cuda, y.get_device())
    # print('Rotate:', rotate.is_cuda, rotate.get_device())
    
    y = torch.index_select(y, dim, rotate)
    negative = soft_cos(x,y, temperature)
    
    result = positive - negative
    
    return result
    
    
    
def byol_loss(x,y):
    # print('Output shape: ', x.shape, y.shape)
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    
    result = 2 - 2 * (x * y).sum(dim=-1)
    # print('Loss: ', result)
    return result


# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        # EMA simple updating function 
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        # Projector still go through several step of input - hidden - output 
        # Not simply a direct linear mapping 
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()

        # 1. Backbone network 
        self.net = net
        self.layer = layer

        # 2. Projector - map from backbone output to projection output 
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        # Projector simply is a MLP to map from backbone output to projection output 
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        
        # 1. Not really understand why need to do this to get representation output 
        if not self.hook_registered:
            self._register_hook()

        if self.layer == -1:
            return self.net(x)

        _ = self.net(x)

        # 2. Using hook and a lot of thing to get hidden - instead of directory assign from backbone net
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x):
        # 1. Get representation from backbone net 
        representation = self.get_representation(x)

        # 2. Get projection from projector 
        # 2.1. Get projector - why need to intialize new projector for every forward step ???
        # And why need to use singleton here ??? 
        projector = self._get_projector(representation)
        # 2.2. Get projection from projector 
        projection = projector(representation)
        return projection


# main class


class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        device,
        loss_type,
        hidden_layer=-2,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        moving_average_decay=0.99,
    ):
        super().__init__()
        
        self.device = device
        self.loss_type = loss_type

        # 1. Online encoder 
        # Backbone network model 
        self.online_encoder = NetWrapper(
            net, projection_size, projection_hidden_size, layer=hidden_layer
        )

        # 2. Target encoder 
        # Backbone network model - Current None - will be copy from online encoder 
        self.target_encoder = None
        # 2.1. Target parameter updater - Exponential moving average 
        self.target_ema_updater = EMA(moving_average_decay)

        # 3. Online preditor 
        # Using to predict online output to target output 
        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))


    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.target_encoder is not None
        ), "target encoder has not been created yet"

        # EMA update for target parameters using current target parameters and online parameters 
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(self, image_one, image_two):

        # Do forward with 2 augmented version of 1 image 
        # Forward with both online and target and take average of loss 
        # A little bit different from original formula - but still the same mearning 

        # 1. Online output - do normally through encoder and predictor 
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        # 2. Target output - do with no grad - encoder no update by backward 
        with torch.no_grad():
            # Why at each forward step - target encoder is updated by same as online encoder ???
            # And why need to use singleton here ??? 
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

        # 3. Calculate loss - target output detach from backward 
        loss_one = loss_fn(online_pred_one, target_proj_two.detach(), self.device, self.loss_type)
        loss_two = loss_fn(online_pred_two, target_proj_one.detach(), self.device, self.loss_type)

        loss = loss_one + loss_two
        return -math.log(4) + loss.mean()