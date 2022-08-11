"""
@article{anders2021software,
      author  = {Anders, Christopher J. and
                 Neumann, David and
                 Samek, Wojciech and
                 Müller, Klaus-Robert and
                 Lapuschkin, Sebastian},
      title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
      journal = {CoRR},
      volume  = {abs/2106.13200},
      year    = {2021},
}
"""
import os
from functools import partial

import click
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor,Normalize
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision import datasets

from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import COMPOSITES, EpsilonPlusFlat
from zennit.image import imsave, CMAPS

import zennit.composites as comp


import script 
import loader
from helping_dir.react.models import resnet
#import jakob_react/react/models/resnet


#options to back
ATTRIBUTORS = {
    'gradient': Gradient,
    'smoothgrad': SmoothGrad,
    'integrads': IntegratedGradients,
    'occlusion': Occlusion,
}




@click.command()
@click.argument('dataset-root', type=click.Path(file_okay=False))
@click.argument('relevance_format', type=click.Path(dir_okay=False, writable=True))
@click.option('--model_name', type=click.Choice(['react', 'imagenet','xView2_Jakob', 'rsicd_Jakob','cifar_Jakob',\
 'mnist_Jakob','sen12ms_Jakob', 'So2SatLCZ4_Jakob',\
  'cifar_regression', 'mnist_regression', 'sen13ms_regression',\
  'rsicd_regression', 'xView2_regression', 'So2SatLCZ4_regression',\
  'mnist_regression_rabby','sen12ms_regression_rabby', 'rsicd_regression_rabby',\
  "cifar_regression_rabby", "xView2_regression_rabby",\
  "So2SatLCZ4_regression_rabby","mnist_regression_conrad",\
  "cifar_regression_conrad","sen12ms_regression_conrad",\
  "rsicd_regression_conrad","xView2_regression_conrad",\
  "So2SatLCZ4_regression_conrad", "mnist_regression_samuel",\
  "cifar_regression_samuel", "rsicd_regression_samuel",\
  "xView2_regression_samuel", "So2SatLCZ4_regression_samuel",\
  'sen12ms_regression_samuel' ]), default='imagenet')
@click.option('--parameters', type=click.Path(dir_okay=False))
@click.option(
    '--inputs',
    'input_format',
    type=click.Path(dir_okay=False, writable=True),
    help='Input image format string.  {sample} is replaced with the sample index.'
)

@click.option('--batch-size', type=int, default=256)
@click.option('--max-samples', type=int)
@click.option('--n-outputs', type=int, default=10)
@click.option('--cpu/--gpu', default=True)
@click.option('--relevance-norm', type=click.Choice(['symmetric', 'absolute', 'unaligned']), default='unaligned')
@click.option('--cmap', type=click.Choice(list(CMAPS)), default='coldnhot')
@click.option('--level', type=float, default=1.0)
@click.option('--attributor', 'attributor_name', type=click.Choice(list(ATTRIBUTORS)), default='gradient')
def main(
    dataset_root,
    relevance_format,
    model_name,
    parameters,
    input_format,
    batch_size,
    max_samples,
    n_outputs,
    cpu,
    cmap,
    level,
    relevance_norm,
    attributor_name
):

    # use the gpu if requested and available, else use the cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    #dependingon the model, use different datasets, models and means
    if model_name is "react":
        load = loader.EurosatDataset(True,root_dir = dataset_root)
        model =script.take_model()
    elif model_name is "imagenet":  
        transform = Compose([
          Resize(256),
          CenterCrop(224),
          ToTensor(),
          Normalize( mean = [0.485, 0.456, 0.406],
                          std = [0.229, 0.224, 0.225]),
        ])
        dataset = loader.AllowEmptyClassImageFolder(dataset_root, transform=transform)
        load = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model =models.vgg16(pretrained=True)
    elif model_name is "xView2_Jakob": 
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_xView2()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/classification/resnet50_xView2.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=7)
    elif model_name is "cifar_Jakob" :
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_cifar()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/classification/resnet50_cifar10_in.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=6)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "rsicd_Jakob": 
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_rsicd()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/classification/resnet50_rsicd.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=23)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "mnist_Jakob": 
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_mnist()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/classification/resnet50_mnist_fashion_in.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=7)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "sen12ms_Jakob": 
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_sen12ms()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/classification/resnet50_sen12ms_in.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=9)
    elif model_name is "So2SatLCZ4_Jakob": 
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_So2SatLCZ4()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/classification/resnet50_so2sat_in.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10)
    elif model_name is 'cifar_regression':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_cifar()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_cifar.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is 'mnist_regression':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_mnist()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_mnist.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is 'sen13ms_regression':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_sen12ms()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_sen12ms.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is 'rsicd_regression':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_rsicd()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_jakob_rsicd.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is 'xView2_regression':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_xView2()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_jakob_xView2.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is 'So2SatLCZ4_regression':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_So2SatLCZ4()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_jakob_so2sat.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is 'mnist_regression_rabby':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_mnist()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_rabby_mnist.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is 'sen12ms_regression_rabby':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_sen12ms()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_rabby_sen12ms_in.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is 'rsicd_regression_rabby':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_rsicd()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/regression_resnet50_rabby_rsicd_in.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "cifar_regression_rabby":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_cifar()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_rabby_cifar.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "xView2_regression_rabby":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_xView2()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_rabby_xView2.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is "So2SatLCZ4_regression_rabby":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_So2SatLCZ4()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_rabby_so2sat.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is "mnist_regression_conrad":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_mnist()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_conrad_mnist.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "cifar_regression_conrad":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_cifar()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_conrad_cifar.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "sen12ms_regression_conrad":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_sen12ms()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_conrad_sen12ms.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is "rsicd_regression_conrad":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_rsicd()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_conrad_rsicd.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "xView2_regression_conrad":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_xView2()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_conrad_xView2.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is "So2SatLCZ4_regression_conrad":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_So2SatLCZ4()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_conrad_so2sat.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is "mnist_regression_samuel":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_mnist()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_samuel_mnist.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "cifar_regression_samuel":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_cifar()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_samuel_cifar.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "rsicd_regression_samuel":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_rsicd()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_samuel_rsicd.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
       model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name is "xView2_regression_samuel":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_xView2()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_samuel_xView2.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is "So2SatLCZ4_regression_samuel":
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_tranform_So2SatLCZ4()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_samuel_so2sat.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
    elif model_name is 'sen12ms_regression_samuel':
       cl = loader.Jakob_Loaders(root=dataset_root)
       load = cl.get_transform_sen12ms()
       dirname = os.path.dirname(__file__)
       model_path = os.path.join(dirname, 'saved_models/regression/reg_resnet50_samuel_sen12ms.pth')
       model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)




    eye = torch.eye(n_outputs, device=device)

    # function to compute output relevance given the function output and a target
    def attr_output_fn(output, target):
        # output times one-hot encoding of the target labels of size (len(target), 1000)
        return output * eye[target]
    
    sample_index = 0
    #adjust attributes
    attributor_kwargs = {
        'smoothgrad': {'noise_level': 0.1, 'n_iter': 40},
        'integrads': {'n_iter': 20},
        'occlusion': {'window': (56, 56), 'stride': (28, 28)},
    }.get(attributor_name, {})
    
    count =0
    #determin composite
    composite = comp.EpsilonPlus(epsilon=1)
    attributor = ATTRIBUTORS[attributor_name](model, composite, **attributor_kwargs)
    
    with attributor:
      for data, target in load:
          data_norm = data.to(device)
          data_norm.requires_grad = True
          output_relevance = partial(attr_output_fn, target=target)

          output, relevance = attributor(data_norm, output_relevance)
          
          relevance = np.array(relevance.sum(1).detach().cpu())
          if relevance_norm == 'symmetric':
                  # 0-aligned symmetric relevance, negative and positive can be compared, the original 0. becomes 0.5
                  amax = np.abs(relevance).max((1, 2), keepdims=True)
                  relevance = (relevance + amax) / 2 / amax
          elif relevance_norm == 'absolute':
                  # 0-aligned absolute relevance, only the amplitude of relevance matters, the original 0. becomes 0.
                  relevance = np.abs(relevance)
                  relevance /= relevance.max((1, 2), keepdims=True)
          elif relevance_norm == 'unaligned':
                  # do not align, the original minimum value becomes 0., the original maximum becomes 1.
                  rmin = relevance.min((1, 2), keepdims=True)
                  rmax = relevance.max((1, 2), keepdims=True)
                  relevance = (relevance - rmin) / (rmax - rmin)
          if model_name is "react":
            fname = relevance_format.format(sample=sample_index+count, target = target)
          else:
            fname = relevance_format.format(sample=sample_index+count, target = 0)
          imsave(fname, relevance,vmin=0, vmax=1,cmap=cmap,level=1.0, grid=True)
          count +=1


if __name__ == '__main__':
    main()
