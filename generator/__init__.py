from .resnet_gap import ResnetGenerator
from .resnet_nat import StableGeneratorResnet
from .resnet import GeneratorResnet # old pretrained model

def build_generator(generator='resnet', gap=False, inception=False, nat=False):    
    
    if generator == 'resnet':
        if gap:
            if inception:
                netG = ResnetGenerator(input_nc=3,
                                    inception=True,
                                        output_nc=3,
                                        ngf=64,
                                        norm_type='batch',
                                        act_type='relu',
                                        gpu_ids=[0])
            else:
                netG = ResnetGenerator(input_nc=3,
                                    inception=False,
                                    output_nc=3,
                                    ngf=64,
                                    norm_type='batch',
                                    act_type='relu',
                                    gpu_ids=[0])
        else:
            if nat:
                netG = StableGeneratorResnet(gen_dropout=0.5, data_dim="high", inception=inception)
            else:
                netG = GeneratorResnet(inception=inception)
                
    else:
        raise ValueError(f"Unknown generator type: {generator}")
        
    return netG