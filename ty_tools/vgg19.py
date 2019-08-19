import torchvision
import torch
import torch.nn as nn

from torchvision.models.vgg import make_layers, load_state_dict_from_url, model_urls, vgg19

cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'stage1': [64, 64,'M'],
    'stage2': [64, 64, 'M', 128, 128, 'M'],
    'stage3': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,],
    'stage4': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, ],
    'stage5': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class AGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(AGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def create_vgg19(cfg='stage2'):
    #net = vgg19(pretrained=True)
    #for n, p in net.named_parameters():
    #    print(n, 'aa')

    model = AGG(make_layers(cfgs[cfg], batch_norm=False))

    #print(net)
    #print(model)
    #for n, p in model.named_parameters():
    #    print(n)

    state_dict = load_state_dict_from_url(model_urls["vgg19"])
    a, b = model.load_state_dict(state_dict, strict=False)
    print('missing keys: ', a)

    return model



if __name__ == '__main__':

    net = create_vgg19('stage2')
    inp = torch.ones((10,3,224,224))
    y = net(inp)
    print(y.shape)
