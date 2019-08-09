import torch

if __name__ == '__main__':

    source_model = '../models/fb-scc-4.pth'
    target_model = '../models/fb-scc-4-no-routing.pth'

    model_dict = torch.load(source_model, map_location='cpu')['model']

    new_dict = {}
    for name, var in model_dict.items():
        if name.find('routing_fc') != -1:
            continue
        new_dict[name] = var

    new_cpt = {'model': new_dict}
    torch.save(new_cpt, target_model)
