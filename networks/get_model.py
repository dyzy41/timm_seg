import networks


def get_net(model_name, in_c=3, num_class=6, img_size=512):
    model_class = getattr(networks, model_name)
    model = model_class(in_c, num_class)
    return model