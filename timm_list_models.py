import timm

model_names = timm.list_models(pretrained=True)
with open('timm_list_models.txt', 'w') as f:
    for mn in model_names:
        f.writelines(mn+'\n')
