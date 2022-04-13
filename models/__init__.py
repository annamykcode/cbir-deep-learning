# import joblib
# import torch
# import torchvision.transforms as transforms
#
# from constants import *
# from models import pretrained_models
#
# from utils.data_prep import get_features
#
# # get available device (CPU/GPU)
# device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
# # app.logger.info(f'Using {device} device ...')
#
# # app.logger.info(f'Loading VGG-16 model from {path_vgg_16} ...')
# # initialize VGG-16
# model = pretrained_models.initialize_model(pretrained=True,
#                                            num_labels=len(LABEL_MAPPING),
#                                            feature_extracting=True)
# # load VGG-16 pretrained weights
# # model.load_state_dict(torch.load(path_vgg_16, map_location='cuda:0'))
# model.load_state_dict(torch.load(PATH_VGG_16, map_location='cpu'))
# # send VGG-16 to CPU/GPU
# model.to(device)
# # register hook
# model.classifier[5].register_forward_hook(get_features())
#
# # app.logger.info(f'Loading PCA model from {path_pca} ...')
# # load PCA pretrained model
# pca = joblib.load(PATH_PCA)
#
# # image transformations
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
