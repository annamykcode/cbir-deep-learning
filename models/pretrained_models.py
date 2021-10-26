from torch import nn
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting):
    """ This helper function sets the .requires_grad attribute of the parameters in the model to False when we
    are feature extracting.

    When we are feature extracting and only want to compute gradients for the newly initialized layer, then we
    want all of the other parameters to not require gradients.

    Args:
        model:
            deep learning model, as pytorch object.
        feature_extracting:
            whether or not we're feature extracting, as boolean.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model( pretrained, num_labels, feature_extracting):
    """ Initialize pretrained deep-learning model and reshape the last layer with the correct number of classes.

    The supported pretrained models are:
        * AlexNet
        * VGG-16
        * GoogLeNet
        * ResNet-50
    Since these deep-learning models have been pretrained on Imagenet, they have output layers of size 1000,
    one node for each class. We reshape the last layers to have the same number of inputs as before, and to
    have the same number of outputs as the number of classes in our the dataset.

    Args:
        pretrained:
            whether or not we want the pretrained version, as boolean.
        num_labels:
            number of labels in our dataset, as integer.
        feature_extracting:
          flag for feature extracting (when False, we finetune the whole model, when True we only update the
          reshaped layer params), as boolean.

    Returns:
        deep-learning model, as pytorch object.
    """
    model = models.vgg16(pretrained=pretrained)

    set_parameter_requires_grad(model, feature_extracting)

    last_layer_in_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(last_layer_in_ftrs, num_labels)

    return model
