from train_utils import output_to_class
from sklearn.metrics import f1_score


def get_mean_F1(model, validation_loader):
    """
    returns the mean F1 score for a given dataloader
    """
    model.eval()
    mean_f1 = 0
    for (data, target) in validation_loader:
            output = model(data)
            mean_f1 += f1_score(target, output_to_class(output), average='micro') / len(validation_loader)
            
    return mean_f1