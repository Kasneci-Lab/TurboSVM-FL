import torch
import torch.nn.functional as F
import copy
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'

# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef}

# a list that contains resnet50 during runtime
resnet50_list = []

# for FedProx
FedProx_mu = 0.01

# for MOON
MOON_temperature = 0.5
MOON_mu = 1.0

class CNN_femnist(torch.nn.Module):
    """
    CNN model for FEMNIST dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, args: object, image_size: int = 28, num_class: int = 62) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            image_size (int): height / width of images. The images should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(CNN_femnist, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 64 * int(image_size / 4) * int(image_size / 4), out_features = 2048),
            torch.nn.ReLU(),
        )

        self.logits = torch.nn.Linear(in_features = 2048, out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """
        h = self.encoder(x)
        x = self.logits(h)
        return x, h
    
class CNN_celeba(torch.nn.Module):
    """
    CNN model for CelebA dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, args: object, image_size: int = 84, num_class: int = 2) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            image_size (int): height / width of image. The image should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(CNN_celeba, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
        )

        self.logits = torch.nn.Linear(in_features = 32 * int(image_size / 16) * int(image_size / 16), out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """

        h = self.encoder(x)
        x = self.logits(h)
        return x, h
    
class LSTM_shakespeare(torch.nn.Module):
    """
    LSTM model for Shakespeare dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, args: object, embedding_dim: int = 8, hidden_size: int = 256, num_class: int = 80) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            embedding_dim (int): dimension of character embedding.
            hidden_size (int): dimension of LSTM hidden state.
            num_class (int): number of classes (unique characters) in the dataset.
        """

        super(LSTM_shakespeare, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings = num_class, embedding_dim = embedding_dim)
        self.encoder   = torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, batch_first = True)
        self.logits    = torch.nn.Linear(in_features = hidden_size, out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """

        x = self.embedding(x)
        x, (hn, cn) = self.encoder(x)
        h = x[:, -1, :]
        x = self.logits(h)
        return x, h
    
class Resnet50_covid19(torch.nn.Module):
    """
    (Obsolete.) Resnet model for Covid-19 dataset.
    """

    def __init__(self, args: object, num_class: int = 2, freeze: bool = True) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            num_class (int): number of classes in the dataset.
            freeze (bool): (obsolete) whether conducting transfer learning or finetuning.
        """

        super(Resnet50_covid19, self).__init__()

        if len(resnet50_list) == 0:
            resnet50 = torch.hub.load('pytorch/vision:v0.15.2', 'resnet50', weights = 'ResNet50_Weights.DEFAULT')
            resnet50.fc = torch.nn.Identity() # remove last FC layer
            for p in resnet50.parameters(): # freeze resnet50
                p.requires_grad = False
            resnet50.to(device)
            resnet50_list.append(resnet50)
        assert(len(resnet50_list) == 1)
        
        self.logits = torch.nn.Linear(in_features = 2048, out_features = num_class)

        self.t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """

        resnet50 = resnet50_list[0]
        
        # x is of shape (batch_size, 1, 224, 224)
        x = x.expand(-1, 3, -1, -1)
        x = self.t(x)
        h = resnet50(x)
        x = self.logits(h)
        return x, h

def model_train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_client_epoch: int) -> None:
    """
    Train a model.

    Arguments:
        model (torch.nn.Module): pytorch model.
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        num_client_epoch (int): number of training epochs.
    """

    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            p, _ = model(x)
            loss = F.cross_entropy(p, y)
            loss.backward()
        
            optim.step()
            optim.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())

def model_train_FedProx(model: torch.nn.Module, global_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_client_epoch: int) -> None:
    """
    Train a model when FedProx is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        num_client_epoch (int): number of training epochs.
    """

    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            p, _ = model(x)
            loss = F.cross_entropy(p, y)
            
            # FedProx
            for p1, p2 in zip(model.parameters(), global_model.parameters()):
                ploss = (p1 - p2.detach()) ** 2
                loss += FedProx_mu * ploss.sum()

            loss.backward()
            optim.step()
            optim.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())

def model_train_MOON(model: torch.nn.Module, global_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, previous_features: torch.Tensor) -> torch.Tensor:
    """
    Train a model when MOON is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        previous_features (torch.Tensor): features extracted by client model in last global epoch.

    Returns:
        total_features (torch.Tensor): features extracted by client model in current global epoch.
    """

    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    cos = torch.nn.CosineSimilarity(dim=-1)

    for batch_id, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        
        # feed into model
        p, features = model(x)
        if batch_id == 0:
            total_features = torch.empty((0, features.size()[1]), dtype=torch.float32).to(device)
        total_features = torch.cat([total_features, features], dim=0)
        loss = F.cross_entropy(p, y)

        # for MOON
        features_tsne = np.squeeze(features)
        _, global_feat = global_model(x)
        global_feat_copy = copy.copy(global_feat)
        posi = cos(features_tsne, global_feat_copy.to(device))
        logits = posi.reshape(-1,1)
        if previous_features == None or torch.count_nonzero(previous_features) == 0:
            previous_features = torch.zeros_like(features_tsne)
            nega = cos(features_tsne, previous_features)
            logits = torch.cat((posi.reshape(-1,1), nega.reshape(-1,1)), dim=1)
        if previous_features.dim() == 3:
            for prev_feat in previous_features[:, batch_id*y.size()[0]:(batch_id+1)*y.size()[0], :]:
                prev_nega = cos(features_tsne,prev_feat)
                logits = torch.cat((logits, prev_nega.reshape(-1,1)), dim=1)
        
        logits /= MOON_temperature # 0.5
        cos_labels = torch.zeros(logits.size(0)).long().to(device)
        loss_contrastive = F.cross_entropy(logits, cos_labels)
        if torch.count_nonzero(previous_features) != 0:
            loss += MOON_mu * loss_contrastive

        loss.backward()
        optim.step()
        optim.zero_grad()
        
        # stability
        for p in model.parameters():
            torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())
    
    return total_features

def model_eval(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               wandb_log: dict[str, float], 
               metric_prefix: str = 'prefix/', 
               returns: bool = False,
               ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Evaludate the performance of a model with differnt metrics (loss, accuracy, MCC score, precision, recall, F1 score).

    Arguments:
        model (torch.nn.Module): pytorch model.
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
        returns (bool): whether to return ground truth labels and logits, or to calculate metrics

    Returns:
        epoch_labels (torch.Tensor): ground truth labels.
        epoch_predicts (torch.Tensor): logits (not softmaxed yet).

    """

    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].eval()

    model.eval()
    epoch_labels   = []
    epoch_predicts = []
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                
                p, _ = model(x)
                
                epoch_labels  .append(y)
                epoch_predicts.append(p)
   
    epoch_labels   = torch.cat(epoch_labels  ).detach().to('cpu')
    epoch_predicts = torch.cat(epoch_predicts).detach().to('cpu')
    
    if returns:
        return epoch_labels, epoch_predicts
    else:
        cal_metrics(epoch_labels, epoch_predicts, wandb_log, metric_prefix, model.binary)

def cal_metrics(labels: torch.Tensor, preds: torch.Tensor, wandb_log: dict[str, float], metric_prefix: str, binary: bool) -> None:
    """
    Compute metrics (loss, accuracy, MCC score, precision, recall, F1 score) using ground truth labels and logits.

    Arguments:
        labels (torch.Tensor): ground truth labels.
        preds (torch.Tensor): logits (not softmaxed yet).
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
        binary (bool): whether doing binary classification or multi-class classification.
    """

    # loss
    loss = F.cross_entropy(preds, labels)
    wandb_log[metric_prefix + 'loss'] = loss
        
    if not binary: # multi-class    
        # get probability
        preds = torch.softmax(preds, axis = 1)

        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds, multi_class = 'ovr')
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1

        # get class prediction
        preds = preds.argmax(axis = 1)
        
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric
    
    else: # binary
        # get probability
        preds = torch.softmax(preds, axis = 1)[:, 1]
        
        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds)
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1
        
        # get class prediction
        preds = preds.round()
        
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric