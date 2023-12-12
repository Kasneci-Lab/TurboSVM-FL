import numpy  as np
import torch
import math
import tqdm
import copy
from sklearn import svm
import wandb
from models import model_eval, cal_metrics
from utils import weighted_avg_params, weighted_avg
from torchmetrics.functional import pairwise_cosine_similarity

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FedAwS cosine similarity margin
margin = 0
    
def federated_learning(args: object, train_clients: list[object], test_clients: list[object], global_model: torch.nn.Module) -> None:
    """
    Main loop for federated learning.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        train_clients (list[Client]): training clients.
        test_clients (list[Client]): test / validation clients.
        global_model (torch.nn.Module): pytorch model (global model on the server).
    """

    # determine how many clients are updated per global round
    num_train_client  = len(train_clients)
    if args.client_C < 1.0: # proportion
        num_update_client = min(max(math.ceil(args.client_C * num_train_client), 1), num_train_client) # number of clients to update per round
    else: # client_C itself is num_update_client
        num_update_client = min(args.client_C, num_train_client)
    print("\nnumber of updated clients per round during training:", num_update_client)
    
    # global optimizer
    global_model.to(device)
    global_optim = args.global_optim(global_model.parameters(), lr = args.global_lr, amsgrad = args.amsgrad)
    logits_optim = args.logits_optim(global_model.logits.parameters(), lr = args.logits_lr, eps = 1e-5)

    # train-valid-test split on server level
    global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in train_clients])
    global_test_dataset  = torch.utils.data.ConcatDataset([c.dataset for c in test_clients ])
    global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False)
    global_test_loader   = torch.utils.data.DataLoader(global_test_dataset , batch_size = args.global_bs, shuffle = False)
    
    # performance before training
    wandb_log = {}
    model_eval(global_model, global_train_loader, wandb_log, 'train/')
    model_eval(global_model, global_test_loader , wandb_log, 'test/' )
    wandb.log(wandb_log)
    
    # for MOON
    previous_features = None
    
    # global round loop
    print()
    for current_global_epoch in tqdm.tqdm(range(args.global_epoch)):
        # select clients which are updated in this round
        update_clients = np.random.choice(train_clients, num_update_client, replace = False)
        client_weights = [c.num_sample for c in update_clients]
        client_models  = [copy.deepcopy(global_model) for c in update_clients]
        
        # training
        for client, client_model in zip(update_clients, client_models):
            previous_features = client.local_train(client_model, global_model, previous_features)
        
        # global model aggregation
        eval(args.fed_agg)(global_model, client_models, client_weights, # basic FL parameters
                           global_optim, # for FedOpt (FedAdam and FedAMS)
                           logits_optim, # for FedAwS and Ours
                           current_global_epoch, args.global_epoch, args.class_C, args.base_agg, args.agg_svc, args.spreadout) # for ours
        
        # stability
        for p in global_model.parameters():
            torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

        # performance metrics
        global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in update_clients])
        global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False)
        wandb_log = {}
        model_eval(global_model, global_train_loader, wandb_log, 'train/')
        model_eval(global_model, global_test_loader , wandb_log, 'test/' )
        wandb.log(wandb_log)
        
    # global_model.to('cpu')
    # wandb.finish()

def server_eval(clients: list[object], wandb_log: dict[str, float], metric_prefix: str) -> None:
    """
    (Obsolete.) Evaluate model performance globally by letting each client conduct inference locally and then collecting all inferences and calculating metrics.

    Arguments:
        clients (list[Client]): list of clients.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """

    labels = []
    preds  = []
    for c in clients:
        l, p = c.local_eval()
        labels.append(l)
        preds .append(p)
    labels = torch.cat(labels)
    preds  = torch.cat(preds )
    cal_metrics(labels, preds, wandb_log, metric_prefix)    

def FedAvg(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], *_) -> None:
    """
    Federated learning algorithm FedAvg.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
    """

    client_params  = [m.state_dict() for m in client_models]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    global_model.load_state_dict(new_global_params)

def FedOpt(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], global_optim: torch.optim, *_) -> None:
    """
    Federated learning algorithm FedOpt. Depending on the choice of optimizer, it can be deviated into different variates like FedAdam and FedAMS.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): pytorch optimizer for global model.
    """

    client_params  = [m.state_dict() for m in client_models]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    
    # pseudo-gradient
    global_model.train()
    for p_name, p in global_model.named_parameters():
        if p.requires_grad:
            p.grad = global_model.state_dict()[p_name] - new_global_params[p_name].to(p.device)
    
    # apply optimizer
    global_optim.step()
    global_optim.zero_grad()

def FedAwS(global_model: torch.nn.Module, 
           client_models: list[torch.nn.Module], 
           client_weights: list[int], 
           global_optim: torch.optim, 
           logits_optim: torch.optim, 
           *_) -> None:
    """
    Federated learning algorithm FedAwS.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): (useless) pytorch optimizer for global model.
        logits_optim (torch.optim): pytorch optimizer for logit layer of global model.
    """

    FedAvg(global_model, client_models, client_weights)
    global_model.train()
    
    # spreadout regularizer
    wb = torch.cat((global_model.logits.weight, global_model.logits.bias.view(-1, 1)), axis = 1)
    cos_sim_mat = pairwise_cosine_similarity(wb)
    cos_sim_mat = (cos_sim_mat > margin) * cos_sim_mat
    loss = cos_sim_mat.sum() / 2
    loss.backward()
    
    # apply optimizer
    logits_optim.step()
    logits_optim.zero_grad()

# TurboSVM-FL
def Ours(global_model: torch.nn.Module, 
         client_models: list[torch.nn.Module], 
         client_weights: list[int], 
         global_optim: torch.optim, 
         logits_optim: torch.optim, 
         current_global_epoch: int, 
         num_global_epoch: int, 
         class_C: int | float, 
         base_agg: str, 
         agg_svc: bool, 
         spreadout: bool,
         ) -> None:
    """
    Federated learning algorithm TurboSVM-FL.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): pytorch optimizer for global model.
        logits_optim (torch.optim): pytorch optimizer for logit layer of global model.
        current_global_epoch (int): current global aggregation round.
        num_global_epoch (int): total number of global aggregation rounds.
        class_C (int|float): number of classes used to train SVM. Can be int (absolute value) or float (proportion).
        base_agg (str): FL aggregation algorithm for encoder (all layers of global model but logit layer).
        agg_svc (bool): whether to aggregate only client models that form support vectors or all client models.
        spreadout (bool): whether to apply max-margin spread-out regularization or not.
    """

    # aggregate client models
    assert(base_agg == 'FedAvg' or base_agg == 'FedOpt')
    eval(base_agg)(global_model, client_models, client_weights, global_optim)
    
    # randomly select classes, whose embeddings will be updated using ours
    num_class = global_model.logits.weight.shape[0]
    if class_C <= 1.0: # proportion
        num_update_class = min(max(math.floor(class_C * num_class), 2), num_class)
    else: # class_C itself is num_update_class
        num_update_class = min(class_C, num_class)
    classes = np.random.choice(num_class, num_update_class, replace = False).tolist()
    classes.sort() # must be sorted when more than 2 classes, as svm coefs are sorted based on class id
    # print("number of SVC class:", num_update_class)
    # print(classes)
    
    # class embeddings, class labels, and client weights
    x, y, w = [], [], []
    for m, cw in zip(client_models, client_weights):
        wb = torch.cat((m.logits.weight[classes], m.logits.bias[classes].view(-1, 1)), axis = 1).detach()
        x.append(wb)
        y += classes
        w += [cw] * num_update_class
    x = torch.cat(x)
    assert(len(x) == len(y) == len(w)) # x, y, w should have same length

    # fit SVM
    C = (num_global_epoch - current_global_epoch) / num_global_epoch # decreasing C successively --> increasing number of support vectors successively
    clf = svm.SVC(kernel = 'linear', max_iter = 50, tol = 1e-3, C = C)
    clf.fit(x, y)

    # collect support vectors and their weights
    SVCs = {class_id : [] for class_id in classes}
    ws   = {class_id : [] for class_id in classes}
    for svc_id in clf.support_:
        svc   = x[svc_id]
        svc_w = w[svc_id]
        class_id = y[svc_id]
        
        SVCs[class_id].append(svc)
        ws  [class_id].append(svc_w)

    # aggregate support vectors and update logit weight as well as logit bias
    if agg_svc:
        with torch.no_grad():
            for class_id in classes:
                new_wb = weighted_avg(SVCs[class_id], ws[class_id])
                global_model.logits.weight[class_id] = new_wb[:-1]
                global_model.logits.bias  [class_id] = new_wb[-1]

    # apply spreadout regularization
    if spreadout:
        loss = 0
        h_id = 0 # id for hyperplane
        for class_a_idx in range(num_update_class):
            class_a = classes[class_a_idx]
            w_a = torch.cat([global_model.logits.weight[class_a], global_model.logits.bias[class_a].view(-1)])
            
            for class_b_idx in range(class_a_idx + 1, num_update_class):
                class_b = classes[class_b_idx]
                w_b = torch.cat([global_model.logits.weight[class_b], global_model.logits.bias[class_b].view(-1)])

                hyperplane_ab = torch.tensor(clf.coef_[h_id]).float().to(device)
                similarity = torch.exp(- ((torch.dot(w_a - w_b, hyperplane_ab) / hyperplane_ab.norm()) ** 2) / 2)
                loss += similarity

                h_id += 1

        loss.backward()
        logits_optim.step()
        logits_optim.zero_grad()
