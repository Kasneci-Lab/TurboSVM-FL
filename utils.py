import torch
import copy
import numpy as np
import random
import argparse
from datetime import datetime

# self-defined functions
from client import get_clients
from models import CNN_femnist, CNN_celeba, LSTM_shakespeare, Resnet50_covid19
from data_preprocessing import get_data_dict_femnist, get_data_dict_celeba, get_data_dict_shakespeare, get_data_dict_covid19

def seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
def Args() -> argparse.Namespace:
    """
    Helper function for argument parsing.

    Returns:
        args (argparse.Namespace): parsed argument object.
    """

    parser = argparse.ArgumentParser()
    
    # path parameters
    parser.add_argument('--femnist_train_path', type = str, default = '../femnist/train/all_data_0_niid_0_keep_0_train_9.json', help = 'femnist train json path')
    parser.add_argument('--femnist_test_path' , type = str, default = '../femnist/test/all_data_0_niid_0_keep_0_test_9.json'  , help = 'femnist test json path')
    parser.add_argument('--celeba_train_path' , type = str, default = '../celeba/train/all_data_0_0_keep_5_train_9.json', help = 'celeba train json path')
    parser.add_argument('--celeba_test_path'  , type = str, default = '../celeba/test/all_data_0_0_keep_5_test_9.json'  , help = 'celeba test json path')
    parser.add_argument('--celeba_image_path' , type = str, default = '../celeba/img_align_celeba/', help = 'celeba image dir path')
    parser.add_argument('--shakespeare_train_path', type = str, default = '../shakespeare/train/all_data_0_0_keep_0_train_9.json', help = 'shakespeare train json path')
    parser.add_argument('--shakespeare_test_path' , type = str, default = '../shakespeare/test/all_data_0_0_keep_0_test_9.json'  , help = 'shakespeare test json path')
    parser.add_argument('--covid19_train_path', type = str, default = '../CC19/train/', help = 'covid19 train dir path')
    parser.add_argument('--covid19_test_path' , type = str, default = '../CC19/test/' , help = 'covid19 test dir path')

    # whether to use default settings for batch size and learning rates
    parser.add_argument('-d', '--default', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use default hyperparmeter settings (batch size and learning rates)')

    # general parameters for both non-FL and FL
    parser.add_argument('-p', '--project', type = str, default = 'femnist', help = 'project name, from femnist, celeba, shakespeare, covid19')
    parser.add_argument('--name', type = str, default = 'name', help = 'wandb run name')
    parser.add_argument('-seed', '--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--min_sample', type = int, default = 64, help = 'minimal amount of samples per client')
    parser.add_argument('-g_bs', '--global_bs', type = int, default = 64, help = 'batch size for global data loader')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = 1e-1, help = 'client learning rate')
    parser.add_argument('--global_epoch', type = int, default = 201, help = 'number of global aggregation rounds')
    parser.add_argument('--reuse_optim', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to reuse client optimizer, should be T for non-fl and F for FL')
    parser.add_argument('-c_op', '--client_optim', default = torch.optim.SGD, help = 'client optimizer')
                    
    # general parameters for FL
    parser.add_argument('-fl', '--switch_FL', type = str, default = 'FedAvg', help = 'FL algorithm, from FedAvg, FedAdam, FedAMS, FedProx, MOON, FedAwS, Ours')
    parser.add_argument('-c_bs', '--client_bs', type = int, default = 64, help = 'batch size for client data loader')
    parser.add_argument('-C', '--client_C', type = int, default = 8, help = 'number of participating clients in each aggregation round')
    parser.add_argument('-E', '--client_epoch', type = int, default = 1, help = 'number of client local training epochs')
    
    # for FedOpt and FedAMS
    parser.add_argument('-g_lr', '--global_lr', type = float, default = 1e-3, help = 'global learning rate')
    parser.add_argument('-g_op', '--global_optim', default = torch.optim.Adam, help = 'global optimizer')
    
    # for ours
    parser.add_argument('--base_agg', type = str, default = 'FedAvg', help = 'basic aggregation method for non-logit layers for our method')
    parser.add_argument('--agg_svc', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether aggregating support vectors or all class embeddings for our method')
    parser.add_argument('--spreadout', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether conduing spread-out regularization for our method')
    parser.add_argument('--class_C', type = float, default = 1.0, help = 'proportion of classes being aggregated for our method')
    parser.add_argument('-l_lr', '--logits_lr', type = float, default = 1e-2, help = 'global learning rate for logit layer for our method')
    parser.add_argument('-l_op', '--logits_optim', default = torch.optim.Adam, help = 'global optimizer for logit layer for our method')
    
    args = parser.parse_args()
    args.time = str(datetime.now())[5:-10]
    args.fed_agg = None
    args.MOON = False
    args.FedProx = False
    args.amsgrad = False
    
    return args

def get_clients_and_model(args: argparse.Namespace) -> tuple[list[object], list[object], torch.nn.Module]:
    """
    Determine dataset and model based on project name.

    Arguments:
        args (argparse.Namespace): parsed argument object.

    Returns:
        train_clients (list[Client]): list of training clients.
        test_clients (list[Client]): list of test/validation clients.
        model (torch.nn.Module): pytorch model for the specific task.
    """

    match args.project:
        case 'femnist':
            train_data_dict = get_data_dict_femnist(args.femnist_train_path, args.min_sample)
            test_data_dict  = get_data_dict_femnist(args.femnist_test_path , args.min_sample)
            model = CNN_femnist(args)

        case 'celeba':
            train_data_dict = get_data_dict_celeba(args.celeba_train_path, args.celeba_image_path, args.min_sample)
            test_data_dict  = get_data_dict_celeba(args.celeba_test_path , args.celeba_image_path, args.min_sample)
            model = CNN_celeba(args)

        case 'shakespeare':
            train_data_dict = get_data_dict_shakespeare(args.shakespeare_train_path, args.min_sample)
            test_data_dict  = get_data_dict_shakespeare(args.shakespeare_test_path , args.min_sample)
            model = LSTM_shakespeare(args)

        case 'covid19':
            train_data_dict = get_data_dict_covid19(args.covid19_train_path, args.min_sample)
            test_data_dict  = get_data_dict_covid19(args.covid19_test_path , args.min_sample)
            model = Resnet50_covid19(args)

        case _:
            raise Exception("wrong project:", args.project)
        
    # get client lists
    train_clients = get_clients(args, train_data_dict) ; del train_data_dict
    test_clients  = get_clients(args, test_data_dict ) ; del test_data_dict

    # some print
    print()
    print("number of train clients:", len(train_clients))
    print("number of test  clients:", len(test_clients ))
    print("length of train dataset:", sum([c.num_sample for c in train_clients]))
    print("length of test  dataset:", sum([c.num_sample for c in test_clients ]))

    return train_clients, test_clients, model

def default_setting(args: argparse.Namespace) -> None:
    """
    Set batch sizes and learning rates according to the choice of dataset and federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    assert(args.default)

    match args.project:
        case 'femnist':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
            args.client_lr  = 1e-1
            args.global_lr  = 1e-3
            args.logits_lr  = 1e-2

        case 'celeba':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 1e-3
            args.global_lr  = 1e-3 # all global learning rates are bad here
            args.logits_lr  = 1e-2

        case 'shakespeare':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
            args.client_lr  = 1
            args.global_lr  = 1e-2
            args.logits_lr  = 1e-1

        case 'covid19':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
                
        case _:
            raise Exception("wrong project:", args.project)
        
def switch_FL(args: argparse.Namespace) -> None:
    """
    Set aggregation strategy according to the choice of federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    match args.switch_FL:

        case 'FedAvg':
            args.fed_agg = 'FedAvg'

        case 'FedAdam':
            args.fed_agg = 'FedOpt'

        case 'FedAMS':
            args.fed_agg = 'FedOpt'
            args.amsgrad = True
    
        case 'FedProx':
            args.fed_agg = 'FedAvg'
            args.FedProx = True

        case 'MOON':
            args.fed_agg = 'FedAvg'
            args.MOON = True

        case 'FedAwS':
            args.fed_agg = 'FedAwS'

        case 'Ours':
            args.fed_agg = 'Ours'
            
        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)
    
def weighted_avg_params(params: list[dict[str, torch.Tensor]], weights: list[int] = None) -> dict[str, torch.Tensor]:
    """
    Compute weighted average of client models.

    Argument:
        params (list[dict[str, torch.Tensor]]): client model parameters. Each element in this list is the state_dict of a client model.
        weights (list[int]): weight per client. Each element in this list is the number of samples of a client.

    Returns:
        params_avg (dict[str], torch.Tensor): averaged global model parameters (state_dict), which can be loaded using global_model.load_state_dict.
    """

    if weights == None:
        weights = [1.0] * len(params)
        
    params_avg = copy.deepcopy(params[0])
    for key in params_avg.keys():
        params_avg[key] *= weights[0]
        for i in range(1, len(params)):
            params_avg[key] += params[i][key] * weights[i]
        params_avg[key] = torch.div(params_avg[key], sum(weights))
    return params_avg

def weighted_avg(values: any, weights: any) -> any:
    """
    Calculate weighted average of a vector of values.

    Arguments:
        values (any): values. Can be list, torch.Tensor, numpy.ndarray, etc.
        weights (any): weights. Can be list, torch.Tensor, numpy.ndarray, etc.

    Returns:
        any: weighted average value.
    """

    sum_values = 0
    for v, w in zip(values, weights):
        sum_values += v * w
    return sum_values / sum(weights)