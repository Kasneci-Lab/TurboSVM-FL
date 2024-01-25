import torch
import wandb

# self-defined functions
from server import federated_learning
from utils import seed, Args, get_clients_and_model, switch_FL, default_setting

def main_FL(args: object):
    """
    Helper function for federated learning. It calls federated_learning in server.py in runtime.

    Arguments:
        args (argparse.Namespace): parsed argument object. 
    """

    # some print
    print("\nusing device:", 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # reproducibility
    seed(args.seed)
        
    # get client lists and model
    train_clients, test_clients, global_model = get_clients_and_model(args)

    # wandb init
    wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
    
    # federated learning
    federated_learning(args, train_clients, test_clients, global_model)
    
# main function call
if __name__ == '__main__':
    args = Args()

    # switch FL algorithms
    switch_FL(args)

    # use default settings
    if args.default:
        default_setting(args)
    
    # wandb run name
    args.name  = 'seed ' + str(args.seed) + ' '
    args.name += args.switch_FL + ': C ' + str(args.client_C) + ', E ' + str(args.client_epoch) + ', '
    match args.switch_FL:
        case 'FedAvg' | 'FedProx' | 'MOON':
            args.name += str(args.client_optim).split('.')[-1][:-2] + ' ' + str(args.client_lr)
        case 'FedAdam' | 'FedAMS':
            args.name += str(args.global_optim).split('.')[-1][:-2] + ' ' + str(args.global_lr)
        case 'FedAwS' | 'Ours':
            args.name += str(args.logits_optim).split('.')[-1][:-2] + ' ' + str(args.logits_lr)
        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)
        
    main_FL(args)