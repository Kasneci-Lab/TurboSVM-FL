import torch
import wandb
import tqdm

# self-defined functions
from models import model_train, model_eval
from utils import seed, Args, get_clients_and_model, default_setting

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def main_non_FL(args: object) -> None:
    """
    Main loop for centralized learning.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    # some print
    print("\nusing device:", device)
    
    # reproducibility
    seed(args.seed)

    # get client lists and model
    train_clients, test_clients, model = get_clients_and_model(args)

    # dataset and data loader 
    train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in train_clients])
    test_dataset  = torch.utils.data.ConcatDataset([c.dataset for c in test_clients ])
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = args.global_bs, shuffle = True)
    test_loader   = torch.utils.data.DataLoader(test_dataset , batch_size = args.global_bs, shuffle = False)
    
    # wandb init
    wandb.init(project = args.project, name = args.name, config = args.__dict__)

    # performance before training
    model.to(device)
    wandb_log = {}
    model_eval(model, train_loader, wandb_log, 'train/')
    model_eval(model, test_loader , wandb_log, 'test/' )
    wandb.log(wandb_log)

    # training loop
    print()
    for current_epoch in tqdm.tqdm(range(args.global_epoch)):
        # train for 1 epoch
        model_train(model, train_loader, 1)
        
        # train and validation metrics
        wandb_log = {}
        model_eval(model, train_loader, wandb_log, 'train/')
        model_eval(model, test_loader , wandb_log, 'test/' )
        wandb.log(wandb_log)
    
    # wandb.finish()
    
# main function call
if __name__ == '__main__':
    args = Args()

    # use default settings
    if args.default:
        default_setting(args)
    
    # reuse_optim should be True in case of non-fl
    args.reuse_optim  = True
    
    # wandb run name
    args.name  = 'seed ' + str(args.seed) + ' '
    args.name += 'non-FL: '
    args.name += str(args.client_optim).split('.')[-1][:-2]
    args.name += ' ' + str(args.client_lr)
    
    main_non_FL(args)