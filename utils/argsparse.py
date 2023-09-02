import argparse


def get_args(jupyter_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--permute_truncated', required=False, action='store_true', 
                        help="Bool for activating permutation invariance")
    parser.add_argument('--use_prob_embedding', required=False, action='store_true', 
                        help="Bool for using original probability based embedding proposed in the original paper")
    parser.add_argument('--sequence_length', required=False, type=int, default=10, 
                        help="Length of truncated subsequences used in the seq2seq training")
    parser.add_argument('--roundvalue_d', required=False, type=int, default=1, 
                        help="'round value' hyperparameter used for probability embedding, if activated")
    parser.add_argument('--ps_epochs', required=False, type=int, default=50, 
                        help="number of training epochs for per-step detectors")
    parser.add_argument('--relabel_rounds', required=False, type=int, default=1, 
                        help="Number of relabel rounds")
    parser.add_argument('--patience', required=False, type=int, default=None,
                        help="Patience for early stopping. Any value activates early stopping.")
    
    parser.add_argument('--seed', required=False, type=int, default=0,
                        help="random seed.")  
    parser.add_argument('--eps', required=False, type=float, default=1.0,
                        help="adversarial evasion attack parameter: epsilon.")        
    parser.add_argument('--eps_step', required=False, type=float, default=0.5,
                        help="adversarial evasion attack parameter: epsilon step.")   
    parser.add_argument('--max_iter', required=False, type=int, default=20,
                        help="adversarial evasion attack parameter: iteration number.")   
    parser.add_argument('--save_path',type=str, default='/home/huan1932/ARIoTEDef/result',help='Output path for saving results')
    parser.add_argument('--batchsize', required=False, type=int, default=32,
                        help="batch size.")       
    parser.add_argument('--timesteps', required=False, type=int, default=1,
                        help="time steps of LSTM.")      
    parser.add_argument('--seq2seq_epochs', required=False, type=int, default=50,
                        help="seq2seq epochs")      
    parser.add_argument('--seq2seq_batchsize', required=False, type=int, default=32,
                        help="seq2seq batchsize")            
    parser.add_argument('--retrainset_mode',type=str, default='adv',help='Output path for saving results')
    
    parser.add_argument('--rec_model_path', type=str, default= None, help='save path of reconnaissance detector')
    parser.add_argument('--inf_model_path', type=str, default= None, help='save path of infection detector')
    parser.add_argument('--att_model_path', type=str, default= None, help='save path of attack detector')

    parser.add_argument('--lr', required=False, type=float, default=0.001,
                        help="initial learning rate of per step detector training")     
    
    
    parser.add_argument('--stdtrain_pedetector', action='store_true', help='default is False')
    
    parser.add_argument('--seq2seq_model_path', type=str, default= None, help='save path of seq2seq translation analyzer')
    parser.add_argument('--stdtrain_seq2seq', action='store_true', help='default is False')
    
    parser.add_argument('--strategy', type=str, default= 'strategy1', help='retraining strategy')
    parser.add_argument('--seq2seq_threshold', type=float, default=0.01, help="seq2seq threshold")              
    
    parser.add_argument('--retrain_seq2seq', action='store_true', help='default is False')
    
    parser.add_argument('--attack',type=str, default='pgd',help='type of evasion attack')
    parser.add_argument('--targeted', action='store_true', help='default is False, targeted attack')
    parser.add_argument('--retrain_testset_mode', type=str, default= 'adv', help='retraining test set composition')
    
    parser.add_argument('--advset_mode', type=str, default= 'advset1', help='retraining test set composition')
    
    if jupyter_args is not None:
        args = parser.parse_args(jupyter_args)
    else: 
        args = parser.parse_args()
    return args

