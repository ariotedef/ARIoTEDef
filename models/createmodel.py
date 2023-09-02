
from models.keraslstm import PSDetector

from seq2seq.seq2seq_attention import Seq2seqAttention

from models.keraslstm import Seq2Seq


def init_psdetector(multistep_dataset, args):

    reconnaissance_detector = PSDetector(name="reconnaissance-detector", args=args)
    reconnaissance_detector.add_dataset(dataset=multistep_dataset['reconnaissance']) 
    print("reconnaissance_detector.dataset['train'][0].shape:",reconnaissance_detector.dataset['train'][0].shape)
    print("reconnaissance_detector.dataset['train'][1].shape:",reconnaissance_detector.dataset['train'][1].shape)

    reconnaissance_input_dim=reconnaissance_detector.dataset['train'][0].shape[1]
    print("reconnaissance_input_dim:",reconnaissance_input_dim)              
    if args.rec_model_path is None:
        reconnaissance_detector.def_model(reconnaissance_input_dim, output_dim=1, timesteps=args.timesteps)
    elif args.rec_model_path is not None:
        print("args.rec_model_path:", args.rec_model_path)
        model_path = args.rec_model_path
        reconnaissance_detector.load_model(model_path)
        


    infection_detector = PSDetector(name="infection-detector", args=args)
    infection_detector.add_dataset(dataset=multistep_dataset['infection']) 
    print("infection_detector.dataset['train'][0].shape:",infection_detector.dataset['train'][0].shape)
    print("infection_detector.dataset['train'][1].shape:",infection_detector.dataset['train'][1].shape)
 
    infection_input_dim=infection_detector.dataset['train'][0].shape[1]
    print("infection_input_dim:",infection_input_dim)               
    if args.inf_model_path is None:
        infection_detector.def_model(infection_input_dim, output_dim=1, timesteps=args.timesteps)
    elif args.inf_model_path is not None:
        print("args.inf_model_path:", args.inf_model_path)
        model_path = args.inf_model_path
        infection_detector.load_model(model_path)
        
    attack_detector = PSDetector(name="action-detector", args=args)
    attack_detector.add_dataset(dataset=multistep_dataset['attack']) 
    print("attack_detector.dataset['train'][0].shape:",attack_detector.dataset['train'][0].shape)
    print("attack_detector.dataset['train'][1].shape:",attack_detector.dataset['train'][1].shape)
   
    attack_input_dim=attack_detector.dataset['train'][0].shape[1]
    print("attack_input_dim:",attack_input_dim)               
    if args.att_model_path is None:
        attack_detector.def_model(attack_input_dim, output_dim=1, timesteps=args.timesteps)
    elif args.att_model_path is not None:
        print("args.att_model_path:", args.att_model_path)
        model_path = args.att_model_path
        attack_detector.load_model(model_path)
        
            
    return reconnaissance_detector, infection_detector, attack_detector


def init_seq2seq(multistep_dataset, args):
    
    infection_seq2seq = Seq2Seq(name='infection-seq2seq', args=args)
    infection_seq2seq.add_dataset(dataset=multistep_dataset['infection']) 
    print("infection_seq2seq.dataset['train'][0].shape:",infection_seq2seq.dataset['train'][0].shape)
    print("infection_seq2seq.dataset['train'][1].shape:",infection_seq2seq.dataset['train'][1].shape)
    
    
    if args.seq2seq_model_path is None:
        print("model")

    elif args.seq2seq_model_path is not None:
        print("args.seq2seq_model_path:", args.seq2seq_model_path)
        model_path = args.seq2seq_model_path
        infection_seq2seq.load_model(model_path)

    return infection_seq2seq