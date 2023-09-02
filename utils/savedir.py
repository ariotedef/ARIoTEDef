from utils.runid import GetRunID     
import os
import datetime


def set_exp_result_dir(args):
    save_path = f'{args.save_path}/seed-{args.seed}'

    cur=datetime.datetime.utcnow()
    date = f'{cur.year:04d}{cur.month:02d}{cur.day:02d}'
    print("date:",date)
    exp_result_dir=f'{save_path}/{date}/{args.advset_mode}/{args.sequence_length}/{args.strategy}/{args.seq2seq_threshold}'    

    # add run id for exp_result_dir
    cur_run_id = GetRunID(exp_result_dir)
    exp_result_dir = os.path.join(exp_result_dir, f'{cur_run_id:05d}')    

    return exp_result_dir