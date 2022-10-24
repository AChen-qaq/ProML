import subprocess
import argparse

def run_all_FewNERD():
#     script = 'python3 train_demo.py  --mode {mode} \
# --lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --name RUN_ALL_PromptNCAJSMixedFixed_FewNERD_{exp_type}_seed={seed}_mix-rate={mix_rate} --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_PromptNCAJSMixedFixed_FewNERD_{exp_type}_seed={seed}_mix-rate={mix_rate}'

#     script = 'python3 train_demo.py  --mode {mode} \
# --lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --name RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_FewNERD_{exp_type}_seed={seed}_mix-rate={mix_rate} --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_FewNERD_{exp_type}_seed={seed}_mix-rate={mix_rate}'

    script = 'python3 train_demo.py  --mode {mode} \
--lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --use_sampled_data --name RUN_ALL_ProML_ablation_FewNERD_{exp_type}_seed={seed}_mix-rate={mix_rate} --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_ProML_FewNERD_{exp_type}_seed={seed}_mix-rate={mix_rate}'


    scripts_list = []
    for seed in range(5):
        for mix_rate in [0.3, 0.5, 0.7]:
            intra5way1shot = script.format(mode='intra', bsz=4, trainN=5, trainK=1, testN=5, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, exp_type='INTRA5way1shot')
            intra5way5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, exp_type='INTRA5way5shot')
            inter5way1shot = script.format(mode='inter', bsz=4, trainN=5, trainK=1, testN=5, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, exp_type='INTER5way1shot')
            inter5way5shot = script.format(mode='inter', bsz=1, trainN=5, trainK=5, testN=5, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, exp_type='INTER5way5shot')

            
            
            for exec_script in [intra5way1shot, intra5way5shot, inter5way1shot, inter5way5shot]:
                scripts_list.append(exec_script)


            
    return scripts_list

def run_all_OntoNotes():
    script = 'python3 train_demo.py  --mode {mode} \
--lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --name RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate} --use-conll {label_set} --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate}'

#     script = 'python3 train_demo.py  --mode {mode} \
# --lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --name RUN_ALL_PromptNCAJSMixedFixed_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate} --use-conll {label_set} --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_PromptNCAJSMixedFixed_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate}'

    scripts_list = []
    for seed in range(5):
        for mix_rate in [0.3, 0.5, 0.7]:
            setA1shot = script.format(mode='intra', bsz=4, trainN=5, trainK=1, testN=5, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, exp_type='TagExtensionA1shot', label_set='A')
            setA5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, exp_type='TagExtensionA5shot', label_set='A')

            setB1shot = script.format(mode='intra', bsz=4, trainN=5, trainK=1, testN=5, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, exp_type='TagExtensionB1shot', label_set='B')
            setB5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, exp_type='TagExtensionB5shot', label_set='B')

            setC1shot = script.format(mode='intra', bsz=4, trainN=5, trainK=1, testN=5, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, exp_type='TagExtensionC1shot', label_set='C')
            setC5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, exp_type='TagExtensionC5shot', label_set='C')



            for exec_script in [setA1shot, setA5shot, setB1shot, setB5shot, setC1shot, setC5shot]:
                scripts_list.append(exec_script)
    
    return scripts_list
                


def run_all_DomainTransfer():
#     script = 'python3 train_demo.py  --mode {mode} \
# --lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --name RUN_ALL_PromptNCAJSMixedFixed_OntoNotes_DomainTransferPretrained_seed={seed}_mix-rate={mix_rate} --use-ontonotes --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_PromptNCAJSMixedFixed_OntoNotes_DomainTransferPretrained_seed={seed}_mix-rate={mix_rate}'

    script = 'python3 train_demo.py  --mode {mode} \
--lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk {testK} --name RUN_ALL_cased_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_DomainTransferPretrained_{exp_type}_seed={seed}_mix-rate={mix_rate} --use-ontonotes --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/RUN_ALL_cased_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_DomainTransferPretrained_{exp_type}_seed={seed}_mix-rate={mix_rate}'
    scripts_list = []

    for seed in range(1):
        for mix_rate in [0.3]:
            ontonotes1shot = script.format(mode='intra', bsz=4, trainN=5, trainK=1, testN=5, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, exp_type='1shot')
            ontonotes5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, exp_type='5shot')

            for exec_script in [ontonotes1shot, ontonotes5shot]:
                scripts_list.append(exec_script)
    return scripts_list


def OntoNotesEval():
    script = 'python3 train_demo.py  --mode {mode} \
--lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length {mx_len} --model PromptNCA --tau 0.05 --seed {seed} --mix-rate {mix_rate} --topk 1 --name RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate} --only_test --full-test --load_ckpt checkpoint/RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate}.pth.tar --use-conll {label_set} --proj-dim 128 --with-dropout --trainK {trainK} > outputs/run_all_logs/TagExtensionEval/RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_{exp_type}_seed={seed}_mix-rate={mix_rate}_FULLTESTEVAL'
    scripts_list = []
    for data_split, totalN in zip(['A', 'B', 'C'], [6, 6, 6]):
        for seed in range(1):
            for mix_rate in [0.3]:
                ontonotes1shot = script.format(mode='intra', bsz=1, trainN=5, trainK=1, testN=5, totalN=totalN, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, eval_mix_rate=mix_rate, exp_type='TagExtension'+data_split+'1shot', label_set=data_split)
                ontonotes5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, totalN=totalN, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, eval_mix_rate=mix_rate, exp_type='TagExtension'+data_split+'5shot', label_set=data_split)

                for exec_script in [ontonotes1shot, ontonotes5shot]:
                    scripts_list.append(exec_script)
    return scripts_list

def DomainTransferEval():
    script = 'python3 train_demo.py  --mode {mode} \
--lr 3e-5 --batch_size {bsz} --trainN {trainN} --N {testN} --K {testK} --Q {testQ} \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length {mx_len} --model PromptNCA --eval-with-finetune --tau 0.05 --seed {seed} --mix-rate {mix_rate} --totalN {totalN} --eval-mix-rate {eval_mix_rate} --topk 1 --name RUN_ALL_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_DomainTransferPretrained_{exp_type}_seed={seed}_mix-rate={mix_rate} --proj-dim 128 --with-dropout --trainK {trainK} --only_test --full-test {arg_dataset} --load_ckpt=checkpoint/RUN_ALL_cased_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_DomainTransferPretrained_{exp_type}_seed={seed}_mix-rate={mix_rate}.pth.tar > outputs/run_all_logs/DomainTransferEval/RUN_ALL_cased_PromptNCAJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_DomainTransferPretrained_{exp_type}_seed={seed}_mix-rate={mix_rate}_eval-mix-rate={eval_mix_rate}_FULLTESTEVAL_{targetDomain}'
    scripts_list = []
    for targetDomain, arg_dataset, totalN in zip(['CoNLL', 'WNUT', 'I2B2', 'GUM'], ['--use-conll2003', '--use-wnut', '--use-i2b2', '--use-gum'], [4, 6, 19, 11]):
        for seed in range(1):
            for mix_rate in [0.3]:
                for eval_mix_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:

                    ontonotes1shot = script.format(mode='intra', bsz=1, trainN=5, trainK=1, testN=5, totalN=totalN, testK=1, testQ=1, mx_len=64, seed=seed, mix_rate=mix_rate, eval_mix_rate=eval_mix_rate, exp_type='1shot', targetDomain=targetDomain, arg_dataset=arg_dataset)
                    ontonotes5shot = script.format(mode='intra', bsz=1, trainN=5, trainK=5, testN=5, totalN=totalN, testK=5, testQ=5, mx_len=32, seed=seed, mix_rate=mix_rate, eval_mix_rate=eval_mix_rate, exp_type='5shot', targetDomain=targetDomain, arg_dataset=arg_dataset)

                    for exec_script in [ontonotes1shot, ontonotes5shot]:
                        scripts_list.append(exec_script)
    return scripts_list


def compute_stat_FewNERD():
    scripts = run_all_FewNERD()
    from tqdm import tqdm
    import re
    pbar = tqdm(scripts, total=len(scripts))
    stat = {}
    for script in pbar:
        path = script.split(' > ')[-1]
        try:
            with open(path) as f:
                logs = f.read()
            ret = re.search('RESULT.*f1:.*\n', logs)
            # if ret is None:
            #     continue
            f1 = 0 if ret is None else float(ret.group().split(':')[-1])
            exp_name = re.search('RUN_ALL_.*$', path).group()
            exp_name = re.sub('_seed=.*_', '_', exp_name)
            if exp_name not in stat:
                stat[exp_name] = []
            stat[exp_name].append(f1)
        except:
            pass
    import numpy as np
    for exp_name, ret in stat.items():
        print(exp_name, round(np.mean(ret) * 100, 2), round(np.std(ret) * 100, 2))

def compute_stat_OntoNotes():
    scripts = OntoNotesEval()
    from tqdm import tqdm
    import re
    pbar = tqdm(scripts, total=len(scripts))
    stat = {}
    for script in pbar:
        path = script.split(' > ')[-1]
        print(path)
        with open(path) as f:
            logs = f.read()
        ret = re.findall('\[EVAL\].*\n', logs)
        assert ret is not None
        # print(logs)
        f1 = [float(x.split(':')[-1]) for x in ret]
        tmp = [f1[0]]
        for i in range(1, len(f1)):
            tmp.append(f1[i] * (i+1) - f1[i - 1] * i)
        f1 = tmp
        # print(f1)
        # f1 = 0 if ret is None else float(ret.group().split(':')[-1])
        exp_name = re.search('RUN_ALL_.*$', path).group()
        exp_name = re.sub('_seed=0_', '_', exp_name)
        if exp_name not in stat:
            stat[exp_name] = []
        # if f1 != 0:
        stat[exp_name].append(f1)

    def print_info(stat, pattern_list, suffix):
        import numpy as np
        for pattern in pattern_list:
            for exp_name, ret in stat.items():
                # print(exp_name)
                if re.search(pattern, exp_name) is not None:
                    # print(exp_name)
                    print('|{mean}({std})'.format(mean=round(np.mean(ret) * 100, 2), std=round(np.std(ret) * 100, 2)), end='')
        
        print('|')

    print('|OURS', end='')
    print_info(stat, ['TagExtensionA1shot', 'TagExtensionB1shot', 'TagExtensionC1shot'], '')
    print('|OURS', end='')
    print_info(stat, ['TagExtensionA5shot', 'TagExtensionB5shot', 'TagExtensionC5shot'], '')
def compute_stat_DomainTransfer():
    scripts = DomainTransferEval()
    from tqdm import tqdm
    import re
    pbar = tqdm(scripts, total=len(scripts))
    stat = {}
    for script in pbar:
        path = script.split(' > ')[-1]
        with open(path) as f:
            logs = f.read()
        ret = re.findall('\[EVAL\].*\n', logs)
        assert ret is not None
        # print(ret)
        f1 = [float(x.split(':')[-1]) for x in ret]
        tmp = [f1[0]]
        for i in range(1, 10):
            tmp.append(f1[i] * (i+1) - f1[i - 1] * i)
        f1 = tmp
        # print(f1)
        # f1 = 0 if ret is None else float(ret.group().split(':')[-1])
        exp_name = re.search('RUN_ALL_.*$', path).group()
        exp_name = re.sub('_seed=0_', '_', exp_name)
        if exp_name not in stat:
            stat[exp_name] = []
        # if f1 != 0:
        stat[exp_name].append(f1)

    def print_info(stat, pattern_list, suffix):
        import numpy as np
        for pattern in pattern_list:
            for exp_name, ret in stat.items():
                # print(exp_name)
                if re.search(pattern, exp_name) is not None:
                    # print(exp_name)
                    print('|{mean}({std})'.format(mean=round(np.mean(ret) * 100, 2), std=round(np.std(ret) * 100, 2)), end='')
        
        print('|')
    for eval_mix_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
        suffix = 'eval-mix-rate={}'.format(eval_mix_rate)
        print('|OURS(eval-mix-rate={})'.format(eval_mix_rate), end='')
        print_info(stat, ['1shot.*'+suffix+'.*CoNLL', '5shot.*'+suffix+'.*CoNLL', '1shot.*'+suffix+'.*WNUT', '5shot.*'+suffix+'.*WNUT', '1shot.*'+suffix+'.*I2B2', '5shot.*'+suffix+'.*I2B2', '1shot.*'+suffix+'.*GUM', '5shot.*'+suffix+'.*GUM'], suffix)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    opt = parser.parse_args()
    # print(opt.start, opt.end)
    scripts = run_all_FewNERD() + run_all_OntoNotes() + run_all_DomainTransfer()
    scripts = DomainTransferEval()
    scripts = run_all_DomainTransfer()
    print(len(scripts))
    from tqdm import tqdm
    pbar = tqdm(scripts[opt.start: opt.end], total=len(scripts[opt.start: opt.end]))
    for script in pbar:
        script = 'CUDA_VISIBLE_DEVICES={device} {command}'.format(device=opt.device, command=script)
        print(script)
        subprocess.call(script, shell=True)

    compute_stat_FewNERD()
    compute_stat_OntoNotes()
    compute_stat_DomainTransfer()



