import os
import subprocess
method_list= ['SGC','GCN','AS-GCN']
dataset = 'cora'
root_path = os.getcwd() + '/'
save_file = root_path + 'batch_run/' + 'ablation_study'

for method in method_list:
    if method != 'AS-GCN':
        command = "cd ../%s \npython train.py --modified --dataset %s --save_file %s" % (
            method, dataset, save_file)
    else:
        command = "cd ../%s \npython train.py --dataset %s --save_file %s" % (
            method, dataset, save_file)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = p.communicate()
    print(output)