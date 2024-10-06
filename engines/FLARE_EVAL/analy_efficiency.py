import json
import os
import numpy as np

if __name__ == '__main__':
    root_path = '/ChenQiang/DataSet/Medical/MICCAI24_FLARE/validation_result/Efficiency/special_case/results/tju_vil_pioneers/'
    #json_file_name = "amos_7894_0000.json" # total
    json_file_name = "amos_8082_0000.json" # total
    json_path = os.path.join(root_path, json_file_name)
    with open(json_path, 'r') as f:
        result = json.loads(f.read())
    print('Avrage GPU memory:{:.2f} MB'.format(np.array(result['gpu_memory']).astype(float).mean()))
    print('Max GPU memory:{:.2f} MB'.format(float(max(result['gpu_memory']))))
    x = np.arange(0, len(result['gpu_memory'])/10, 0.1)
    y = np.array(result['gpu_memory']).astype(float)
    print('Total GPU memory:{:.2f}MB'.format(
        np.trapz(y, x, dx=0.1)
    ))
    print('Time:{:.2f}s'.format(float(result['time'])))
    # print('CPU memory:{:.2f}'.format(float(max(result['gpu_memory']))) / 1000)
    #print(result)

