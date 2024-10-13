import json

import matplotlib.pyplot as plt
import numpy as np

EPOCH_NUM = 20


def get_model_info(filepath):
    data_list = []
    with open(filepath) as json_file:
        for i, line in enumerate(json_file):
            if i == EPOCH_NUM:
                break
            dt = json.loads(line)
            if 'training time' in dt:
                continue
            data_list.append(dt)

    return data_list


def show(models, params, epochs, filename):
    plt.figure(figsize=(8, 12))
    data = dict()

    for model_name, model in models.items():
        data[model_name] = dict()
        for prm_name in params:
            data[model_name][prm_name] = [item[prm_name] for item in model]

    for i, prm_name in enumerate(params.keys()):
        plt.subplot(len(list(params.keys())), 1, i + 1)
        for model_name in data:
            plt.plot(epochs, data[model_name][prm_name], label=model_name, marker='o')
        plt.title(params[prm_name])
        plt.xlabel('Эпохи')
        xprm = prm_name.split('_')
        plt.ylabel(''.join(xprm))
        plt.xticks(np.arange(0, max(epochs) + 1, 5))
        if i == 0:
            plt.yscale('log')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{filename}.png')


# 'E-KAN(sum)': get_model_info('output/mnist/fc_kan/fc_kan__mnist__dog-bs__sum__full_0.json'),
models0 = {
    'MLP': get_model_info('output/mnist/mlp/mlp__mnist__full_0.json'),
    'BSRBF': get_model_info('output/mnist/bsrbf_kan/bsrbf_kan__mnist__full_0.json'),
    'E-KAN(SUM)': get_model_info('output/mnist/fc_kan/fc_kan__mnist__dog-bs__sum__full_0.json'),
    'Faster-KAN': get_model_info('output/mnist/faster_kan/faster_kan__mnist__full_0.json'),
    'Gottlieb-KAN': get_model_info('output/mnist/gottlieb_kan/gottlieb_kan__mnist__full_0.json'),
    'E-KAN(SMN)': get_model_info('output/mnist/fc_kan/fc_kan__mnist__dog-bs__quadratic__full_0.json'),
}

params_dict1 = {"val_loss": 'Функция потерь на тестовом наборе данных', "f1_macro": 'Метрика F1 score'}
params_dict2 = {"train_loss": 'Функция потерь на обучающем наборе данных', "pre_macro": 'Точность (Precision)',
                "re_macro": 'Чувствительность (Recall)'
                }
show(models0, params_dict1, list(range(1, EPOCH_NUM + 1)), 'e-kan-20-g')
