'''This is a sample inference script to demonstrate how to run inference on the model for a single .wav file

Authors
* Jia Qi Yip 2024
'''
import torch
import json
from model.mossformer2 import Mossformer2Wrapper



def ori():
    model_configs = ["mossformer2-librimix-2spk", "mossformer2-wsj0mix-3spk", "mossformer2-whamr-2spk"]

    for mc in model_configs:
        model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/{mc}')
        model.inference(f'./test_samples/{mc}/item0_mix.wav',f'./test_samples/{mc}/model_output')

def test_libritts_2spk():
    
    with open('/Users/donkeyddddd/Documents/Rx_projects/git_projects/MossFormer2/MossFormer2_standalone/assets/mossformer2-librimix-2spk/config.json', 'r') as f:
        test_config = json.load(f)
    

    model = Mossformer2Wrapper(test_config)
    model.loadPretrained()
    model.inference(
        f'/Users/donkeyddddd/Documents/Rx_projects/git_projects/MossFormer2/MossFormer2_standalone/test_samples/mossformer2-librimix-2spk/item0_mix.wav',
        f'/Users/donkeyddddd/Documents/Rx_projects/git_projects/MossFormer2/MossFormer2_standalone/test_samples/mossformer2-librimix-2spk/model_output'
        )


    xxx = 1


if __name__=="__main__":
    test_libritts_2spk()

    xxx = 1