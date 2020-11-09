import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.preprocessing import normalize
import numpy as np
from torch import nn

import cv2
from backbone.model_irse import IR_50,IR_101,IR_152,IR_SE_50,IR_SE_101
from backbone.models_irse101 import resnet101
from backbone.iresnet import iresnet34,iresnet50,iresnet100
from torch.nn import functional as F
from function import *
import random
import shutil

class model_attack(nn.Module):
    def __init__(self,model_names,attack_methods):
        super(model_attack,self).__init__()
        all_model = []
        self.model_num = len(model_names)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for model_name in model_names:
            all_model.append(self.get_models(model_name).to(self.device))
        self.model = Ensemble(all_model)
        self.model.eval()
        self.si = self.di = self.ti = self.mi = False 
        self.u = 0
        if attack_methods.count('si') == 1:
            self.si = True
        if attack_methods.count('di') == 1:
            self.di = True
        if attack_methods.count('ti') == 1:
            self.ti = True
            kernel = gkern(5, 3).astype(np.float32)
            self.Gaussian = GaussianBlur(kernel)
        if attack_methods.count('mi') == 1:
            self.mi = True
            self.u = 1

    def forward(self,input_path,output_path,pair_path,ts,dataset,max_step,alpha,weights=None):
        if weights:
            assert(abs(sum(weights)-1)<=0.01)
            assert(len(weights) == self.model_num)
        else:
            weights = [1/self.model_num] * self.model_num
        os.makedirs(output_path,exist_ok=True)
        pairs = obtain_pair(pair_path)
        image_names = os.listdir(input_path)
        image_names = [i for i in image_names if i[-1] == 'g']
        image_names.sort()
        nums = len(image_names)

        # for num in range(0,5):
        for num in range(nums):
            img_path = os.path.join(input_path,image_names[num])
            img = Image.open(img_path)
            img = img2tensor(img).to(self.device)
            img_cp = img

            img_best = img_cp
            loss_best = -1
            if dataset == 'test':
                target_path = os.path.join(input_path,pairs[image_names[num]])
            elif dataset == 'val':
                target_path = os.path.join(input_path,pairs[image_names[num][:4]+'_adv.png'])
            else:
                print('The dataset is wrong! Please check!')

            target_img = Image.open(target_path)
            target_img = img2tensor(target_img).to(self.device)
            target_feature = self.model(target_img)
            last_grad_data = None
            for step in range(max_step):
                img.requires_grad = True
                loss = 0
                diverse_img = img
                if self.di:
                    diverse_img = input_diversity(img)
                if not self.si:
                    out_feature = self.model(diverse_img)
                    for m in range(len(out_feature)):
                        loss += torch.cosine_similarity(out_feature[m],
                        target_feature[m],dim = 1)*weights[m]
                if self.si:
                    for n in range(3):
                        img_sim = diverse_img / (2**n)
                        out_feature = self.model(img_sim)
                        for m in range(len(out_feature)):
                            loss += torch.cosine_similarity(out_feature[m],
                            target_feature[m],dim = 1)*weights[m]
                    loss /= 3
                if loss>loss_best:
                    loss_best = loss
                    img_best = img
                if loss>ts:
                    break
                self.model.zero_grad()
                loss.backward(retain_graph = True)
                data_grad = img.grad.data
                if self.ti:
                    data_grad = self.Gaussian(data_grad)
                grad_norm = torch.norm(data_grad,p=2,dim=2,keepdim=True)
                grad_norm = torch.norm(grad_norm,p=2,dim=3,keepdim=True)
                if last_grad_data is None:
                    sign_data_grad = data_grad / grad_norm
                else:
                    sign_data_grad = self.u*last_grad_data + data_grad/grad_norm
                last_grad_data = sign_data_grad
                img = img + alpha * sign_data_grad
                delta_img = img - img_cp
                delta_img = torch.clamp(delta_img,-20.4/128,20.4/128)
                img = img_cp + delta_img
                img = torch.clamp(img,-127.5/128,127.5/128)
                img = torch.from_numpy(img.detach().cpu().numpy()).cuda()

            fake = (np.transpose(img_best.squeeze(0).detach().cpu().numpy(),[1,2,0])*128+127.5)
            cv2.imwrite(os.path.join(output_path,image_names[num].split(".")[0]+"_adv.png"),fake[...,::-1],[int(cv2.IMWRITE_JPEG_QUALITY),96])
            print("The {}th image has been attacked! Loss(best) is {:.8f}, step is {:}".\
            format(num,loss_best.detach().cpu().numpy()[0].tolist(),step))
        print("Finished...")

    def get_models(self,model_name):
        if model_name=='irse101':
            config = HParams()
            model = resnet101(config)
            model.load_state_dict(torch.load("./backbone/insight-face-v3.pt"))
            model = torch.nn.DataParallel(model)
            model.eval()
            print('Load IRSE101 done!')
            return model
        if model_name == 'ir152':
            model = IR_152([112,112]).to("cuda")
            model.load_state_dict(torch.load("./backbone/Backbone_IR_152_Epoch_37.pth"))
            model.eval()
            print('Load IR152 done!')
            return model
        if model_name == 'ir100':
            model = iresnet100(pretrained=True)
            model.eval()
            print("Load IR100 done!")
            return model
        if model_name == 'ir50':
            model = iresnet50(pretrained=True)
            model.eval()
            print("Load IR50 done!")
            return model
        if model_name == 'ir34':
            model = iresnet34(pretrained=True)
            model.eval()
            print("Load IR34 done!")
            return model
        print('The model name is wrong! Please check!')
        exit()


if __name__ == '__main__':

    # model_names = ['irse101','ir34','ir50','ir100','ir152']
    model_names = ['irse101']
    attack_methods = 'sidi'
    model = model_attack(model_names,attack_methods)


    input_path = r'/opt/data/private/webunk/ori_dataset/test'
    output_path = r'/opt/data/private/webunk/output/test'
    pair_path = r'/opt/data/private/webunk/ori_dataset/test/pair.txt'
    # input_path = r'/opt/data/private/webunk/ori_dataset/val'
    # output_path = r'/opt/data/private/webunk/output/val/for_test'
    # pair_path = r'/opt/data/private/webunk/pair_val.txt'
    ts = 0.91
    dataset = 'test'
    max_step = 300
    alpha = 0.5
    # weights = [0.4,0.2,0.1,0.2,0.1]
    model(input_path,output_path,pair_path,ts,dataset,max_step,alpha,weights = None)

