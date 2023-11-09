
import os
import clip
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# device
device = 'cuda:0'

# paths
name_li_path = '../assets'

# hyperparametrs
time_wait = 10
obj_sm_freq_thre = 2
obj_sm_freq_wind = 10 
obj_pred_threshold = 0.1


### EgoHOS paths
in_od_stuff = '../EgoHOS/mmsegmentation/work_dirs/'
in_config_hands = os.path.join(in_od_stuff, 'seg_twohands_ccda/seg_twohands_ccda.py')
in_ckpt_hands = os.path.join(in_od_stuff, 'seg_twohands_ccda/best_mIoU_iter_56000.pth')

in_config_cb = os.path.join(in_od_stuff, 'twohands_to_cb_ccda/twohands_to_cb_ccda.py')
in_ckpt_cb = os.path.join(in_od_stuff, 'twohands_to_cb_ccda/best_mIoU_iter_76000.pth')


in_config_obj1 = os.path.join(in_od_stuff, 'twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py')
in_ckpt_obj1 = os.path.join(in_od_stuff, 'twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth')

# make tmp dirs
out_dir = '../assets/tmpAss/'
in_dir = '../assets/tmpAss/tmpImg/'


# load the clip model
clip_model_name = "ViT-L/14@336px"
clip_model, clip_preprocess = clip.load(clip_model_name, device=device)


# load blip2 model
b2md_name = 'Salesforce/blip2-flan-t5-xl' #'Salesforce/blip2-flan-t5-xl'
b2_processor = Blip2Processor.from_pretrained(b2md_name)
b2_model = Blip2ForConditionalGeneration.from_pretrained(b2md_name, torch_dtype=torch.float16)
b2_model.to(device)

