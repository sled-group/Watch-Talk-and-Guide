from queue import Queue
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
import os
import sys
import json
import clip
import torch
import requests
import numpy as np
from PIL import Image
from skimage.io import imsave
from typing import Optional
from transformers import Blip2Processor, Blip2ForConditionalGeneration


from open import *
from config import *

sys.path.insert(0, '../EgoHOS/mmsegmentation')
from mmseg.apis import inference_segmentor, init_segmentor


# get the step/ASR if in_i falls under its corresponding timeframe
def parse_text_time(in_i, in_what, vid_start, frame_ratio):
	for xi in in_what:
		if (in_i >= (xi[0] - vid_start)*frame_ratio) and \
		   (in_i < (xi[1] - vid_start)*frame_ratio):
			return xi[2:]
	return []


def get_prompt(prompt_recipe, chat_history, fill_time, fill_user_asr=None, \
			   fill_inst_asr=None, vb_generated_text=None, predicted_objects=None):
	# prompt recipe
	prompt_out = "You are an instructor guiding a user to complete the task of " + prompt_recipe
	
	# prompt chat history
	if len(chat_history) > 0:
		prompt_out += "Chat History:\n"
		for chi in chat_history:
			prompt_out += "Time: " + str(chi[0]) + ", " + chi[1] + ": " + chi[2] + "\n"
		prompt_out += '\n'
	
	# prompts observations
	prompt_obs = "It has been " + str(fill_time) + " seconds into the recipe.\n"
	# add BLIP
	if (vb_generated_text is not None) and (vb_generated_text != ""):
		prompt_obs += "Scene description: " + vb_generated_text + "\n"
	# add objects
	if (predicted_objects is not None) and (len(predicted_objects)>0):
		for poik, poiv in predicted_objects.items():
			prompt_obs += "The user is interacting with " + poik + "\n"
			prompt_obs += poiv + "\n"
	if fill_user_asr is not None:
		prompt_obs += "User said " + fill_user_asr + "\n"
	prompt_out += prompt_obs + "\n"
		
	
	# prompts questions
	prompt_que = "Answer the following questions:\n"
	if fill_user_asr is not None:
		prompt_que += "1. What is his dialog intention? Choose among Question, Answer, Confirmation, Hesitation, Self Description, and Other\n"
	prompt_que += "2. Which step is the user at? For example Step 3\n"
	prompt_que += "3. Should you say anything? Yes or no\n"
	prompt_que += "3.1. If yes, you would say:\n"
	prompt_que += "3.2. If yes, choose your dialog intention among Instruction, Confirmation, Question, Answer, or Other\n"
	prompt_que += "3.3. If your dialog intention is Instruction, is it about current step, next step, details, or mistake correction\n"
	prompt_que += "4. Did the user make a mistake? Yes or No? If yes, choose among wrong object, wrong state, wrong action\n"
	prompt_out += prompt_que

	prompt_out += "Answer:\n"
	
	return prompt_out, prompt_obs

############################################### BLIP2 ###############################################
############################################### BLIP2 ###############################################
def call_BLIP2(fra):
	inputs = b2_processor(fra, return_tensors="pt").to(device, torch.float16)
	generated_ids = b2_model.generate(**inputs)
	generated_text = (b2_processor.decode(generated_ids[0], skip_special_tokens=True))
	return generated_text

############################################### OBJ DETECION ###############################################
############################################### OBJ DETECION ###############################################

def mask_to_box(mask):
	y, x = np.where(mask != 0)
	return [np.min(x), np.min(y), np.max(x), np.max(y)]

def get_clip_probs(img: np.ndarray, text: list[str]):
	image = clip_preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
	text = clip.tokenize(text).to(device)
	with torch.no_grad():
		logits_per_image, logits_per_text = clip_model(image, text)
		probs = logits_per_image.softmax(dim=-1).cpu().numpy()
	return probs

def get_obj_mask(img, egoModels):
	# save og image
	img_nm = "tmp.png"
	img = Image.fromarray(img)
	img = img.save(in_dir+img_nm)

	# get models
	[model_obj1, model_cb, model_hand] = egoModels
	
	# detect two hands
	seg_result_hand = inference_segmentor(model_hand, in_dir+img_nm)[0]
	imsave(out_dir + 'pred_twohands/'+img_nm, seg_result_hand.astype(np.uint8))

	# detect contact boundary
	seg_result_cb = inference_segmentor(model_cb, in_dir+img_nm)[0]
	imsave(out_dir + 'pred_cb/'+img_nm, seg_result_cb.astype(np.uint8))
	
	# detect objects 1st order
	seg_result_obj1 = inference_segmentor(model_obj1, in_dir+img_nm)[0]
	
	# remove files
	os.remove(in_dir+img_nm)
	os.remove(out_dir + 'pred_twohands/'+img_nm)
	os.remove(out_dir + 'pred_cb/'+img_nm)
	
	return (seg_result_obj1==1), (seg_result_obj1==2), (seg_result_obj1==3)

def predict_object(
	img: np.ndarray, 
	obj_mask: np.ndarray, 
	object_candidates: list[str], 
	pinwheel_dic_clean:dict,
) -> tuple[Optional[str], dict[str, str]]:
	x_min, y_min, x_max, y_max = mask_to_box(obj_mask)
	if x_max <= x_min or y_max <= y_min:
		return None
	obj_img = img[y_min:y_max, x_min:x_max]
	probs = get_clip_probs(obj_img, object_candidates)
	obj_prob = np.max(probs)
	if obj_prob > obj_pred_threshold:
		obj_name = object_candidates[np.argmax(probs)]
		
		if obj_name[11:] not in pinwheel_dic_clean:
			return obj_name, None

		# get stat estimates
		state_prompts_candidates = []
		for val in pinwheel_dic_clean[obj_name[11:]]:
			state_prompt_i = "The " + obj_name[11:] + " is " + val
			state_prompts_candidates.append(state_prompt_i)
		### TODO:
		probs_states = get_clip_probs(obj_img, state_prompts_candidates)        
		state_prob = np.max(probs_states)
		state_name = state_prompts_candidates[np.argmax(probs_states)]
		
		return obj_name, state_name
	
	return None, None



def get_obj_states(fra, obj_history, obj_history_state_dic, egoModels, objs_pinwheel_out, pinwheel_dic_clean):
	# get object mask
	obj_LH_mask, obj_RH_mask, obj_BH_mask = get_obj_mask(fra, egoModels)
	has_obj_BH = (obj_BH_mask.sum() > 1000)
	has_obj_LH = (obj_LH_mask.sum() > 1000) and not has_obj_BH
	has_obj_RH = (obj_RH_mask.sum() > 1000) and not has_obj_BH
	
	# predict objects
	predicted_objects = {}
	if has_obj_LH:
		obj_pred, state_pred = predict_object(fra, obj_LH_mask, objs_pinwheel_out, pinwheel_dic_clean)
		if obj_pred is not None:
			predicted_objects[obj_pred[11:]] = state_pred

	if has_obj_RH:
		obj_pred, state_pred = predict_object(fra, obj_RH_mask, objs_pinwheel_out, pinwheel_dic_clean)
		if obj_pred is not None:
			predicted_objects[obj_pred[11:]] = state_pred

	if has_obj_BH:
		obj_pred , state_pred= predict_object(fra, obj_BH_mask, objs_pinwheel_out, pinwheel_dic_clean)
		if obj_pred is not None:
			predicted_objects[obj_pred[11:]] = state_pred
	
	### smooth
	# update obj_history and obj_history_state_dic
	if len(obj_history) >= obj_sm_freq_wind:
		obj_history.pop(0)
	obj_history.append(list(predicted_objects))
	for pi, pv in predicted_objects.items():
		obj_history_state_dic[pi] = pv

	
	# count obj frequency in history
	obj_sm_count = {}
	for ohi in obj_history:
		for ohii in ohi:
			if ohii not in obj_sm_count.keys():
				obj_sm_count[ohii] = 1
			else:
				obj_sm_count[ohii] += 1
	
	# get the most freq > thre objs
	predicted_objects_sm = {}
	for ki, vi in obj_sm_count.items():
		if vi > obj_sm_freq_thre:
			predicted_objects_sm[ki] = obj_history_state_dic[ki]
	
	return predicted_objects_sm, obj_history, obj_history_state_dic


def prep_obj(recipe_type):
	### read in object list and states
	pinwheel_dic = json.load(open(os.path.join(name_li_path, recipe_type.lower()+'DictManual.txt')))
	objs_pinwheel = list(pinwheel_dic.keys())

	### clean up, and assemble prompts
	pinwheel_dic_clean = {}
	for oi in objs_pinwheel:
		oi_li = oi.split(" or ")
		for oii in oi_li:
			pinwheel_dic_clean[oii] = pinwheel_dic[oi]
	objs_pinwheel_clean = list(pinwheel_dic_clean.keys())
	objs_pinwheel_out = []
	for oi in objs_pinwheel_clean:
		objs_pinwheel_out.append("a photo of " + oi)


	### load EgoHOS
	os.makedirs(os.path.join(out_dir,'pred_twohands'), exist_ok = True)
	os.makedirs(os.path.join(out_dir,'pred_cb'), exist_ok = True)
	os.makedirs(os.path.join(out_dir,'pred_obj1'), exist_ok = True)
	model_hand = init_segmentor(in_config_hands, in_ckpt_hands, device=device)
	model_cb = init_segmentor(in_config_cb, in_ckpt_cb, device=device)
	model_obj1 = init_segmentor(in_config_obj1, in_ckpt_obj1, device=device)

	return model_obj1, model_cb, model_hand, objs_pinwheel_out, pinwheel_dic_clean





