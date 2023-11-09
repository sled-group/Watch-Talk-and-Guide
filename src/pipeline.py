import argparse
from util import *

def genAndCall(prompt_recipe, chat_history, timei, fill_user_asr=None, fill_inst_asr=None, \
									vb_generated_text=None, predicted_objects=None):
	prompt_i, prompt_obi = get_prompt(prompt_recipe, chat_history, timei, fill_user_asr=fill_user_asr, \
							fill_inst_asr=fill_inst_asr, vb_generated_text=vb_generated_text, \
							predicted_objects=predicted_objects)	
	time_last_say = timei
	print(prompt_obi)
	print("############ OUTPUT FROM API ########")

	# Send a completion call to generate an answer
	response = openai.Completion.create(engine=deployment_name, prompt=prompt_i, \
										max_tokens=100, temperature=0, stop = ['\n\n', '---', '"""', "'''"])
	text = response['choices'][0]['text']
	
	print(text)
	print()
	print()

	return time_last_say, prompt_i, text



def getInstruct(file_name, MTYPE, all_vid_path, out_path):
	############################################ import everything ############################################
	############################################ import everything ############################################

	### get recipe type
	recipe_type = file_name.split('_')[-1]
	print("### RECIPE: " + recipe_type)

	# import start and end time
	with open(os.path.join(all_vid_path, file_name, 'Video/VideoMpegTiming.txt')) as f:
		lines = f.readlines()
	vid_start = int(lines[0][:-1])
	vid_end = int(lines[1][:-1])
	print("Start/End Time:\t", vid_start, vid_end)

	# import steps
	in_step = []
	with open(os.path.join(all_vid_path, file_name,'StepDetection/StepDetection.txt')) as f:
		lines = f.readlines()
	for li in lines:
		tp = li[:-1].split('\t')
		tp[0] = int(tp[0])
		tp[1] = int(tp[1])
		in_step.append(tp)
	print("Num Steps:\t", len(in_step))


	# import instruct asr
	in_inst_asr = []
	with open(os.path.join(all_vid_path, file_name,'TextASR/InstructorAnnotations_intent.txt')) as f:
		lines = f.readlines()
	for li in lines:
		tp = li[:-1].split('\t')
		tp[0] = int(tp[0])
		tp[1] = int(tp[1])
		in_inst_asr.append(tp)
	print("Num Inst ASR:\t", len(in_inst_asr))
		
	# import user asr
	in_user_asr = []
	with open(os.path.join(all_vid_path, file_name,'TextASR/UserAnnotations_intent.txt')) as f:
		lines = f.readlines()
	for li in lines:
		tp = li[:-1].split('\t')
		tp[0] = int(tp[0])
		tp[1] = int(tp[1])
		in_user_asr.append(tp)
	print("Num User ASR:\t", len(in_user_asr))

	### import GT recipe
	recipe_name = os.path.join(name_li_path, "recipe_" + recipe_type + ".txt")
	prompt_recipe = ""
	with open(recipe_name) as f:
		lines = f.readlines()
	for li in lines:
		prompt_recipe += li


	### Import video
	video = VideoFileClip(os.path.join(all_vid_path, file_name,"Video/Video.mpeg"))
	audio = video.audio
	num_frames = int(video.fps * video.duration)
	frame_ratio = num_frames/(vid_end - vid_start)
	frames = video.iter_frames()

	print("duration\t", str(video.duration))
	print("num_frames\t", str(num_frames))
	print()


	############################################ Model Specific Prep ############################################
	############################################ Model Specific Prep ############################################
	
	### Prepare Obj Detection
	if MTYPE == 'objDet':
		model_obj1, model_cb, model_hand, objs_pinwheel_out, pinwheel_dic_clean = prep_obj(recipe_type)
		egoModels = [model_obj1, model_cb, model_hand]


	############################################ Going through Frames ############################################
	############################################ Going through Frames ############################################
	last_say_user = ""
	last_say_inst = ""
	time_last_say = 0
	chat_history = []
	obj_history = []
	obj_history_state_dic = {}


	### run detection model on each frame
	for i, fra in enumerate(frames):
		# remove before start and after end
		if (in_step[0][2] == 'Start') and (i <= (in_step[0][1] - vid_start)*frame_ratio):
			continue
		if (in_step[-1][2] == 'Done') and (i >= (in_step[-1][0] - vid_start)*frame_ratio):
			break
		timei = round(i/frame_ratio/1e7, 1)
		prompt_i = ""
		vb_generated_text = None
		predicted_objects = None
		prompt_type3 = None

		# everytime inst talks
		say_ins = parse_text_time(i, in_inst_asr, vid_start, frame_ratio)
		if (len(say_ins) != 0) and (say_ins[0] != last_say_inst):
			prompt_type3 = "##### INSTRUCTOR PROMPT: " + str(i) +  '/' + str(num_frames)
			print(prompt_type3)
			last_say_inst = say_ins[0]
			if MTYPE == 'objDet':
				predicted_objects, obj_history, obj_history_state_dic = get_obj_states(fra, obj_history, \
									obj_history_state_dic, egoModels, objs_pinwheel_out, pinwheel_dic_clean)
			elif MTYPE == 'blip2':
				vb_generated_text = call_BLIP2(fra)
			time_last_say, prompt_i, text = genAndCall(prompt_recipe, chat_history, timei, fill_user_asr=None, \
										fill_inst_asr=last_say_inst, vb_generated_text=vb_generated_text, \
										predicted_objects=predicted_objects)
			chat_history.append([timei, "You", last_say_inst])

		
		# everytime user talks
		say_usr = parse_text_time(i, in_user_asr, vid_start, frame_ratio)
		if (len(say_usr) != 0) and (say_usr[0] != last_say_user):			
			prompt_type3 = "##### USER PROMPT: " + str(i) + '/' + str(num_frames)
			print(prompt_type3)
			last_say_user = say_usr[0]
			if MTYPE == 'objDet':
				predicted_objects, obj_history, obj_history_state_dic = get_obj_states(fra, obj_history, \
									obj_history_state_dic, egoModels, objs_pinwheel_out, pinwheel_dic_clean)
			elif MTYPE == 'blip2':
				vb_generated_text = call_BLIP2(fra)
			time_last_say, prompt_i, text = genAndCall(prompt_recipe, chat_history, timei, fill_user_asr=last_say_user, \
									fill_inst_asr=None, vb_generated_text=vb_generated_text, \
									predicted_objects=predicted_objects)
			chat_history.append([timei, "User", last_say_user])

		
			# not talking for wait time
		if (prompt_i == "") and ((i/frame_ratio/1e7 - time_last_say) > time_wait):
			prompt_type3 = "##### WAIT PROMPT: " + str(i) + '/' + str(num_frames)
			print(prompt_type3)
			if MTYPE == 'objDet':
				predicted_objects, obj_history, obj_history_state_dic = get_obj_states(fra, obj_history, \
									obj_history_state_dic, egoModels, objs_pinwheel_out, pinwheel_dic_clean)
			elif MTYPE == 'blip2':
				vb_generated_text = call_BLIP2(fra)
			time_last_say, prompt_i, text = genAndCall(prompt_recipe, chat_history, timei, fill_user_asr=None, \
									fill_inst_asr=None, vb_generated_text=vb_generated_text, \
									predicted_objects=predicted_objects)


		# write prompts to file
		if prompt_i != "":
			with open(os.path.join(out_path, "prompt_" + file_name + ".txt"), "a") as file1:
				file1.writelines("##### FRAME: " + str(i) + '/' + str(num_frames) + '\n')
				file1.writelines(prompt_i)

			with open(os.path.join(out_path, "api_" + file_name + ".txt"), "a") as file2:
				file2.writelines("\n\n" + prompt_type3)
				file2.writelines("\n##### FRAME: " + str(i) + '/' + str(num_frames) + '\n')
				file2.writelines(text)
		



def main(MTYPE, video_list, in_path, out_path):
	# open a list of file names
	with open(video_list) as f:
		lines = f.readlines()

		# go through each recording one by onne
		for li in lines:
			print(li[:-1])
			getInstruct(li[:-1], MTYPE, in_path, out_path)



if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--MTYPE', '-t',
				help='Vision to Language Method Type: ["lanOnly", "blip2", "objDet"]', required=True)
	argparser.add_argument('--in_path', '-i',
				help='Video input path', required=True)
	argparser.add_argument('--video_list', '-l',
				help='Path to the file with a list of videos to be processed', required=True)
	argparser.add_argument('--out_path', '-o',
				help='Guidance output path', required=True)
	args = argparser.parse_args()

	main(args.MTYPE, args.video_list, args.in_path, args.out_path)

