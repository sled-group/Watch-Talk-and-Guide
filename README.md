# Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake?

## TLDR

- **Authors:** Yuwei Bao, Keunwoo Peter Yu, Yichi Zhang, Shane Storks, Itamar Bar-Yossef, Alexander De La Iglesia, Megan Su, Xiao Lin Zheng, Joyce Chai
- **Organization:** University of Michigan, Computer Science and Engineering
- **Published in:** EMNLP 2023, Singapore
- **Links:** [Arxiv](https://arxiv.org/abs/2311.00738), [Github](https://github.com/sled-group/Watch-Talk-and-Guide), [Dataset](#Dataset)



## WTaG: Watch, Talk and Guide Dataset
<img src="/assets/demo.gif" width="100%"/>

WTaG is human-human task guidance dataset with natural language communications, mistake and corrections, and synchronized egocentric videos with transcriptions. WTaG is richly annotated with step detection, user and instructor dialog intention, and mistakes. 


### Dataset Stats

- #Videos Length:  10 hours
- #Videos:         56
- #Recipes:        3
- #User:           17
- #Instructor:     3
- #Total Utterances: 4233
- Median Video Length: 10 mins

### Dataset Features and Annotations

- Synchronized user and instructor egocentric video + audio transcriptions
- Annotation: Step detection
- Annotation: Human filtered audio transcriptions
- Annotation: User utterrance dialog intention
- Annotation: User mistakes
- Annotations: Instructor utterrance dialog intention (+ detailed)

![alt text](https://github.com/sled-group/Watch-Talk-and-Guide/blob/main/assets/anno_intent.png)



## Abstract
Despite tremendous advances in AI, it remains a significant challenge to develop interactive task guidance systems that can offer situated, personalized guidance and assist humans in various tasks. These systems need to have a sophisticated understanding of the user as well as the environment, and make timely accurate decisions on when and what to say. To address this issue, we created a new multimodal benchmark dataset, Watch, Talk and Guide (WTaG) based on natural interaction between a human user and a human instructor. We further proposed two tasks: User and Environment Understanding, and Instructor Decision Making. We leveraged several foundation models to study to what extent these models can be quickly adapted to perceptually enabled task guidance. Our quantitative, qualitative, and human evaluation results show that these models can demonstrate fair performances in some cases with no task-specific training, but a fast and reliable adaptation remains a significant challenge. Our benchmark and baselines will provide a stepping stone for future work on situated task guidance.



![alt text](https://github.com/sled-group/Watch-Talk-and-Guide/blob/main/assets/example.png)



### Tasks

#### User and Environment Understanding

1. User intent prediction: Dialog intent of user’s last utterance, if any (options).
2. Step detection: Current step (options).
3. Mistake Existence and Mistake Type: Did the user make a mistake at time t (yes/no). If so, what type of mistake (options).


#### Instructor Decition Making

1. When to Talk: Should the instructor talk at time t (yes/no).
2. Instructor Intent: If yes to 1, instructor’s dialog intention (options).
3. Instruction Type: If yes to 1 and intent in 2 is “Instruction”, what type (options).
4. Guidance generation: If yes to 1, what to say in natural language.


### Dataset
- Regular Download [WTaG](https://www.dropbox.com/scl/fo/rri8hdi1b6xlxude1zv28/h?rlkey=xtx9v386hhsqk4ct6ix8kw53u&dl=0)
- Audio-Video WTaG, please sign additional licencing aggreement [HERE](https://docs.google.com/forms/d/e/1FAIpQLScM87-G5giqEGAhDTz2fp4puatHQLRX9qaEDMEDc0u7dyt2PA/viewform?usp=sf_link)




### SetUp
- Install [EgoHOS](https://github.com/owenzlz/EgoHOS)
- Install [CLIP](https://github.com/openai/CLIP)
- Insert your openai credentials to [open.py](https://github.com/sled-group/Watch-Talk-and-Guide/blob/main/src/open.py)
- Download dataset [WTaG](#Dataset)


### Running
To evaluate the three methods on WTaG:
```
python src/pipeline.py [-h] --MTYPE MTYPE --in_path IN_PATH --video_list VIDEO_LIST --out_path
                   OUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  --MTYPE MTYPE, -t MTYPE
                        Vision to Language Method Type: ["lanOnly", "blip2", "objDet"]
  --in_path IN_PATH, -i IN_PATH
                        Video input path
  --video_list VIDEO_LIST, -l VIDEO_LIST
                        Path to the file with a list of videos to be processed
  --out_path OUT_PATH, -o OUT_PATH
                        Guidance output path
 ```


 ## Citation
```
@inproceedings{bao-etal-2023-foundation,
    title = "Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake?",
    author = "Bao, Yuwei  and
      Yu, Keunwoo  and
      Zhang, Yichi  and
      Storks, Shane  and
      Bar-Yossef, Itamar  and
      de la Iglesia, Alex  and
      Su, Megan  and
      Zheng, Xiao  and
      Chai, Joyce",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.824",
    doi = "10.18653/v1/2023.findings-emnlp.824",
    pages = "12325--12341",
}
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. For further questions, please contact yuweibao@umich.edu.

