# Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake?

## TLDR

- **Authors:** Yuwei Bao, Keunwoo Peter Yu, Yichi Zhang, Shane Storks, Itamar Bar-Yossef, Alexander De La Iglesia, Megan Su, Xiao Lin Zheng, Joyce Chai
- **Organization:** University of Michigan, Computer Science and Engineering
- **Published in:** EMNLP 2023, Singapore
- **Links:** [Arxiv](https://arxiv.org/abs/2311.00738), [Github](https://github.com/sled-group/Watch-Talk-and-Guide), [Dataset](https://forms.gle/CMgWadMtbaA7tNnE6)


## Abstract
Despite tremendous advances in AI, it remains a significant challenge to develop interactive task guidance systems that can offer situated, personalized guidance and assist humans in var- ious tasks. These systems need to have a so- phisticated understanding of the user as well as the environment, and make timely accurate decisions on when and what to say. To ad- dress this issue, we created a new multimodal benchmark dataset, Watch, Talk and Guide (WTaG) based on natural interaction between a human user and a human instructor. We further proposed two tasks: User and Environment Understanding, and Instructor Decision Making. We leveraged several foundation models to study to what extent these models can be quickly adapted to perceptually enabled task guidance. Our quantitative, qualitative, and human evaluation results show that these mod- els can demonstrate fair performances in some cases with no task-specific training, but a fast and reliable adaptation remains a significant challenge. Our benchmark and baselines will provide a stepping stone for future work on situated task guidance.



![alt text](https://github.com/sled-group/Watch-Talk-and-Guide/blob/main/assets/example.png)


### SetUp
- Install [EgoHOS](https://github.com/owenzlz/EgoHOS)
- Install [CLIP](https://github.com/openai/CLIP)
- Insert your openai credentials to [open.py](https://github.com/sled-group/Watch-Talk-and-Guide/blob/main/src/open.py)
- Download dataset [WTaG](https://forms.gle/CMgWadMtbaA7tNnE6)


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
@misc{bao2023foundation,
      title={Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake?}, 
      author={Yuwei Bao and Keunwoo Peter Yu and Yichi Zhang and Shane Storks and Itamar Bar-Yossef and Alexander De La Iglesia and Megan Su and Xiao Lin Zheng and Joyce Chai},
      year={2023},
      eprint={2311.00738},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. For further questions, please contact yuweibao@umich.edu.

