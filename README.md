# SIV_UniTN_TAS_project

## Preparation

1. Clone the main repo [SIV_UniTN_TAS_project](https://github.com/marcorags/SIV_UniTN_TAS_project) including also the submodule:
    - ``` git clone –recurse-submodules https://github.com/marcorags/SIV_UniTN_TAS_project ```
    - Otherwise, if the repo was already cloned: ``` git submodule update –init –recursive ```
2. Move into the folder:
    - ``` cd .\SIV_UniTN_TAS_project ```
3. Install the requirements:
    - ``` pip install -r requirements.txt ```
4. Download the json dataset from [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D) following the instructions.
5. Create folder data in CVPR2024-FACT:
    - ```mkdir  .\CVPR2024-FACT\data```

## Format Data

Before feeding the dataset to the FACT model preprocess it.
From the ./SIV_UniTN_TAS_project folder run:

``` py ./utils/format.py ```

## Training

```shell
python -m CVPR2024-FACT.train --cfg CVPR2024-FACT/configs/fsjump.yaml 
```

To modify the model parameters refer to the fsjump.yaml file.

## Evaluation

```shell
python -m CVPR2024-FACT.eval
```

## Additional Notes
- Ensure that Python is installed on your system (recommended version: >=3.8).
- If you encounter issues with the packages or running the code, verify that all dependencies listed in `requirements.txt` have been correctly installed.

## Report
The report (a brief explanation of the project and its conceptualization) can be read [here](https://github.com/marcorags/SIV_UniTN_TAS_project/blob/main/SIV_Report_Fiorentino_Ragusa.pdf)

## Credits

This project was inspired by:
- [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT)
- [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D)
- [3D Pose-Based Temporal Action Segmentation for Figure Skating: A Fine-Grained and Jump Procedure-Aware Annotation Approach](https://arxiv.org/abs/2408.16638)
