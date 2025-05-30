# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

## 1. Requirements
- folder inside project root named `outputs` with all wanted rtf report files
- csv file named `podatki` inside project `code` folder with csv data for the same dates as the rtf reports

## 2. Local data setup
When all of the requirements are setup you will have to run `code/test_distance.py`. This will generate a file named `useful_matches.json`. 

## 3. Running on HPC 
> We have already done steps 1 and 2. So you only need to do this one.

Connect to HPC and move to `/d/hpc/projects/onj_fri/vmzteam`. Then run `run_slurm.sh`.

This will create a file called `similarity_scores.csv` which contains the calculated BLUE and BERT scores of the generated reports. And it will also create a folder named `llm_outputs` in the `vmzteam` folder that contains the generated reports.