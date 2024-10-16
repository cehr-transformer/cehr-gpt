# CEHR-GPT for ML4H 2024

This project is the continuation of the CEHR-BERT work, which has been published
at https://proceedings.mlr.press/v158/pang21a.html. 

## Warning
This repo is only for demonstration purposes and do not use it in any production. 
A newer version of CEHR-GPT is being developed in Huggingface, and will be released soon. 

## Getting Started

### Pre-requisite

The project is built in python 3.10, and project dependency needs to be installed

`pip3 install -r requirements.txt`

Create the following folders for the tutorial below

```bash
mkdir -p ~/Documents/omop_test/cehrgpt;
mkdir -p ~/Documents/omop_test/cehrgpt/model;
mkdir -p ~/Documents/omop_test/cehrgpt/log;
```
### Training CEHR-GPT
#### 1. Download OMOP tables as parquet files

We have created a spark app to download OMOP tables from Sql Server as parquet files. You need adjust the properties
in `db_properties.ini` to match with your database setup.

```bash
spark-submit tools/download_omop_tables.py -c db_properties.ini -tc person visit_occurrence condition_occurrence procedure_occurrence drug_exposure measurement observation_period concept concept_relationship concept_ancestor -o ~/Documents/omop_test/
```
Generate a random patient split
```bash
spark-submit tools/generate_patient_splits.py --input_folder ~/Documents/omop_test/ --output_folder ~/Documents/omop_test/
```
#### 2. Generate the required concept list

This will generate a list of concepts that occurred at least in 100 patients. We will use this table in the next step to
filter out the low frequency concepts to preserve privacy.
```bash
PYTHONPATH=./: spark-submit spark_apps/generate_included_concept_list.py \
--input_folder ~/Documents/omop_test/ \
--output_folder ~/Documents/omop_test/ \
--min_num_of_patients 100
```

#### 3. Generate training data for CEHR-GPT
```bash
PYTHONPATH=./: spark-submit spark_apps/generate_training_data.py  \
-i ~/Documents/omop_test/ \
-o ~/Documents/omop_test/cehrgpt/ \
-tc condition_occurrence procedure_occurrence drug_exposure \
--gpt_patient_sequence \
--att_type day \
--date_filter 1985-01-01 \
--include_concept_list \
--is_new_patient_representation -iv  
```

#### 4. Training cehr-gpt
```bash
PYTHONPATH=./: python3 trainers/train_gpt.py \
-i ~/Documents/omop_test/cehrgpt/patient_sequence/train \
-o ~/Documents/omop_test/cehrgpt/ \
--concept_path ~/Documents/omop_test/concept \
-d 16 -b 32 -m 512 --print_every 100000 \
-e 2 --min_num_of_concepts 20 --max_num_of_visits 100000 \
--num_of_patients 1024 --sampling_batch_size 256 -lr 5e-5 \
--save_checkpoint --save_freq 10000 \
--including_long_sequence &> ~/Documents/omop_test/cehrgpt/log/gpt_model_continous_training.out &
```

#### 5. Generate synthetic sequences
```bash
cp ~/Documents/omop_test/cehrgpt/model/bert_model_[epoch]_[loss].h5 ~/Documents/omop_test/cehrgpt/model/bert_model.h5
#This will use TopPStrategy and top_p=0.95 will be used in stead of top_k=300 
PYTHONPATH=./: python3 -u tools/generate_batch_gpt_sequence.py \
--model_folder ~/Documents/omop_test/cehrgpt/model \
--output_folder ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/ \
--num_of_patients 100000 --batch_size 256 --buffer_size 1024 \
--context_window 512 --top_k 300 --top_p 0.95 --temperature 1.0 \
--demographic_data_path ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--sampling_strategy TopPStrategy &> ~/Documents/omop_test/cehrgpt/log/model_inference_correct_snapshot_top_p95.out &
```

#### 6. Convert synthetic sequences back to OMOP
```bash
PYTHONPATH=./: python3 tools/omop_converter_batch.py \
--patient_sequence_path ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/ \
--output_folder ~/Documents/omop_test/cehrgpt/restored_omop_top_p95/ \
--concept_path ~/Documents/omop_test/concept --buffer_size 10000 --cpu_cores 10
```

### Utility Analyses
#### Co-occurrence Analysis
Generate the co-occurrence for the training data
```bash
PYTHONPATH=./:$PYTHONPATH python analysis/gpt/generate_cooccurrence.py \
--sequence_data_path ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--concept_path ~/Documents/omop_test/concept \
--data_cooccurrence_path ~/Documents/omop_test/cehrgpt/patient_sequence/train_cooccurrence \
--is_temporal_cooccurrence
```
Generate the co-occurrence for the synthetic data
```bash
PYTHONPATH=./:$PYTHONPATH python analysis/gpt/generate_cooccurrence.py \
--sequence_data_path ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/ \
--concept_path ~/Documents/omop_test/concept \
--data_cooccurrence_path ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95_cooccurrence \
--is_temporal_cooccurrence
```
Compute the KL divergence of the two co-occurrence matrices
```bash
PYTHONPATH=./:$PYTHONPATH python analysis/gpt/compare_cooccurrence.py \
--reference_cooccurrence_path ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--comparison_cooccurrence_path ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95_cooccurrence/
```
#### Replicate Predictive Tasks using synthetic data
```bash
venv-pack -o pyspark_venv.tar.gz
export SPARK_EXECUTOR_CORES=2
export SPARK_DRIVER_MEMORY=1g
export SPARK_EXECUTOR_MEMORY=1g
```
##### HF readmission
```bash
export OMOP_FOLDER=~/Documents/omop_test/cehrgpt/restored_omop_top_p95/;
mkdir -p $OMOP_FOLDER/cohorts/hf_readmission;
mkdir -p $OMOP_FOLDER/evaluation_gpt/hf_readmission/;

PYTHONPATH=./:$PYTHONPATH spark-submit --archives pyspark_venv.tar.gz#environment  --conf "spark.pyspark.driver.python=python" \
spark_apps/prediction_cohorts/hf_readmission.py -c hf_readmission_bow  -i $OMOP_FOLDER -o $OMOP_FOLDER/cohorts/hf_readmission \
-dl 1985-01-01 -du 2023-12-31 -l 18 -u 100 -ow 360 -ps 0 -pw 30 -f --att_type cehr_bert \
--ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv;

PYTHONPATH=./:$PYTHONPATH python3 evaluations/evaluation.py \
-a baseline_model -d $OMOP_FOLDER/cohorts/hf_readmission/hf_readmission_bow/ \
-ef $OMOP_FOLDER/evaluation_gpt/hf_readmission/;
```

##### Hospitalization
```bash
export OMOP_FOLDER=~/Documents/omop_test/cehrgpt/restored_omop_top_p95/;
mkdir -p $OMOP_FOLDER/cohorts/hospitalization;
mkdir -p $OMOP_FOLDER/evaluation_gpt/hospitalization/;

PYTHONPATH=./:$PYTHONPATH spark-submit --archives pyspark_venv.tar.gz#environment  --conf "spark.pyspark.driver.python=python" \
spark_apps/prediction_cohorts/hospitalization.py -c hospitalization_bow -i $OMOP_FOLDER -o $OMOP_FOLDER/cohorts/hospitalization/ \
-dl 1985-01-01 -du 2023-05-01 -l 18 -u 100 -ow 540 -hw 180 -ps 0 -pw 360 -f -iw --att_type cehr_bert \
--ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv;

PYTHONPATH=./:$PYTHONPATH python3 evaluations/evaluation.py \
-a baseline_model -d $OMOP_FOLDER/cohorts/hospitalization/hospitalization_bow/ \
-ef $OMOP_FOLDER/evaluation_gpt/hospitalization/;
```

### Privacy Analyses
Create the folder to store the privacy metrics 
``
mkdir ~/Documents/omop_test/cehrgpt/privacy
``
#### 1. Attribute Inference Analysis
```bash
PYTHONPATH=./:$PYTHONPATH python analysis/privacy/attribute_inference.py \
--training_data_folder ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--output_folder ~/Documents/omop_test/cehrgpt/privacy \
--synthetic_data_folder ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/ \
--tokenizer_path ~/Documents/omop_test/cehrgpt/model \
--attribute_config analysis/privacy/attribute_inference_config.yml \
--n_iterations 10 --num_of_samples 10000
```
#### 2. Membership Inference Analysis
```bash
PYTHONPATH=./:$PYTHONPATH python analysis/privacy/member_inference.py \
--training_data_folder  ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--evaluation_data_folder  ~/Documents/omop_test/cehrgpt/patient_sequence/test \
--output_folder ~/Documents/omop_test/cehrgpt/privacy \
--synthetic_data_folder ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/ \
--tokenizer_path ~/Documents/omop_test/cehrgpt/model \
--n_iterations 10 --num_of_samples 10000
```
#### 3. Nearest Neighbor Inference Analysis
```bash
mkdir ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/privacy/nearest_neighbor_inference;
PYTHONPATH=./:$PYTHONPATH python analysis/privacy/nearest_neighbor_inference.py \
--training_data_folder ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--evaluation_data_folder ~/Documents/omop_test/cehrgpt/patient_sequence/test \
--metrics_folder ~/Documents/omop_test/cehrgpt/privacy/nearest_neighbor_inference \
--synthetic_data_folder ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/ \
--concept_tokenizer_path ~/Documents/omop_test/cehrgpt/model \
--n_iterations 10 --num_of_samples 10000
```
#### 4. Re-identification Risk Inference Analysis
```bash
mkdir ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/privacy/reid;
PYTHONPATH=./:$PYTHONPATH python analysis/privacy/reid_inference.py \
--training_data_folder ~/Documents/omop_test/cehrgpt/patient_sequence/train \
--evaluation_data_folder ~/Documents/omop_test/cehrgpt/patient_sequence/test \
--output_folder ~/Documents/omop_test/cehrgpt/privacy/reid \
--synthetic_data_folder ~/Documents/omop_test/cehrgpt/generated_sequences_top_p95/
```