Monolingual Model Training.

`build_lid_data.py` - collated data from various sources and uploads to huggingface. Collated dataset can be assessed 
on hugging face [Language ID Training Data – Hugging Face](https://huggingface.co/datasets/JessicaOjo/lid-training-data)

`training_script.py` - converts the data into the training format, subsample as described in the report and splits into dev and train. 
The training section performs hyperparameter tuning on several parameters, evaluates on the dev set. Final model is evaluated on the test datasets.

Result files;
- [Baseline Results](https://drive.google.com/drive/folders/1x9Y4Fc8puHmdHGJr3vJ1_mQTHmUeAgrk?usp=sharing)
- [OUR model results](https://drive.google.com/drive/folders/1iLiU-vL0SnT1ClwrD7JkTnSXStFUCp9u?usp=sharing)

