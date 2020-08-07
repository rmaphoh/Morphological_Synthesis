
# cuda 10.1 cudnn 7.5
#!/bin/bash  
source /home/yukun/source/cuda_10.1.source
#source activate amd

source activate tensorflow2

export PYTHONPATH=.:$PYTHONPATH
#python test.py
#python examples/predict_simplified_score.py data/left_eye.jpg data/right_eye.jpg
#python examples/predict_drusen.py data/AMD/A0015.jpg
#python examples/train.py data/iAMD_pigment_label_sample.csv checkpoints/AMD/iAMD_challenge/inceptionV3_fl_512_tensorflow/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5
#python examples/predict_AMD.py -d 'inceptionv3_512_tensorflow' ./data/final_img.jpg
#python examples/predict_AMD.py --model_name=inceptionv3_512_tensorflow --image_directory=/SAN/medic/OVS2020/CMIC_OphthalNet/data/generated_fake 
python main.py --model_name=Unet --train_directory=./representation_syn/train_files.csv --test_directory=./representation_syn/test_files.csv \
                --batch_size=4 \
                --checkpoint_path=./checkpoints/Unet/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5