source /dccstor/anirlaha1/deep/tflow-0.10/bin/activate
export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/targets/ppc64le-linux/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda/
export PYTHONPATH=$PYTHONPATH:/dccstor/cssblr/anirban/



mkdir im_en_01
mkdir im_de_01

#jbsub -queue p8 -require k80 -cores 1+1 -out im_en_01/log.txt -err im_en_01/err.txt \
python train_bridge_captions.py \
--i_e_e_train=../../wmt_workshop_task2/train/caption/train_captions_en.txt \
--i_e_i_train=../../wmt_workshop_task2/train/image/images.npy \
--i_e_e_valid=../../wmt_workshop_task2/valid/caption/valid_captions_en.uniq.txt \
--i_e_i_valid=../../wmt_workshop_task2/valid/image/images.uniq.npy \
--i_e_e_test=../../wmt_workshop_task2/test/caption/captions_en.uniq.txt \
--i_e_i_test=../../wmt_workshop_task2/test/image/images.uniq.npy \
--save_dir=im_en_01

#jbsub -queue p8 -require k80 -cores 1+1 -out im_de_01/log.txt -err im_de_01/err.txt \
python train_bridge_captions.py \
--i_e_e_train=../../wmt_workshop_task2/train/caption/train_captions_de.txt \
--i_e_i_train=../../wmt_workshop_task2/train/image/images.npy \
--i_e_e_valid=../../wmt_workshop_task2/valid/caption/valid_captions_de.uniq.txt \
--i_e_i_valid=../../wmt_workshop_task2/valid/image/images.uniq.npy \
--i_e_e_test=../../wmt_workshop_task2/test/caption/captions_de.uniq.txt \
--i_e_i_test=../../wmt_workshop_task2/test/image/images.uniq.npy \
--save_dir=im_de_01
