source /dccstor/anirlaha1/deep/tflow-0.10/bin/activate
export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/targets/ppc64le-linux/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda/
export PYTHONPATH=$PYTHONPATH:/dccstor/cssblr/anirban/

jbsub -queue p8 -require k80 -cores 1+1 -out out.txt -err err.txt \
python train_bridge_captions.py \
  --i_e_i_train=../mscoco_maxlength16/train/image/images.npy \
  --i_e_e_train=../mscoco_maxlength16/train/caption/train_captions_en.txt \
  --i_e_i_valid=../mscoco_maxlength16/valid/image/images.uniq.npy \
  --i_e_e_valid=../mscoco_maxlength16/valid/caption/valid_captions_en.0 \
  --e_f_e_train=../mscoco_maxlength16/train/caption/train_captions_en.txt \
  --e_f_f_train=../mscoco_maxlength16/train/caption/train_captions_fr.txt \
  --e_f_e_valid=../mscoco_maxlength16/valid/caption/valid_captions_en.0 \
  --e_f_f_valid=../mscoco_maxlength16/valid/caption/valid_captions_fr.0 \
  --i_f_i_test=../mscoco_maxlength16/test/image/images.uniq.npy \
  --i_f_f_test=../mscoco_maxlength16/test/cf_caption/captions_fr.0 \
  --pretrain_epochs=25 \
  --save_every=10000 \
  --valid_every=10000 \
  --lmbda=1.0 \
  --max_epochs=100 \
  --save_dir=im_en_fr_bow_0
