source /dccstor/anirlaha1/deep/venv/bin/activate
export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda-7.5/
export PYTHONPATH=$PYTHONPATH:/dccstor/anirlaha1/


lang1=$1
lang2=$2
./script.sh $lang1 $lang2 128 2048 2 0.0001 0.1 1
./script.sh $lang1 $lang2 128 2048 2 0.0001 0.5 2
./script.sh $lang1 $lang2 128 2048 2 0.0001 1 3
./script.sh $lang1 $lang2 128 2048 2 0.0005 0.1 4
./script.sh $lang1 $lang2 128 2048 2 0.0005 0.5 5
./script.sh $lang1 $lang2 128 2048 2 0.0005 1 6

./script.sh $lang1 $lang2 128 2048 3 0.0001 0.1 7
./script.sh $lang1 $lang2 128 2048 3 0.0001 0.5 8
./script.sh $lang1 $lang2 128 2048 3 0.0001 1 9
./script.sh $lang1 $lang2 128 2048 3 0.0005 0.1 10
./script.sh $lang1 $lang2 128 2048 3 0.0005 0.5 11
./script.sh $lang1 $lang2 128 2048 3 0.0005 1 12

./script.sh $lang1 $lang2 128 2048 4 0.0001 0.1 13
./script.sh $lang1 $lang2 128 2048 4 0.0001 0.5 14
./script.sh $lang1 $lang2 128 2048 4 0.0001 1 15
./script.sh $lang1 $lang2 128 2048 4 0.0005 0.1 16
./script.sh $lang1 $lang2 128 2048 4 0.0005 0.5 17
./script.sh $lang1 $lang2 128 2048 4 0.0005 1 18

#./script.sh $lang1 $lang2 128 1024 2 0.0001 0.1 19
#./script.sh $lang1 $lang2 128 1024 2 0.0001 0.5 20
#./script.sh $lang1 $lang2 128 1024 2 0.0001 1 21
#./script.sh $lang1 $lang2 128 1024 2 0.0005 0.1 22
./script.sh $lang1 $lang2 128 1024 2 0.0005 0.5 23
./script.sh $lang1 $lang2 128 1024 2 0.0005 1 24

./script.sh $lang1 $lang2 128 1024 3 0.0001 0.1 25
./script.sh $lang1 $lang2 128 1024 3 0.0001 0.5 26
./script.sh $lang1 $lang2 128 1024 3 0.0001 1 27
./script.sh $lang1 $lang2 128 1024 3 0.0005 0.1 28
./script.sh $lang1 $lang2 128 1024 3 0.0005 0.5 29
./script.sh $lang1 $lang2 128 1024 3 0.0005 1 30

./script.sh $lang1 $lang2 128 1024 4 0.0001 0.1 31
./script.sh $lang1 $lang2 128 1024 4 0.0001 0.5 32
./script.sh $lang1 $lang2 128 1024 4 0.0001 1 33
./script.sh $lang1 $lang2 128 1024 4 0.0005 0.1 34
./script.sh $lang1 $lang2 128 1024 4 0.0005 0.5 35
./script.sh $lang1 $lang2 128 1024 4 0.0005 1 36

./script.sh $lang1 $lang2 64 1024 2 0.0001 0.1 37
./script.sh $lang1 $lang2 64 1024 2 0.0001 0.5 38
./script.sh $lang1 $lang2 64 1024 2 0.0001 1 39
./script.sh $lang1 $lang2 64 1024 2 0.0005 0.1 40
./script.sh $lang1 $lang2 64 1024 2 0.0005 0.5 41
./script.sh $lang1 $lang2 64 1024 2 0.0005 1 42

./script.sh $lang1 $lang2 64 1024 3 0.0001 0.1 43
./script.sh $lang1 $lang2 64 1024 3 0.0001 0.5 44
./script.sh $lang1 $lang2 64 1024 3 0.0001 1 45
./script.sh $lang1 $lang2 64 1024 3 0.0005 0.1 46
./script.sh $lang1 $lang2 64 1024 3 0.0005 0.5 47
./script.sh $lang1 $lang2 64 1024 3 0.0005 1 48

./script.sh $lang1 $lang2 64 1024 4 0.0001 0.1 49
./script.sh $lang1 $lang2 64 1024 4 0.0001 0.5 50
./script.sh $lang1 $lang2 64 1024 4 0.0001 1 51
./script.sh $lang1 $lang2 64 1024 4 0.0005 0.1 52
./script.sh $lang1 $lang2 64 1024 4 0.0005 0.5 53
./script.sh $lang1 $lang2 64 1024 4 0.0005 1 54

./script.sh $lang1 $lang2 128 1024 2 0.0001 0.1 19
./script.sh $lang1 $lang2 128 1024 2 0.0001 0.5 20
./script.sh $lang1 $lang2 128 1024 2 0.0001 1 21
./script.sh $lang1 $lang2 128 1024 2 0.0005 0.1 22

./script.sh $lang1 $lang2 128 1024 2 0.0001 0.2 55
./script.sh $lang1 $lang2 128 1024 2 0.0001 0.3 56
./script.sh $lang1 $lang2 128 1024 2 0.0001 0.4 57
./script.sh $lang1 $lang2 128 1024 2 0.0002 0.1 58
./script.sh $lang1 $lang2 128 1024 2 0.0002 0.2 59
./script.sh $lang1 $lang2 128 1024 2 0.0002 0.3 60
./script.sh $lang1 $lang2 128 1024 2 0.0002 0.4 61
./script.sh $lang1 $lang2 128 1024 2 0.0002 0.5 62
./script.sh $lang1 $lang2 128 1024 2 0.0003 0.1 63
./script.sh $lang1 $lang2 128 1024 2 0.0003 0.2 64
./script.sh $lang1 $lang2 128 1024 2 0.0003 0.3 65
./script.sh $lang1 $lang2 128 1024 2 0.0003 0.4 66
./script.sh $lang1 $lang2 128 1024 2 0.0003 0.5 67
./script.sh $lang1 $lang2 128 1024 2 0.0004 0.1 68
./script.sh $lang1 $lang2 128 1024 2 0.0004 0.2 69
./script.sh $lang1 $lang2 128 1024 2 0.0004 0.3 70
./script.sh $lang1 $lang2 128 1024 2 0.0004 0.4 71
./script.sh $lang1 $lang2 128 1024 2 0.0004 0.5 72
./script.sh $lang1 $lang2 128 1024 2 0.0005 0.2 73
./script.sh $lang1 $lang2 128 1024 2 0.0005 0.3 74
./script.sh $lang1 $lang2 128 1024 2 0.0005 0.4 75
