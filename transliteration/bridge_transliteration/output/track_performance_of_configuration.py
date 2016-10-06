import os
import json
import sys
lang1=sys.argv[1]
lang2=sys.argv[2]
dir=lang1+"-en-"+lang2

for lang_pair in os.listdir('.'):
    #try:
	if lang_pair.startswith(dir):
		best_epoch = ""
                best_corr = 0.
		epoch_e2f_acc = {}
		epoch_g2f_acc = {}
		dim = 0
                args = json.load(open(lang_pair+'/args.json'))
                dim = float(args['rnn_size'])
		for file in sorted(os.listdir(lang_pair+'/output')):
			config = lang_pair.split('_')[-1]
			if file.startswith('e2f_acc.txt'):
				for line in open(lang_pair+'/output/'+file).readlines():
					line=line.strip()
					epoch = line.split()[5]
					acc = float(line.split()[14])
					epoch_e2f_acc[epoch]=acc
			if file.startswith('g2f_acc.txt'):
                                for line in open(lang_pair+'/output/'+file).readlines():
                                        line=line.strip()
                                        epoch = line.split()[5]
                                        acc = float(line.split()[14])
                                        epoch_g2f_acc[epoch]=acc
			if file.startswith('valid_g2e'):
				epoch_number = file.split('_')[-1].replace('.txt','')
				avg_corr = 0.
				count = 0.
				for line in open(lang_pair+'/output/'+file).readlines():
					avg_corr = avg_corr + float(line.strip().split(' ')[-1])
					count = count + 1.
				avg_corr = avg_corr/count
				avg_corr = avg_corr/dim
				prod = avg_corr * epoch_e2f_acc[epoch_number]
				#if avg_corr > 0.7:
				print 'Config'+' '+str(config)+' '+'Step'+' '+str(epoch_number)+' '+'Avg.Corr.'+' '+str(avg_corr)+' '+'e2f_Acc.'+' '+str(epoch_e2f_acc[epoch_number])+' '+'g2f_Acc.'+' '+str(epoch_g2f_acc[epoch_number])+' '+'Prod.'+' '+str(prod)
				if best_corr < avg_corr:
					best_corr = avg_corr
					best_epoch = epoch_number
