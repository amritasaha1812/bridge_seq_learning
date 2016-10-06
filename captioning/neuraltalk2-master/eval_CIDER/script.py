import json, sys
lines_r1 = open(sys.argv[1]).readlines()
lines_r2 = open(sys.argv[2]).readlines()
lines_r3 = open(sys.argv[3]).readlines()
lines_r4 = open(sys.argv[4]).readlines()
lines_r5 = open(sys.argv[5]).readlines()
lines_pred = open(sys.argv[6]).readlines()
data_ref = []
data_pred = []
image_id = 1
for line1,line2,line3,line4,line5,line_pred in zip(lines_r1,lines_r2,lines_r3,lines_r4,lines_r5,lines_pred):
    line = {}
    line['image_id'] = image_id
    line['caption'] = line1.strip()
    data_ref.append(line)

    line = {}
    line['image_id'] = image_id
    line['caption'] = line2.strip()
    data_ref.append(line)

    line = {}
    line['image_id'] = image_id
    line['caption'] = line3.strip()
    data_ref.append(line)

    line = {}
    line['image_id'] = image_id
    line['caption'] = line4.strip()
    data_ref.append(line)

    line = {}
    line['image_id'] = image_id
    line['caption'] = line5.strip()
    data_ref.append(line)

    line = {}
    line['image_id'] = image_id
    line['caption'] = line_pred.strip()
    data_pred.append(line)
    image_id = image_id + 1
json.dump(data_ref, open('ref.json','wb'), indent=4)
json.dump(data_pred, open('pred.json','wb'), indent=4)

