"""
This script should be run from root directory of this codebase:
https://github.com/tylin/coco-caption
"""

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys


def loadImgToAnns(json_file):
        data = json.load(open(json_file))
        imgToAnns = {}
        for entry in data:
            if entry['image_id'] not in imgToAnns.keys():
                    imgToAnns[entry['image_id']] = []
            caption = {}
            caption['caption'] = entry['caption']
            caption['image_id'] = entry['caption']
            imgToAnns[entry['image_id']].append(caption)
        return imgToAnns


res_json = sys.argv[1]
ref_json = sys.argv[2]
'''
annFile = 'annotations/captions_val2014.json'
coco = COCO(annFile)
valids = coco.getImgIds()

checkpoint = json.load(open(input_json, 'r'))
preds = checkpoint['val_predictions']

# filter results to only those in MSCOCO validation set (will be about a third)
preds_filt = [p for p in preds if p['image_id'] in valids]
print 'using %d/%d predictions' % (len(preds_filt), len(preds))
json.dump(preds_filt, open('tmp.json', 'w')) # serialize to temporary json file. Sigh, COCO API...
'''
coco = loadImgToAnns(ref_json)
cocoRes = loadImgToAnns(res_json)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.keys()
cocoEval.evaluate()

# create output dictionary
out = {}
for metric, score in cocoEval.eval.items():
    out[metric] = score
# serialize to file, to be read from Lua
json.dump(out, open('results.json', 'w'))

'''
def loadImgToAnns(json_file):
    data = json.load(open(json_file))
    imgToAnns = {}
    for entry in data:
        if entry['image_id'] not in imgToAnns.keys():
                imgToAnns[entry['image_id']] = []
        imgToAnns[entry['image_id']].append(entry['caption'])
    return imgToAnns
'''
