import codecs, sys
from xml.dom import minidom
import numpy as np

xmldoc = minidom.parse(sys.argv[1])
names = xmldoc.getElementsByTagName('Name')

srcfp=codecs.open(sys.argv[2], 'w', 'utf-8')
tgtfp=codecs.open(sys.argv[3], 'w', 'utf-8')

print len(names)

sources = []
targets = []
for name in names:
    src = name.childNodes[1].childNodes[0].nodeValue
    print src
    tgts = []
    #for i in range(3, len(name.childNodes), 2) :
    for i in range(3, 4, 2) :
        tgts.append(name.childNodes[i].childNodes[0].nodeValue)

    src_tokens = src.strip().split(' ')
    for tgt in tgts :
        tgt_tokens = tgt.strip().split()
        if len(tgt_tokens) != len(src_tokens) :
            continue
        for i in range(len(tgt_tokens)) :
            sources.append(' '.join([x for x in src_tokens[i].lower().strip()][::-1]))
            targets.append(' '.join([x for x in tgt_tokens[i].lower().strip()]))

perm = list(np.arange(len(sources)))
np.random.shuffle(perm)

#for src, tgt in  zip(sources, targets):
for i in perm :
    print >> srcfp, sources[i]
    print >> tgtfp, targets[i]

srcfp.close()
tgtfp.close()
