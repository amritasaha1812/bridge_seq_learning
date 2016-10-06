import json
from watson_developer_cloud import LanguageTranslationV2 as LanguageTranslation
import sys
import re
import codecs
language_translation = LanguageTranslation(
    username='7fe3b207-99e1-44b6-bcad-b012bc31048d',
    password='gamYIBLvmCj2')
num = 0
step_number = sys.argv[1].split('_')[-1]
#print "... writing into 'IBM_translated_captions_fr.'"+step_number
#translations = codecs.open('IBM_translated_captions_fr.'+step_number,'w', encoding='utf-8')
for line in open(sys.argv[1]):
	if line.startswith('Valid Predicted:'):
	        line = re.sub(' +',' ',line.strip().replace('Valid Predicted: ','').replace('<EOS>', '').replace('<GO>','').replace('<PAD>','').replace('<OOV>',''))
		#print "\""+line+"\""
        	translation = language_translation.translate(
	                text=line,
        	        source='en',
                	target='fr')
		print translation.strip().encode(encoding='utf-8')
	        #translations.write(translation.strip().encode(encoding='utf-8')+'\n')
#translations.close()
