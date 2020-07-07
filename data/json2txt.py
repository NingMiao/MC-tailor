import json
import os

dir_list=[x for x in os.listdir() if x.startswith('on')]
for dir in dir_list:
  file_list=os.listdir(dir)
  for file in file_list:
    if not file.endswith('json'):
      continue
    with open(os.path.join(dir, file)) as f:
      data=json.load(f)
      data_new=[]
      for item in data:
        data_new.append(' '.join(item['tokens']))
    with open(os.path.join(dir, file[:-4]+'txt'), 'w') as g:
      g.write('\n'.join(data_new))