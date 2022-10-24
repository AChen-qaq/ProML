import xml.etree.ElementTree as ET
import os


def get_label(attrs_list, l, r):
    for attrs in attrs_list:
        if len(attrs) == 0:
            continue
        st, ed = int(attrs['start']), int(attrs['end'])
        if l >= ed or r <= st:
            continue
        return attrs['TYPE']
    return 'O'


fl = os.listdir('data/testing-PHI-Gold-fixed')
for fn in fl:
    path = os.path.join('data/testing-PHI-Gold-fixed', fn)
    tree = ET.parse(path)
    root = tree.getroot()
    raw_text = root.find('TEXT').text
    attrs_list = [label.attrib for label in root.find('TAGS').iter() if len(label.attrib) > 0]
    fst = [int(attrs['start']) for attrs in attrs_list] + [int(attrs['end']) for attrs in attrs_list] + [i for i in range(len(raw_text)) if raw_text[i].isspace()] + [i+1 for i in range(len(raw_text)) if raw_text[i].isspace()] + [i for i in range(len(raw_text)) if raw_text[i] in [',', '.', '?', ':', '!']] + [i+1 for i in range(len(raw_text)) if raw_text[i] in [',', '.', '?', ':', '!']]
    fst = sorted(list(set(fst)))
    flag = False
    for i in range(len(fst)-1):
        token = raw_text[fst[i]:fst[i+1]]
        if token == '\n' and flag == False:
            print('\n')
            flag = True
        if token.isspace():
            continue
        flag = False
        label = get_label(attrs_list, fst[i], fst[i+1]) 
        print(token, label)
        if token in ['.', '?', '!']:
            print('\n')
            flag = True

    
