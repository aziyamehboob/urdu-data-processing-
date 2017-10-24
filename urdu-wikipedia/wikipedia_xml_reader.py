"""
Read given xml file as a stream without loading the whole file in memory and extract urdu
text from it
"""

import sys
import xml.etree.ElementTree as etree

NS = '{http://www.mediawiki.org/xml/export-0.10/}'


def plain_name(s_in):
    if not s_in:
        return ''

    us = ''
    for ch in s_in:
        if ord(ch) > 1000 or ch.isalnum():
            us += ch

    return us

def just_urdu(s_in):
    """
    Just return Urdu text from the given string).
    Current implementation just removes all english characters, numbers punctuations etc
    """
    
    if not s_in:
        return ''

    us = ''
    in_urdu = False
    for ch in s_in:
        if ord(ch) > 1000:
            in_urdu = True
            us += ch
        else:
            if in_urdu:
                us += ' '
            in_urdu = False

    return us


if '__main__' == __name__:
    if len(sys.argv) != 3:
        print("%s xml_input_file output_folder" % sys.argv[0])
        sys.exit(1)

    print("Processing ", sys.argv[1])
    #idx = 0
    #nsmap = {}
    #for event, elem in etree.iterparse(sys.argv[1], events=('start-ns', )):
        #ns, url = elem
        #print(ns, url)
        #nsmap[ns] = url
        #idx += 1
        #if idx == 10:
            #break
    #print(nsmap)
    
    #idx = 0
    data_dict = {}
    for event, elem in etree.iterparse(sys.argv[1], events=('start', 'end', )):
        tag = elem.tag.split('}')[1]
        if 'start' == event and 'page' == tag:
            data_dict = {}
        elif 'end' == event and 'page' == tag:
            elem.clear()
            #print(data_dict['ns'], '-', data_dict['title'])
            if data_dict['ns'] == '0' and len(data_dict['text']) > 500:
                # Save to file
                filepath = sys.argv[2] + '/' + plain_name(data_dict['title'])
                open(filepath, 'w').write(data_dict['text'])
                print("  wrote ", filepath)

        if 'title' == tag and 'end' == event:
            data_dict['title'] = elem.text
        elif 'text' == tag and 'end' == event:
            data_dict['text'] = just_urdu(elem.text)
        elif 'ns' == tag and 'end' == event:
            data_dict['ns'] = elem.text

        #idx += 1
        #if idx == 15000:
            #break

