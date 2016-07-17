__author__ = 'ubriela'


import xml.etree.ElementTree as ET


def cdi_parse(cdi_file='../../dataset/napa/cdi_geo.xml'):
    tree = ET.parse(cdi_file)
    root = tree.getroot()[0]
    total = 0
    results = []
    for child in root:
        total = total + int(child[2].text)
        val = (float(child[1].text), int(child[2].text), float(child[4].text), float(child[5].text))
        # print child[4].text, '\t', child[5].text
        results.append(val)
    return results

    # print total