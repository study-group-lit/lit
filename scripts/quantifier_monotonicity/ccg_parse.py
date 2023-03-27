import os
import re
import subprocess

from lxml import etree
from dataclasses import dataclass
from typing import List

from datasets import load_from_disk
from datasets import load_dataset


@dataclass
class Quantifier:
    name: str
    left_rising: bool
    right_rising: bool
    tags: List[List[str]]

        
all_quantifiers = [
    Quantifier("a few", True, True, [["DT", "JJ"]]),
    Quantifier("a large number of", True, True, [["DT", "JJ", "NN", "IN"]]),
    Quantifier("a little", True, True, [["DT", "JJ"]]),
    Quantifier("a number of", True, True, [["DT", "NN", "IN"]]),
    Quantifier("a small number of", True, True, [["DT", "JJ", "NN", "IN"]]),
    Quantifier("all", False, True, [["DT"]]),
    Quantifier("any", False, True, [["DT"]]),
    Quantifier("enough", True, True, [["DT"]]),
    Quantifier("each", False, True, [["DT"]]),
    Quantifier("every", True, True, [["DT"]]),
    Quantifier("few", False, False, [["DT"]]),
    Quantifier("fewer", False, False, [["DT"]]),
    Quantifier("less", False, False, [["DT"], ["RB"], ["IN"], ["JJR"]]), # Also adverb and preposition
    Quantifier("lots of", True, True, [["RB", "IN"], ["NNS", "IN"]]), # Idiom: adverb + preposition
    Quantifier("many", True, True, [["DT"], ["JJ"]]),
    Quantifier("most", False, True, [["DT"]]),
    Quantifier("most of", False, True, [["JJS", "IN"]]),
    Quantifier("much", True, True, [["DT"]]),
    Quantifier("much of", True, True, [["NN", "IN"]]),
    Quantifier("no", False, False, [["DT"]]),
    Quantifier("none of", False, False, [["NN", "IN"]]),
    Quantifier("not many", False, False, [["RB", "JJ"]]),
    Quantifier("not much", False, False, [["RB", "JJ"]]),
    Quantifier("numerous", True, True, [["JJ"]]), # Adjective
    Quantifier("plenty of", True, True, [["NN", "IN"]]), # Idiom: Pronoun + preposition
    Quantifier("several", True, True, [["DT"], ["JJ"]]), # Also pronoun
    Quantifier("some", True, True, [["DT"]]),
    Quantifier("whole", False, True, [["RB"], ["JJ"]]), # Adverb
    Quantifier("many of", True, True, [["NN", "IN"]]), # Noun + preposition
]

CANDC_PATH = "bin/candc-1.00"
SED_PATH = "scripts/quantifier_monotonicity/tokenizer.sed"
CANDC2TRANSCCG_PATH = "scripts/quantifier_monotonicity/candc2transccg.py"

def parse(sentence: str) -> "XML":
    sentence = sentence.replace("\"", "\\\"")
    ps = subprocess.run(
        f"echo \"{sentence}\" | "
        f"sed -f {SED_PATH} | "
        f"{CANDC_PATH}/bin/candc --models {CANDC_PATH}/models/ --candc-printer xml | "
        f"python {CANDC2TRANSCCG_PATH}",
        stdout=subprocess.PIPE,
        shell=True
        )
    xml = etree.fromstring(ps.stdout)
    return xml

xml = parse("Bush says most of Congress \" acting like a teenager with a new credit card \"")
floating_list = ["both", "all", "each"]
floating_flg = 0
element_id = (etree.XPath(".//span[@base=\"all\"]/@id"))(xml)[0]
verb_id = []
child_ids, child_verb_ids = [], []
while True:
    parent_id = xml.xpath("//ccg/span[contains(@child, '" + element_id + "')]/@id")
    parent_category = xml.xpath("//ccg/span[contains(@child, '" + element_id + "')]/@category")[0]
    #print(parent_category)
    if not re.search("^NP\[?", parent_category):
        tmp4 = xml.xpath("//ccg/span[contains(@child, '" + element_id + "')]/@child")
        if len(tmp4) > 0:
            verb_id = tmp4[0].split(" ")
            verb_id.remove(element_id)
            verb_base =  xml.xpath("//ccg/span[contains(@id, '" + element_id + "')]/@base")
            if 'be' in verb_base and "all" in floating_list:
                #floating
                floating_flg = 1
            break
    else:
        element_id = parent_id

list_target_id = element_id.split(" ")
while True:
    childid = []
    for parentid in list_target_id:
        tmp = xml.xpath("//ccg/span[contains(@id, '" + parentid + "')]/@child")
        if len(tmp) > 0:
            childid.extend(tmp[0].split(" "))
    if len(childid) == 0:
        break
    else:
        child_ids.extend(childid)
        list_target_id = childid

# extract the whole VP subtree
list_verb_id = verb_id[0].split(" ")
while True:
    childid = []
    for parentid in list_verb_id:
        tmp5 = xml.xpath("//ccg/span[contains(@id, '" + parentid + "')]/@child")
        if len(tmp5) > 0:
            childid.extend(tmp5[0].split(" "))
    if len(childid) == 0:
        break
    else:
        child_verb_ids.extend(childid)
        list_verb_id = childid


nouns, verbs = [], []
for nounphrase in sorted(child_ids, key=lambda x:int((re.search(r"sp([0-9]+)", x)).group(1))):
    tmp2 = xml.xpath("//ccg/span[@id='" + nounphrase + "']/@surf")
    if len(tmp2) > 0:
        nouns.extend(tmp2)
print(nouns)

for verbphrase in sorted(child_verb_ids, key=lambda x:int((re.search(r"sp([0-9]+)", x)).group(1))):
    tmp3 = xml.xpath("//ccg/span[@id='" + verbphrase + "']/@surf")
    if len(tmp3) > 0:
        verbs.extend(tmp3)

print(verbs)

# print(element.get("pos"))
# print(etree.tostring(element, pretty_print=True).decode("utf-8"))
print(etree.tostring(xml, pretty_print=True).decode("utf-8"))
print(element_id, verb_id)