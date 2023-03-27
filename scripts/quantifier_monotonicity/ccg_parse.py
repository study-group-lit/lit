import os
import subprocess

from lxml import etree
from dataclasses import dataclass

from datasets import load_from_disk
from datasets import load_dataset

@dataclass
class Quantifier:
    name: str
    left_rising: bool
    right_rising: bool

all_quantifiers = [
    Quantifier("a few", True, True),
    Quantifier("a large number of", True, True),
    Quantifier("a little", True, True),
    Quantifier("a number of", True, True),
    Quantifier("a small number of", True, True),
    Quantifier("all", False, True),
    Quantifier("any", False, True),
    Quantifier("enough", True, True),
    Quantifier("each", False, True),
    Quantifier("every", True, True),
    Quantifier("few", False, False),
    Quantifier("fewer", False, False),
    Quantifier("less", False, False),
    Quantifier("lots of", True, True),
    Quantifier("many", True, True),
    Quantifier("most", False, True),
    Quantifier("much", True, True),
    Quantifier("no", False, False),
    Quantifier("none of", False, False),
    Quantifier("not many", False, False),
    Quantifier("not much", False, False),
    Quantifier("numerous", True, True),
    Quantifier("plenty of", True, True),
    Quantifier("several", True, True),
    Quantifier("some", True, True),
    Quantifier("whole", False, True),
    Quantifier("many of", True, True),
]

CANDC_PATH = "bin/candc-1.00"
SED_PATH = "scripts/quantifier_monotonicity/tokenizer.sed"

def parse(sentence: str) -> "XML":
    ps = subprocess.run(
        f"echo \"{sentence}\" | sed -f {SED_PATH} | {CANDC_PATH}/bin/candc --models {CANDC_PATH}/models/ --candc-printer xml", 
        stdout=subprocess.PIPE,
        shell=True
        )
    xml = etree.fromstring(ps.stdout)
    return xml

xml = parse("new : Hans : \" we need a president who can bring us all together \"")
# element = (etree.XPath(".//lf[@word=\"no\"]"))(xml)[0]
# print(element.get("pos"))
# print(etree.tostring(element, pretty_print=True).decode("utf-8"))
print(etree.tostring(xml, pretty_print=True).decode("utf-8"))