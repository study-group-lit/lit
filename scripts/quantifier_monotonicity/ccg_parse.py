import os
import os.path
import re
import subprocess
from lxml import etree
from typing import List
from nltk.corpus import wordnet as wn
import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
nltk.download("wordnet")
from pattern.en import *
from datasets import load_from_disk
from datasets import load_dataset
import pandas as pd
import inflect
inflect = inflect.engine()
import pattern.text
from pattern.helpers import decode_string

def monkeypatch_pattern():
    from codecs import BOM_UTF8
    BOM_UTF8 = BOM_UTF8.decode("utf-8")
    decode_utf8 = decode_string
    def _read(path, encoding="utf-8", comment=";;;"):
        """Returns an iterator over the lines in the file at the given path,
        strippping comments and decoding each line to Unicode.
        """
        if path:
            if isinstance(path, str) and os.path.exists(path):
                # From file path.
                f = open(path, "r", encoding="utf-8")
            elif isinstance(path, str):
                # From string.
                f = path.splitlines()
            else:
                # From file or buffer.
                f = path
            for i, line in enumerate(f):
                line = line.strip(BOM_UTF8) if i == 0 and isinstance(line, str) else line
                line = line.strip()
                line = decode_utf8(line, encoding)
                if not line or (comment and line.startswith(comment)):
                    continue
                yield line
    pattern.text._read = _read

monkeypatch_pattern()

CANDC_PATH = "bin/candc-1.00"
SED_PATH = "scripts/quantifier_monotonicity/tokenizer.sed"
CANDC2TRANSCCG_PATH = "scripts/quantifier_monotonicity/candc2transccg.py"
results = pd.DataFrame(index=[], columns=['determiner', 'monotonicity', 'gold_label', 'replace_target', 'replace_source', 'replace_mode', 'ori_sentence', 'new_sentence'])


def keep_plurals(noun, newnoun):
    if inflect.singular_noun(noun) is False:
        # singular
        return singularize(newnoun)
    else:
        # plural
        return pluralize(newnoun)


def keep_tenses(verb, newverb):
    ori_tense = tenses(verb)[0]
    ori_tense2 = [x for x in ori_tense if x is not None]
    #print(ori_tense2)
    tense, person, number, mood, aspect = None, None, None, None, None

    if 'infinitive' in ori_tense2:
        tense = INFINITIVE
    elif 'present' in ori_tense2:
        tense = PRESENT
    elif 'past' in ori_tense2:
        tense = PAST
    elif 'future' in ori_tense2:
        tense = FUTURE

    if 1 in ori_tense2:
        person = 1
    elif 2 in ori_tense2:
        person = 2
    elif 3 in ori_tense2:
        person = 3
    else:
        person = None

    if 'singular' in ori_tense2:
        number = SINGULAR
    elif 'plural' in ori_tense2:
        number = PLURAL
    else:
        number = None

    if 'indicative' in ori_tense2:
        mood = INDICATIVE
    elif 'imperative' in ori_tense2:
        mood = IMPERATIVE
    elif 'conditional' in ori_tense2:
        mood = CONDITIONAL
    elif 'subjunctive' in ori_tense2:
        mood = SUBJUNCTIVE
    else:
        mood = None
    
    #if 'imperfective' in ori_tense2:
    #   aspect = IMPERFECTIVE
    #elif 'perfective' in ori_tense2:
    #   aspect = PERFECTIVE
    if 'progressive' in ori_tense2:
        aspect = PROGRESSIVE
    else:
        aspect = None
    newverb_tense = conjugate(newverb, 
        tense = tense,        # INFINITIVE, PRESENT, PAST, FUTURE
       person = person,              # 1, 2, 3 or None
       number = number,       # SG, PL
         mood = mood,     # INDICATIVE, IMPERATIVE, CONDITIONAL, SUBJUNCTIVE
        aspect = aspect,
      negated = False,          # True or False
        parse = True)
    #print(newverb, newverb_tense)
    if newverb_tense is None:
        return newverb
    return newverb_tense


def replace_sentence_WN_nv(determiner, nounmono, verbmono, noun, nounsense, verb, verbsense, sentence, results):
    nounsynset = nounsense
    nounhypernyms = nounsynset.hypernyms() if nounsynset is not None else []
    nounhyponyms = nounsynset.hyponyms() if nounsynset is not None else []
    verbsynset = verbsense
    verbhypernyms = verbsynset.hypernyms() if verbsynset is not None else []
    verbhyponyms = verbsynset.hyponyms() if verbsynset is not None else []

    nounhypersim = [
        nounhypernym.wup_similarity(verbsynset) \
        if verbsynset is not None and nounhypernym.wup_similarity(verbsynset) is not None else 0 \
        for nounhypernym in nounhypernyms
    ]
    nounhyposim = [
        nounhyponym.wup_similarity(verbsynset) \
        if verbsynset is not None and nounhyponym.wup_similarity(verbsynset) is not None else 0 \
        for nounhyponym in nounhyponyms
    ]
    verbhypersim = [
        verbhypernym.wup_similarity(nounsynset) 
        if nounsynset is not None and verbhypernym.wup_similarity(nounsynset) is not None else 0
        for verbhypernym in verbhypernyms
    ]
    verbhyposim = [
        verbhyponym.wup_similarity(nounsynset) 
        if nounsynset is not None and verbhyponym.wup_similarity(nounsynset) is not None else 0
        for verbhyponym in verbhyponyms
    ]
    synsetdict = {}
    if len(nounhypersim) > 0:
        synsetdict["noun_hypernym"] = nounhypernyms[nounhypersim.index(max(nounhypersim))]
    if len(nounhyposim) > 0:
        synsetdict["noun_hyponym"] = nounhyponyms[nounhyposim.index(max(nounhyposim))]
    if len(verbhypersim) > 0:
        synsetdict["verb_hypernym"] = verbhypernyms[verbhypersim.index(max(verbhypersim))]
    if len(verbhyposim) > 0:
        synsetdict["verb_hyponym"] = verbhyponyms[verbhyposim.index(max(verbhyposim))]
    #print(synsetdict)
    contradictiondeterminer = contradiction_mapping[determiner]
    if contradictiondeterminer is None:
        print(f"{contradictiondeterminer} could not be mapped to a contradicting quantifier")
    
    newsentence = re.sub(determiner, contradictiondeterminer, sentence)
    record = pd.Series([contradictiondeterminer, nounmono, "contradiction", determiner, contradictiondeterminer, "quantifier_antonym", sentence, newsentence], index=results.columns)
    results = results.append(record, ignore_index = True)
    for rel, synset in synsetdict.items():
        synsetwords = synset.lemma_names()
        #print(synsetwords)
        for synsetword in synsetwords:
            new_synsetword = re.sub("_", " ", synsetword)
            if re.search("noun", rel):
                newnoun = keep_plurals(noun, new_synsetword)
                newsentence = re.sub(noun, newnoun, sentence)
                gold_label = check_label(nounmono, rel)
                record = pd.Series([determiner, nounmono, gold_label, noun, newnoun, rel, sentence, newsentence], index=results.columns)
                results = results.append(record, ignore_index = True)
                record = pd.Series([determiner, nounmono, rev_label(gold_label, nounmono), noun, newnoun, rel, newsentence, sentence], index=results.columns)
                results = results.append(record, ignore_index = True)
            else:
                newverb = keep_tenses(verb, new_synsetword)
                newsentence = re.sub(verb, newverb, sentence)
                gold_label = check_label(verbmono, rel)
                record = pd.Series([determiner, verbmono, gold_label, verb, newverb, rel, sentence, newsentence], index=results.columns)
                results = results.append(record, ignore_index = True)
                record = pd.Series([determiner, verbmono, rev_label(gold_label, verbmono), verb, newverb, rel, newsentence, sentence], index=results.columns)
                results = results.append(record, ignore_index = True)

    return results


def remove_duplicates(x):
    y=[]
    for i in x:
        if i not in y:
            y.append(i)
    return y


contradiction_mapping = {
    "a": "no",
    "a few": "many",
    "a large number of": "just a small number of",
    "a little": "a high",
    "a number of": "zero",
    "a small number of": "a large number of",
    "all": "a few",
    "any": "all",
    "both": "neither",
    "every": "none",
    "each": "just a few",
    "enough": "insufficiently many",
    "few": "many",
    "fewer": "more",
    "less": "more",
    "lots of": "no more than a few",
    "most": "least",
    "many": "few",
    "many of": "few of",
    "much": "limited amount of",
    "neither": "both",
    "no": "most",
    "none of": "several",
    "not many": "each",
    "not much": "much",
    "never": "sometimes",
    "numerous": "limited amount of",
    "plenty of": "42",
    "several": "just one",
    "some": "zero",
    "this": "that",
    "that": "this",
    "the": "none of",
    "whole": "only a part"
}

def check_monotonicity(determiner):
    nounmono, verbmono = "non_monotone", "non_monotone"
    upward_noun = ["some", "a"]
    upward_verb = ["every", "each", "all", "some", "both", "most", "many", "several", "this", "that", "a", "the"]
    downward_noun = ["every", "each", "all", "no", "neither", "any", "never"]
    downward_verb = ["no", "neither", "any", "never", "few"]
    if determiner in upward_noun:
        nounmono = "upward_monotone"
    if determiner in upward_verb:
        verbmono = "upward_monotone"
    if determiner in downward_noun:
        nounmono = "downward_monotone"
    if determiner in downward_verb:
        verbmono = "downward_monotone"
    return nounmono, verbmono


def replace_sentence(determiner, nounmono, noun, newnoun, sentence, results):
    pat = re.compile(noun)
    newpat = re.compile(newnoun)
    newsentence = re.sub(noun, newnoun, sentence)
    gold_label = check_label(nounmono, 'simple')
    record = pd.Series([determiner, nounmono, gold_label, noun, newnoun, 'simple', sentence, newsentence], index=results.columns)
    record = pd.Series([determiner, nounmono, rev_label(gold_label, nounmono), noun, newnoun, 'simple', newsentence, sentence], index=results.columns)
    results = results.append(record, ignore_index = True)
    return results


def check_label(monotonicity, mode):
    modegroup = ""
    if re.search("hypo", mode):
        modegroup = "down"
    elif re.search("hyper", mode):
        modegroup = "up"
    elif mode == "simple":
        modegroup = "up"
    if monotonicity == "upward_monotone" and modegroup == "up":
        return "entailment"
    elif monotonicity == "upward_monotone" and modegroup == "down":
        return "neutral"
    elif monotonicity == "downward_monotone" and modegroup == "up":
        return "neutral"
    elif monotonicity == "downward_monotone" and modegroup == "down":
        return "entailment"
    else:
        return "neutral"

def rev_label(gold_label, monotonicity):
    #reverse the gold_label
    if monotonicity == "non_monotone":
        return "neutral"
    elif gold_label == "entailment":
        return "neutral"
    elif gold_label == "neutral":
        return "entailment"

def rev_mono(monotonicity):
    #reverse the polarity
    if monotonicity == "non_monotone":
        return "non_monotone"
    elif monotonicity == "downward_monotone":
        return "upward_monotone"
    elif monotonicity == "upward_monotone":
        return "downward_monotone"

def align_quotes(sentence):
    left = True
    skip_next = False
    ret = ""
    for c in sentence:
        ret += c
        if c == "\"":
            if left:
                skip_next = True
            else:
                ret = ret[:-2] + c
                left = not left
        elif skip_next:
            if c == " " and left:
                ret = ret[:-1]
            left = not left
            skip_next = False
    return ret

def parse(sentence: str, determiner: str) -> "XML":
    global results

    sentence = align_quotes(sentence)
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
    nounmono, verbmono = check_monotonicity(determiner)
    floating_list = ["both", "all", "each"]
    floating_flg = 0
    print(sentence, determiner)
    element_id = (etree.XPath(f".//span[@base=\"{determiner}\"]/@id"))(xml)[0]
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
                if element_id in verb_id:
                    verb_id.remove(element_id)
                verb_base =  xml.xpath("//ccg/span[contains(@id, '" + element_id + "')]/@base")
                if 'be' in verb_base and determiner in floating_list:
                    #floating
                    floating_flg = "true"
                break
        else:
            element_id = parent_id[0]

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
    nouns = list(filter(lambda s: len(s) > 0, map(lambda s: ("".join(c for c in s if c.isalnum())).strip(), nouns)))
    print(nouns)

    for verbphrase in sorted(child_verb_ids, key=lambda x:int((re.search(r"sp([0-9]+)", x)).group(1))):
        tmp3 = xml.xpath("//ccg/span[@id='" + verbphrase + "']/@surf")
        if len(tmp3) > 0:
            verbs.extend(tmp3)
    print(verbs)

    if floating_flg == "true":
        # remove floating
        return None, None
    # replace an subjective word by its hypernym and hyponym
    elif len(nouns) > 0 and len(verbs) > 0:
        noun = " ".join(nouns)
        newnoun = nouns[-1]
        print(newnoun)
        newnounpos = xml.xpath("//ccg/span[@surf='" + newnoun + "']/@pos")[0]
        if re.search("^PRP", newnounpos):
            # remove pronouns
            print(": is pronoun\n")
            return
        if re.search("^NNP", newnounpos):
            # replace its specific hypernym if a proper noun exists
            # print(" contains koyumeishi\n")
            # print(nlp(newnoun))
            pass
            #semtag = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='tok' and text()='" + newnoun + "']/following-sibling::tag[@type='sem']/text()")
            #if len(semtag) > 0:
            #    if semtag[0] == "PER" or semtag[0] == "GPO":
            #        newnoun = "someone"
            #    elif semtag[0] == "GPE" or semtag[0] == "GEO":
            #        newnoun = "somewhere"
            #    else:
            #        print(target+" contains other semtag"+semtag[0]+"\n")
            #        newnoun = "something"
            #    results = replace_sentence(determiner, nounmono, noun, newnoun, sentence, results, target)
            #    continue
        if len(nouns) > 2:
            newnewnoun = determiner + " " + nouns[-1]
            results = replace_sentence(determiner, nounmono, noun, newnewnoun, sentence, results)
        verb = " ".join(verbs)
        verb_chunks = xml.xpath("//tokens/token[@chunk='I-VP']/@surf")
        newverb = next(filter(lambda v: v in verb_chunks, verbs))
        #newverb = verbs[-1]
        print()
        print(results)
        #print(etree.tostring(xml, pretty_print=True).decode("utf-8"))

        # replace hypernym and hyponym using senseid
        words = word_tokenize(sentence)
        #nounsense = wn.synsets(newnoun, pos=wn.NOUN)
        #verbsense = wn.synsets(newverb, pos=wn.VERB)
        nounsense = lesk(words, newnoun, 'n')
        verbsense = lesk(words, newverb, 'v')
        print(newnoun, newverb)
        print(newverb, verbsense)
        results = replace_sentence_WN_nv(determiner, nounmono, verbmono, newnoun, nounsense, newverb, verbsense, sentence, results)
        print(nounsense)
        print(verbsense)
        print(results)
    return xml, element_id, verb_id


if __name__ == "__main__":
    xml, element_id, verb_id = parse("all men love a woman", "all")

# print(element.get("pos"))
# print(etree.tostring(element, pretty_print=True).decode("utf-8"))
#print(etree.tostring(xml, pretty_print=True).decode("utf-8"))
#print(element_id, verb_id)