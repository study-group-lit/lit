# README
# Testing the sample generation script by feeding a few sample sentences into the generator.
# The results are printed to the console.

from tqdm import tqdm
from ccg_parse import generate_samples

sentences = [
    'Bush says most of Congress " acting like a teenager with a new credit card "',
    "Shaun White is the most successful snowboarder",  # 1
    "O'Neill : no judge on Supreme Court now has legislative background",  # 2
    "three - time winner Boris Becker believes any of the top four players could triumph",  # 3
    "Arun Kundnani : some urge a U.S. government program aimed at extreme Muslim views",  # 4
    "the owner of this estate is no ordinary Lord of the Manor -- it 's Russian tycoon Max",  # 5
    '" this wonderful couple is a danger to no one , " writes Bourdain',  # 6
    "while in prison , Mandela became most significant black leader in South Africa",  # 7
    'new : Winfrey : "we need a president who can bring us all together"',  # 8
    "Brazile : Norquist is the man most responsible for GOP gridlock in Washington",  # 9
    "Earl Jr. believes Tiger has no one to keep him on the right path",  # 10
    "world no. 3 Lee Westwood agrees with the move saying phones are key for business",  # 11
    "there 's no substitute for American leadership in this critical region , he says",  # 12
    "rising tide of Taliban and threat of violence has some residents worried",  # 13
    'Ban : " i can not find any other better suited leader "',
]


def do_sentence(sentence):
    print(generate_samples(sentence))


for sentence in sentences:
    do_sentence(sentence)
