import sys
import os
import time

CONTROL_CHARS = ["\x08", "\r", "\x1b"]

def decode(input_string):
    # Initial state
    # String is stored as a list because
    # python forbids the modification of
    # a string
    displayed_string = [] 
    cursor_position = 0
    escape_sequence = None

    # Loop on our input (transitions sequence)
    for character in input_string:

        # Alphanumeric transition
        if character not in CONTROL_CHARS and escape_sequence is None:
            # Add the character to the string
            displayed_string[cursor_position:cursor_position+1] = character 
            # Move the cursor forward
            cursor_position += 1

        # Backward transition
        elif character == "\x08":
            # Move the cursor backward
            cursor_position -= 1
        elif character == "\r":
            cursor_position = ("".join(displayed_string[:cursor_position])).rfind("\n")+1
        elif character == "\x1b":
            escape_sequence = ""
        elif escape_sequence is not None:
            escape_sequence += character
            if escape_sequence == "[A":
                escape_sequence = None
                cursor_position = ("".join(displayed_string[:cursor_position])).rfind("\n")
                cursor_position = ("".join(displayed_string[:cursor_position])).rfind("\n")+1
        else:
            print("{} is not handled by this function".format(repr(character)))

    # We transform our "list" string back to a real string
    return "".join(displayed_string)

file = sys.argv[1]
prev_size = 0

interval = 2
prev_time = time.time()
next_time = prev_time + interval

while True:
    now = time.time()
    while now < next_time:
        time.sleep(max(0.5, next_time - now))
        now = time.time()
    prev_time = now
    next_time = prev_time + interval
    # Check file
    size = os.path.getsize(file)
    if size == prev_size:
        continue
    with open(file, "r", newline='') as f:
        f.seek(prev_size)
        text = f.read(size-prev_size)
        prev_size = size
        print(decode(text))