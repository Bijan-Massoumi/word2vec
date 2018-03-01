replaceWithWhiteSpace = ["?","!","(",",",".",")","'",'"',":"]


def preprocess(str_in):
    global replaceWithWhiteSpace
    str1_out = str_in.lower()
    for char in replaceWithWhiteSpace:
        str1_out = str1_out.replace(char," ")
    str1_out = filter(lambda x: x != "-", str1_out)
    return str1_out



if __name__ == "__main__":
    test = """love? AMe---'--"--r'ica" asdfeaf"""
    print preprocess(test)
