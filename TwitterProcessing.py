import re

class TwitterProcessing:
    """
    This class is used to pre-process tweets.  This centralises the processing to one location.  Add more cleaning steps if necessary.
    """

    def __init__(self, tokeniser, lStopwords):
        """
        Initialise the tokeniser and set of stopwords to use.

        @param tokeniser:
        @param lStopwords:
        """

        self.tokeniser = tokeniser
        self.lStopwords = lStopwords



    def process(self, text):
        """
        Perform the processing.
        @param text: the text (tweet) to process

        @returns: list of (valid) tokens in text
        """

        text = text.lower()
        tokens = self.tokeniser.tokenize(text)
        tokensStripped = [tok.strip() for tok in tokens]

        # pattern for digits
        # the list comprehension in return statement essentially remove all strings of digits or fractions, e.g., 6.15
        regexDigit = re.compile("^\d+\s|\s\d+\s|\s\d+$")
        # regex pattern for http
        regexHttp = re.compile("^http")

        return [tok for tok in tokensStripped if tok not in self.lStopwords and regexDigit.match(tok) == None and regexHttp.match(tok) == None]