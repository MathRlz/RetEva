from jiwer import cer, wer, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces, Compose

normalize = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces()
])

def word_error_rate(ref: str, hyp: str) -> float:
    ref = normalize(ref)
    hyp = normalize(hyp)
    return wer(ref, hyp)

def character_error_rate(ref: str, hyp: str) -> float:
    ref = normalize(ref)
    hyp = normalize(hyp)
    return cer(ref, hyp)

