import re
import sys
import nltk
# from nltk.corpus import stopwords 

# create a set of stop words, if we need to use them later for cleaning text
# using spacy.lang.en.stop_words.STOP_WORDS without loading spacy as it is time consuming
stop_words = {
    'be', 'other', 'thru', 'hers', 'one', 'another', 'nothing', 'those',
    'when', 'where', 'if', 'thereupon', 'so', 'really', 'should', 'first',
    'give', 'whom', 'neither', 'could', 'at', 'once', 'doing', 'whereafter',
    'its', 're', 'whether', 'since', 'empty', 'too', 'his', 'last', 'perhaps',
    'ca', 'in', 'now', 'over', 'thereafter', 'front', 'serious', 'else',
    'latterly', 'whereas', 'did', 'itself', 'everything', 'though', 'herself',
    'each', 'forty', 'any', 'have', 'such', 'around', 'towards', 'nowhere',
    'toward', 'latter', 'which', 'yourself', 'with', 'four', 'become', 'own',
    'themselves', 'whereby', 'throughout', 'thus', 'by', 'been', 'only', 'and',
    'nor', 'myself', 'whoever', 'yourselves', 'their', 'cannot', 'beyond',
    'to', 'two', 'bottom', 'can', 'than', 'made', 'below', 'hereby', 'keep',
    'various', 'enough', 'anything', 'also', 'done', 'alone', 'somehow',
    'them', 'him', 'upon', 'we', 'what', 'twenty', 'please', 'indeed',
    'under', 'still', 'elsewhere', 'others', 'were', 'whereupon', 'why',
    'behind', 'within', 'they', 'because', 'anyhow', 'quite', 'call',
    'everyone', 'get', 'formerly', 'is', 'former', 'himself', 'ourselves',
    'she', 'used', 'due', 'unless', 'ever', 'both', 'herein', 'rather',
    'almost', 'or', 'six', 'on', 'from', 'there',  'therefore', 'does',
    'more', 'fifteen', 'between', 'although', 'noone', 'meanwhile', 'nine',
    'least', 'onto', 'wherein', 'hereafter', 'anyway', 'seeming', 'up', 'the',
    'our', 'before', 'who', 'few', 'here', 'always', 'most', 'third', 'yet',
    'he', 'moreover', 'hereupon', 'sixty', 'otherwise', 'an', 'every', 'none',
    'not', 'ten', 'three', 'off', 'whose', 'you', 'nevertheless', 'your',
    'some', 'no', 'full', 'might', 'do', 'out', 'of', 'less', 'just',
    'wherever', 'say', 'these', 'whither', 'eleven', 'namely', 'until',
    'anyone', 'beside', 'show', 'except', 'already', 'whole', 'yours', 'using',
    'across', 'during', 'make', 'per', 'while', 'next', 'many', 'becoming',
    'via', 'nobody', 'hundred', 'see', 'together', 'into', 'above', 'sometime',
    'afterwards', 'down', 'but', 'go', 'against', 'hence', 'seemed', 'besides',
    'then', 'top', 'whenever', 'without', 'had', 'may', 'anywhere', 'a', 'put',
    'further', 'whence', 'became', 'seem', 'how', 'mostly', 'five', 'this',
    'amount', 'regarding', 'side', 'much', 'fifty', 'all', 'i', 'amongst',
    'as', 'move', 'even', 'thence', 'through', 'same', 'someone', 'never',
    'would', 'somewhere', 'will', 'that', 'it', 'beforehand', 'are',
    'everywhere', 'being', 'very', 'well', 'however', 'again', 'me', 'must',
    'after', 'therein', 'take', 'am', 'was', 'eight', 'either', 'mine',
    'something', 'us', 'becomes', 'often', 'for', 'my', 'seems', 'her',
    'sometimes', 'whatever', 'about', 'along', 'back', 'ours', 'several',
    'thereby', 'has', 'among', 'twelve','number'}


def clean_text_string(input: str, remove_stop_words: bool=True):
    '''
    This is to clean the text: remove everything other than alphabets.
    For eg., 10. Comment ... how nice!! : will become comment how nice
    >>> clean_text_for_comparison('Wow... Loved this place.', True)
    'wow loved place'
    >>> clean_text_for_comparison('10. Serial/Batch No.:', False)
    'serial batch no'
    '''
    
    regex = re.compile('[^a-zA-Z]')
    result = regex.sub(' ', input).lstrip().rstrip().lower()
    if(remove_stop_words):
        result = ' '.join(
            [r for r in result.split()
             if not stop_words.__contains__(r) and len(r) != 1])

    return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()
