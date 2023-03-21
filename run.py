'''from pdf_struct.loader import doc, pdf
from pdf_struct.core import transition_labels
from pdf_struct.core import predictor

#annos = transition_labels.load_annos('pdf-struct-dataset/contract_text_en/anno')

#doc.load_from_directory('pdf-struct-dataset/contract_text_en/raw',annos)

predictor.k_fold_train_predict('pdf-struct-dataset/contract_pdf_en/raw/2_waycda.pdf')'''

from typing import List, Optional

import numpy as np

from pdf_struct.core.feature_extractor import BaseFeatureExtractor, \
    single_input_feature, pairwise_feature
from pdf_struct.loader.pdf import TextBox
from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct.core.export import to_tree, to_paragraphs
from pdf_struct.core.predictor import train_classifiers, \
    predict_with_classifiers


class MinimalPDFFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, text_boxes: List[TextBox]):
        '''This class is instantiated for each document.
        The constructor receives text_boxes which is basically all the
        lines of the input document.
        You can calculate document-specific global constants here to use
        in the actual feature extraction.

        text_boxes is list of objects all of which inherit from
        `pdf_struct.core.document.TextBlock`. This will be
        `pdf_struct.loader.pdf.TextBox` if you choose `pdf` in `pdf-struct train`
        or `pdf_struct.loader.text.TextLine` if you choose `text`.
        '''
        bboxes = np.array(
            [tb.bbox for tb in text_boxes])
        page_top = bboxes[:, 3].max()
        page_bottom = bboxes[:, 1].min()
        self.header_thresh = \
            page_top - 0.15 * (page_top - page_bottom)

    @single_input_feature([1])
    def header_region(self, tb: Optional[TextBox]):
        '''A member function with `@single_input_feature` will be called
        for each pair of consecutive text blocks. For each pair, such function
        will be applied to text blocks whose indices are specified in the argument.
        `1` means the line means the first of the pair and `2` means the latter
        `0` is the line before `1` and `3` means the line after `2`.
        The function should return `bool`, `int` or `float`. It can also return
        dict (keys will be appended to the function name to create feature names)
        or list (numbers will be automatically appended).
        tb can be `None` when it is outside the document region (specifying `3`
        will results in `None` towards the end of the document).

        Here, we are classifying whether the first line is in a header region
        as this can be a strong clue when determining the relationship between
        the pair.
        '''
        return bool(tb.bbox[3] > self.header_thresh)

    @pairwise_feature([(0, 1), (1, 2)])
    def page_change(self, tb1: Optional[TextBox], tb2: Optional[TextBox]):
        '''Same as `@single_input_feature` but works on pair of text blocks
        as specified in its argument.
        '''
        if tb1 is None or tb2 is None:
           return True
        return tb1.page != tb2.page



annos = transition_labels.load_annos('pdf-struct-dataset/contract_pdf_en/anno')

FILE_TYPE = 'pdf'
documents = loader.modules[FILE_TYPE].load_from_directory('pdf-struct-dataset/contract_pdf_en/raw', annos)
assert len(documents) > 0

documents = [MinimalPDFFeatureExtractor.append_features_to_document(document)
             for document in documents]

clf, clf_ptr = train_classifiers(documents)

# Now make predictions
document = loader.modules[FILE_TYPE].load_document('pdf-struct-dataset/contract_pdf_en/raw', None, None)
document = MinimalPDFFeatureExtractor.append_features_to_document(document)

pred = predict_with_classifiers(clf, clf_ptr, [document])[0]
print(to_paragraphs(pred))