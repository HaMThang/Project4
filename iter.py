from pdf_struct.feature_extractor import TextContractFeatureExtractor
from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct import feature_extractor
from pdf_struct.core.export import to_tree, to_paragraphs
from pdf_struct.core.predictor import train_classifiers, \
    predict_with_classifiers
from pdf_struct.export.hocr import export_result
import tqdm
import pickle


annos = transition_labels.load_annos('datasets/anno')

FILE_TYPE = 'docx'
documents = loader.modules[FILE_TYPE].load_from_directory('datasets/raw', annos)
assert len(documents) > 0
feature_extractor_cls = feature_extractor.feature_extractors['TextContractFeatureExtractor']
documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]

clf, clf_ptr = train_classifiers(documents)

with open('clf.pkl', 'wb') as file:
    pickle.dump(clf, file)

with open('clf_ptr.pkl', 'wb') as file:
    pickle.dump(clf_ptr, file)

# Now make predictions
document = loader.modules[FILE_TYPE].load_document('datasets/raw/Project1_Lê Hữu Lợi_10120764_124201.docx', None, None)

document = TextContractFeatureExtractor.append_features_to_document(document)

pred = predict_with_classifiers(clf, clf_ptr, [document])[0]

print(to_paragraphs(pred))
#export_result(pred,'datasets/Project1')

