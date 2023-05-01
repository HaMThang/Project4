import glob
import os
import tqdm
import docx2txt

from pdf_struct import loader
from pdf_struct.loader.doc import TextLine


def create_training_data(in_path, out_path):
    text = docx2txt.process(in_path)
    lines = text.split('\n')
    text_lines = TextLine.from_lines(lines)

    if len(text_lines) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    with open(out_path, 'w', encoding ='utf-8') as fout:
        for line in text_lines:
            fout.write(f'{line.text}\t0\t\n')

def init_dataset(file_type, indir, outdir):
    paths = glob.glob(os.path.join(indir, f'*.{file_type}'))
    os.makedirs(outdir)
    for path in tqdm.tqdm(paths):
        out_filename = os.path.splitext(os.path.basename(path))[0] + '.tsv'
        loader.modules[file_type].create_training_data(path, os.path.join(outdir, out_filename))

create_training_data('datasets/raw/Nhóm 6_Hệ thống gợi ý sản phẩm mua kèm.docx', 'datasets/anno/Nhóm 6_Hệ thống gợi ý sản phẩm mua kèm.tsv')
#init_dataset('docx','datasets/raw','datasets/anno')