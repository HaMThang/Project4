import docx2txt

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